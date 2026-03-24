from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F
from diffusers import (
    AutoencoderKLCogVideoX,
    CogVideoXDPMScheduler,
    CogVideoXPipeline,
    CogVideoXTransformer3DModel,
    CogVideoXImageToVideoPipeline,
)
from diffusers.models.embeddings import get_3d_rotary_pos_embed
from PIL import Image
from transformers import AutoTokenizer, T5EncoderModel
from typing_extensions import override

from finetune.schemas import Components
from finetune.trainer import Trainer
from finetune.utils import unwrap_model
from finetune.datasets.ref_real_sr_dataset import RefRealSRDataset
from finetune.utils.ref_utils import get_ref_frames_api
from ..utils import register

from accelerate.logging import get_logger
from finetune.constants import LOG_LEVEL, LOG_NAME
import math
from torchvision import transforms
from diffusers.pipelines.cogvideo.pipeline_output import CogVideoXPipelineOutput

logger = get_logger(LOG_NAME, LOG_LEVEL)

class KFVSRS1RefTrainer(Trainer):
    UNLOAD_LIST = ["text_encoder", "vae"]

    @override
    def load_components(self) -> Components:
        components = Components()
        model_path = str(self.args.model_path)

        # I2V Pipeline class usually, but Trainer uses Components struct. 
        # Components class fields are fixed.
        components.pipeline_cls = CogVideoXImageToVideoPipeline 

        components.tokenizer = AutoTokenizer.from_pretrained(model_path, subfolder="tokenizer")

        components.text_encoder = T5EncoderModel.from_pretrained(
            model_path, subfolder="text_encoder"
        )

        components.transformer = CogVideoXTransformer3DModel.from_pretrained(
            model_path, subfolder="transformer"
        )

        components.vae = AutoencoderKLCogVideoX.from_pretrained(model_path, subfolder="vae")

        components.scheduler = CogVideoXDPMScheduler.from_pretrained(
            model_path, subfolder="scheduler"
        )
        
        # Verify transformer channels for I2V (should be 32)
        if components.transformer.config.in_channels != 32:
             logger.warning(f"Expected transformer in_channels to be 32 for I2V, but got {components.transformer.config.in_channels}. Proceeding, but check model path.")

        return components

    @override
    def initialize_pipeline(self) -> CogVideoXImageToVideoPipeline:
        pipe = CogVideoXImageToVideoPipeline(
            tokenizer=self.components.tokenizer,
            text_encoder=self.components.text_encoder,
            vae=self.components.vae,
            transformer=unwrap_model(self.accelerator, self.components.transformer),
            scheduler=self.components.scheduler,
        )
        return pipe
    
    @override
    def prepare_dataset(self) -> None:
        self.dataset = RefRealSRDataset(
            **(self.args.model_dump()),
            device=self.accelerator.device,
            max_num_frames=self.state.train_frames,
            height=self.state.train_height,
            width=self.state.train_width,
            trainer=self,
        )
        
        # Prepare VAE and text encoder 
        self.components.vae.requires_grad_(False)
        self.components.text_encoder.requires_grad_(False)
        self.components.vae = self.components.vae.to(
            self.accelerator.device, dtype=self.state.weight_dtype
        )
        self.components.text_encoder = self.components.text_encoder.to(
            self.accelerator.device, dtype=self.state.weight_dtype
        )
        
        self.data_loader = torch.utils.data.DataLoader(
            self.dataset,
            collate_fn=self.collate_fn,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            pin_memory=self.args.pin_memory,
            shuffle=True,
        )

    @override
    def encode_video(self, video: torch.Tensor) -> torch.Tensor:
        vae = self.components.vae
        video = video.to(vae.device, dtype=vae.dtype)
        latent_dist = vae.encode(video).latent_dist
        latent = latent_dist.sample() * vae.config.scaling_factor
        return latent

    @override
    def encode_text(self, prompt: str) -> torch.Tensor:
        prompt_token_ids = self.components.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.state.transformer_config.max_text_seq_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        prompt_token_ids = prompt_token_ids.input_ids
        prompt_embedding = self.components.text_encoder(
            prompt_token_ids.to(self.accelerator.device)
        )[0]
        return prompt_embedding
    
    @override
    def collate_fn(self, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Trainer.collate_fn raises NotImplementedError, so we must not call it.
        
        # We need RefRealSRDataset output keys: 
        # "hq_video", "lq_video", "prompt", "prompt_embedding", "video_metadata", "ref_video", "ref_indices"
        
        ret = {
            "hq_videos": [], 
            "lq_videos": [], 
            "prompts": [], 
            "prompt_embeddings": [], 
            "video_metadatas": [], 
            "ref_frames": [], # Renamed from ref_videos to match content (List of [C, R, H, W])
            "ref_indices": [],
            "encoded_lq_videos": [],
            "encoded_hq_videos": []
        }

        for sample in samples:
            ret["hq_videos"].append(sample["hq_video"])
            ret["lq_videos"].append(sample["lq_video"])
            ret["prompts"].append(sample["prompt"])
            ret["prompt_embeddings"].append(sample["prompt_embedding"])
            ret["video_metadatas"].append(sample["video_metadata"])
            
            # Ref specific
            # Dataset returns "ref_video" key, but it contains frames [C, R, H, W]
            ret["ref_frames"].append(sample.get("ref_video"))
            ret["ref_indices"].append(sample.get("ref_indices"))

            if sample.get("encoded_hq_video") is not None:
                ret["encoded_hq_videos"].append(sample["encoded_hq_video"])
            if sample.get("encoded_lq_video") is not None:
                ret["encoded_lq_videos"].append(sample["encoded_lq_video"])

        ret["hq_videos"] = torch.stack(ret["hq_videos"])
        ret["lq_videos"] = torch.stack(ret["lq_videos"])
        ret["prompt_embeddings"] = torch.stack(ret["prompt_embeddings"])
        
        if len(ret["encoded_hq_videos"]) > 0:
            ret["encoded_hq_videos"] = torch.stack(ret["encoded_hq_videos"])
        if len(ret["encoded_lq_videos"]) > 0:
            ret["encoded_lq_videos"] = torch.stack(ret["encoded_lq_videos"])

        return ret

    def _prepare_ref_latent_i2v(self, batch, lq_latent_shape, dtype, device):
        # batch["ref_frames"] is a list of [C, R, H, W] tensors (or None), where R is num_ref_frames
        # batch["ref_indices"] is a list of [R] indices
        ref_frames_batch = batch.get("ref_frames")
        ref_indices = batch.get("ref_indices")
        
        batch_size = lq_latent_shape[0]
        C, F, H, W = lq_latent_shape[1], lq_latent_shape[2], lq_latent_shape[3], lq_latent_shape[4]
        
        # Initialize full zero padding for reference channels [B, C, F, H, W]
        # We will fill in the specific temporal positions where we have reference frames
        full_ref_latent = torch.zeros((batch_size, C, F, H, W), device=device, dtype=dtype)
        
        for b in range(batch_size):
            # 1. Identify valid Reference Frames for this batch item
            r_frames = None
            indices = None
            
            # Case A: Explicit Reference Frames from Dataset
            if ref_frames_batch is not None and len(ref_frames_batch) > b:
                if ref_frames_batch[b] is not None:
                    r_frames = ref_frames_batch[b] # [C, R, H, W]
                    indices = ref_indices[b] # [R] (Tensor or list)
            
            # Case B: Fallback (Use first frame of HQ video as Ref at index 0)
            if r_frames is None:
                # [C, 1, H, W]
                r_frames = batch["hq_videos"][b:b+1, :, 0, :, :].squeeze(0).unsqueeze(1) 
                indices = [0]
            
            # Handle Tensor vs List for indices
            if isinstance(indices, torch.Tensor):
                indices = indices.tolist()
            
            # r_frames is [C, R, H, W]
            # Ensure it is on device
            r_frames = r_frames.to(device=device, dtype=self.components.vae.dtype)
            R = r_frames.shape[1]
            
            for i in range(R):
                # Get single frame [C, H, W]
                # ref_frame is in [-1, 1], shape [1, C, H, W]
                ref_frame = r_frames[:, i, :, :].unsqueeze(0) 

                # --- 1. Stronger Augmentation for Reference Frames ---
                # Convert to [0, 1] for torchvision transforms
                ref_aug = (ref_frame * 0.5 + 0.5).clamp(0, 1)
                
                # A. Color Jitter (Moderate)
                # Apply with 50% probability to avoid drifting too much
                if torch.rand(1) < 0.5:
                    jitter = transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.0)
                    ref_aug = jitter(ref_aug)
                
                # B. Gaussian Blur (Weak-Moderate)
                # Apply with 30% probability
                if torch.rand(1) < 0.3:
                    # Kernel size must be odd
                    blur = transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))
                    ref_aug = blur(ref_aug)
                
                # Back to [-1, 1]
                ref_frame_aug = (ref_aug * 2.0 - 1.0).clamp(-1, 1)

                # C. Gaussian Noise (Always applied slightly)
                # Noise level: sigma ~ 0.05 seems appropriate for latent training
                noise = torch.randn_like(ref_frame_aug) * 0.05
                ref_frame_aug = ref_frame_aug + noise
                
                # Use augmented frame for encoding
                ref_frame = ref_frame_aug
                
                # Add Noise (I2V augmentation)
                # image_noise_sigma = torch.normal(mean=-3.0, std=0.5, size=(1,), device=device)
                # image_noise_sigma = torch.exp(image_noise_sigma).to(dtype=ref_frame.dtype)
                # noisy_ref = ref_frame + torch.randn_like(ref_frame) * image_noise_sigma[:, None, None, None]
                
                # VAE Encode
                # Input to VAE must be 5D [B, C, F, H, W] even for single frame
                # latent_dist = self.components.vae.encode(noisy_ref.unsqueeze(2)).latent_dist
                latent_dist = self.components.vae.encode(ref_frame.unsqueeze(2)).latent_dist
                lat = latent_dist.sample() * self.components.vae.config.scaling_factor
                # lat: [1, C, F_lat, H_lat, W_lat] -> [1, C, 1, H, W] usually (temporal compression might make F_lat 1 if F=1)
                
                # Check lat shape. If F_lat is 1, correct. 
                # lat[0, :, 0, :, :] gets the first frame features.
                
                # Place in full_ref_latent at correct temporal index
                # Map video frame index to latent index
                try: 
                    vid_idx = indices[i]
                except IndexError:
                    vid_idx = 0
                    
                target_idx = vid_idx // 4 
                
                if target_idx < F:
                    full_ref_latent[b, :, target_idx, :, :] = lat[0, :, 0, :, :]
                    
        return full_ref_latent


    @override
    def compute_loss(self, batch) -> torch.Tensor:
        prompt_embedding = batch["prompt_embeddings"]
        
        if self.args.is_latent:
            lq_latent = batch["encoded_lq_videos"]
            hq_latent = batch["encoded_hq_videos"]
        else:
             with torch.no_grad():
                self.components.vae.to(self.accelerator.device)
                lq_videos = batch["lq_videos"].to(self.accelerator.device)
                hq_videos = batch["hq_videos"].to(self.accelerator.device)
                lq_latent = self.encode_video(lq_videos)
                hq_latent = self.encode_video(hq_videos)

        ref_latent = self._prepare_ref_latent_i2v(batch, lq_latent.shape, lq_latent.dtype, self.accelerator.device)
        
        input_latent = torch.cat([lq_latent, ref_latent], dim=1) 
        
        patch_size_t = self.state.transformer_config.patch_size_t
        if patch_size_t is not None:
            ncopy = input_latent.shape[2] % patch_size_t
            input_first_frame = input_latent[:, :, :1, :, :]
            input_latent = torch.cat([input_first_frame.repeat(1, 1, ncopy, 1, 1), input_latent], dim=2)
            
            hq_first_frame = hq_latent[:, :, :1, :, :]
            hq_latent = torch.cat([hq_first_frame.repeat(1, 1, ncopy, 1, 1), hq_latent], dim=2)
            
        input_latent = input_latent.permute(0, 2, 1, 3, 4) 
        reshape_hq_latent = hq_latent.permute(0, 2, 1, 3, 4) 
        
        batch_size, num_frames, num_channels, height, width = input_latent.shape
        
        _, seq_len, _ = prompt_embedding.shape
        prompt_embedding = prompt_embedding.view(batch_size, seq_len, -1).to(dtype=input_latent.dtype)
        
        if self.args.noise_step != 0:
             # Add noise to LQ part [B, F, C, H, W]
             lq_part = input_latent[:, :, :16, :, :]
             ref_part = input_latent[:, :, 16:, :, :]
             
             noise = torch.randn_like(lq_part)
             add_timesteps = torch.full(
                (batch_size,),
                fill_value=self.args.noise_step,
                dtype=torch.long,
                device=self.accelerator.device,
            )
             lq_part_noisy = self.components.scheduler.add_noise(lq_part.transpose(1, 2), noise.transpose(1, 2), add_timesteps).transpose(1, 2)
             input_latent = torch.cat([lq_part_noisy, ref_part], dim=2)
             
        timesteps = torch.full(
            (batch_size,),
            fill_value=self.args.sr_noise_step,
            dtype=torch.long,
            device=self.accelerator.device,
        )
        
        vae_scale_factor_spatial = 2 ** (len(self.components.vae.config.block_out_channels) - 1)
        transformer_config = self.state.transformer_config
        rotary_emb = (
            self.prepare_rotary_positional_embeddings(
                height=height * vae_scale_factor_spatial,
                width=width * vae_scale_factor_spatial,
                num_frames=num_frames,
                transformer_config=transformer_config,
                vae_scale_factor_spatial=vae_scale_factor_spatial,
                device=self.accelerator.device,
            )
            if transformer_config.use_rotary_positional_embeddings
            else None
        )
        
        if self.state.transformer_config.ofs_embed_dim is not None:
            ofs = torch.full((batch_size,), fill_value=2.0, device=self.accelerator.device, dtype=input_latent.dtype)
        else:
            ofs = None

        predicted_noise = self.components.transformer(
            hidden_states=input_latent,
            encoder_hidden_states=prompt_embedding,
            timestep=timesteps,
            image_rotary_emb=rotary_emb,
            ofs=ofs,
            return_dict=False,
        )[0]
        
        # Take only the first 16 channels (video part) and transpose to [B, C, F, H, W]
        predicted_noise = predicted_noise[:, :, :16, :, :].transpose(1, 2)
        
        lq_sample = input_latent[:, :, :16, :, :].transpose(1, 2) 
        latent_pred = self.components.scheduler.get_velocity(
            predicted_noise, lq_sample, timesteps
        )
        
        loss = F.mse_loss(latent_pred.float(), reshape_hq_latent.transpose(1, 2).float(), reduction="mean")
        return loss

    @override
    def prepare_for_validation(self):
        super().prepare_for_validation()
        
        # 1. API Frames Config
        api_output_dir = self.args.output_dir / "validation_api_refs"
        self.state.api_ref_dir = api_output_dir
        
        # 2. GT Frames Config (New)
        gt_output_dir = self.args.output_dir / "validation_gt_refs"
        self.state.gt_ref_dir = gt_output_dir 
        
        if self.accelerator.is_main_process:
            from finetune.utils.ref_utils import get_ref_frames_api, save_ref_frames_locally
            from pathlib import Path
            
            api_output_dir.mkdir(parents=True, exist_ok=True)
            gt_output_dir.mkdir(parents=True, exist_ok=True) # Prepare GT dir
            
            logger.info(f"[DualVal] Pre-fetching Frames. API: {api_output_dir}, GT: {gt_output_dir}")
            
            video_paths = self.state.validation_videos 
            for video_path in video_paths:
                if not isinstance(video_path, (str, Path)): continue 
                video_name = Path(video_path).stem
                
                t_frames = None if self.args.raw_test else self.state.train_frames
                
                # A. API
                try:
                    get_ref_frames_api(
                        video_path=str(video_path), 
                        output_dir=str(api_output_dir), 
                        video_id=video_name, 
                        target_frames=t_frames,
                        is_match=self.args.raw_test
                    )
                except Exception as e:
                    logger.warning(f"Failed to get API refs for {video_name}: {e}")
                    
                # B. GT (New)
                try:
                     save_ref_frames_locally(
                        video_path=str(video_path),
                        output_dir=str(gt_output_dir),
                        video_id=video_name,
                        target_frames=t_frames,
                        is_match=self.args.raw_test
                     )
                except Exception as e:
                     logger.warning(f"Failed to save GT refs locally for {video_name}: {e}")
        
        self.accelerator.wait_for_everyone()

    @override
    def validation_step(
        self, eval_data: Dict[str, Any], pipe: CogVideoXPipeline
    ) -> List[Tuple[str, Image.Image | List[Image.Image]]]:
        from torchvision import transforms
        from PIL import Image
        prompt, video = eval_data["prompt"], eval_data["video_tensor"]
        ref_video_gt = eval_data.get("ref_video")
        video_name = eval_data.get("video_name", str(self.accelerator.process_index))
        
        if video.ndim == 4:
            video = video.unsqueeze(0)

        b, c, f, h, w = video.shape
        
        if c > f and c == 3:
             # Already [B, C, F, H, W]
             pass
        elif f == 3 and c > f:
             # [B, F, C, H, W] -> permute to [B, C, F, H, W]
             video = video.permute(0, 2, 1, 3, 4)
             b, c, f, h, w = video.shape

        H_, W_ = h, w
        video_reshaped = video.permute(0, 2, 1, 3, 4).reshape(-1, c, h, w) # [B*F, C, H, W]
        video_up = torch.nn.functional.interpolate(video_reshaped, size=(H_*4, W_*4), mode="bilinear", align_corners=False)
        video_up = (video_up / 255.0 * 2.0) - 1.0
        # Restore [B, C, F, H, W]
        video_up = video_up.reshape(b, f, c, H_*4, W_*4).permute(0, 2, 1, 3, 4)
        
        # Prepare Reference Sources
        ref_sources = {}
        
        # 1. GT Reference
        # Check if local GT frames exist first (New Request)
        gt_ref_dir_status = "NOT SET"
        loaded_gt_frames = []
        use_cached_gt = False
        
        if hasattr(self.state, 'gt_ref_dir') and self.state.gt_ref_dir is not None:
             gt_ref_dir_status = str(self.state.gt_ref_dir)
             ref_sources["video"] = "GT_CACHED_PLACEHOLDER"
             use_cached_gt = True
        elif ref_video_gt is not None:
            ref_sources["video"] = ref_video_gt # Fallback to passed Tensor
        else:
            ref_sources["video"] = video_up[0].permute(1, 0, 2, 3) # Last fallback
            
        # 2. API Reference
        api_ref_dir_status = "NOT SET"
        if hasattr(self.state, 'api_ref_dir') and self.state.api_ref_dir is not None:
             api_ref_dir_status = str(self.state.api_ref_dir)
             ref_sources["video_api"] = "API_PLACEHOLDER"
        
        logger.info(f"[DualVal] Sample: {video_name}, GT_Dir: {gt_ref_dir_status}, API_Dir: {api_ref_dir_status}")

        ret_artifacts = []
        
        for ref_type, ref_source in ref_sources.items():
            run_inference = True
            current_ref_frames = None
            
            # Select Indices Logic
            # Note: For cached/API placeholders, we assume indices are standard [0, mid, end]
            # based on target_frames logic shared in prep.
            # We must recalculate expected indices to know filenames.
            
            if ref_type == 'video' and isinstance(ref_source, torch.Tensor):
                 num_frames_video = ref_source.shape[0]
            else:
                 # If placeholder, we derive from video_up (inference video)
                 num_frames_video = video_up.shape[2] 
            
            # if num_frames_video >= 3:
            #     ref_idx_list = [0, num_frames_video // 2, num_frames_video - 1]
            # elif num_frames_video == 2:
            #     ref_idx_list = [0, 1]
            # elif num_frames_video == 1:
            #     ref_idx_list = [0]
            # else:
            ref_idx_list = []

            # --- Loading Logic ---
            if ref_type == "video_api":
                 # Load API frames
                 api_frames_dir = self.state.api_ref_dir / "temp_frames_output"
                 loaded_frames = []
                 for idx in ref_idx_list:
                      fname = f"{video_name}_frame_{idx:05d}.png"
                      fpath = api_frames_dir / fname
                      if fpath.exists():
                           logger.info(f"[DualVal] Found API frame: {fpath}")
                           from PIL import Image
                           img = Image.open(fpath).convert("RGB")
                           
                           # Resize to match GT spatial dimensions
                           target_h, target_w = video_up.shape[-2:]
                           if img.size != (target_w, target_h):
                                logger.info(f"[DualVal] Resizing API frame from {img.size} to {(target_w, target_h)}")
                                img = img.resize((target_w, target_h), Image.LANCZOS)
                                
                           t_img = transforms.ToTensor()(img)
                           t_img = (t_img * 2.0) - 1.0
                           loaded_frames.append(t_img.to(self.accelerator.device, dtype=self.components.vae.dtype))
                      else:
                           logger.warning(f"[DualVal] API frame NOT found at {fpath}. Skipping API pass.")
                           run_inference = False
                           break
                 if run_inference:
                      current_ref_frames = loaded_frames
            
            elif ref_type == "video" and use_cached_gt:
                 # Load GT frames from local cache
                 gt_frames_dir = self.state.gt_ref_dir
                 loaded_frames = []
                 
                 # Try loading
                 all_found = True
                 for idx in ref_idx_list:
                      fname = f"{video_name}_frame_{idx:05d}.png"
                      fpath = gt_frames_dir / fname
                      if fpath.exists():
                           logger.info(f"[DualVal] Found Cached GT frame: {fpath}")
                           from PIL import Image
                           img = Image.open(fpath).convert("RGB")
                           
                           # Resize to match GT spatial dimensions (Just in case)
                           target_h, target_w = video_up.shape[-2:]
                           if img.size != (target_w, target_h):
                                img = img.resize((target_w, target_h), Image.LANCZOS)
                                
                           t_img = transforms.ToTensor()(img)
                           t_img = (t_img * 2.0) - 1.0
                           loaded_frames.append(t_img.to(self.accelerator.device, dtype=self.components.vae.dtype))
                      else:
                           logger.warning(f"[DualVal] Cached GT frame NOT found at {fpath}")
                           all_found = False
                           break
                 
                 if all_found:
                      current_ref_frames = loaded_frames
                 else:
                      # If not found, Fallback to tensor logic if available (ref_video_gt)
                      logger.warning("[DualVal] Falling back to Tensor extraction for GT.")
                      if ref_video_gt is not None:
                           current_ref_frames = []
                           for idx in ref_idx_list:
                                r_frame = ref_video_gt[idx].to(self.accelerator.device, dtype=self.components.vae.dtype)
                                
                                # Normalize if 0-255
                                if r_frame.max() > 1.05:
                                    r_frame = r_frame / 255.0

                                if r_frame.min() >= 0.0 and r_frame.max() <= 1.0:
                                       r_frame = (r_frame * 2.0) - 1.0
                                if r_frame.dim() == 3 and r_frame.shape[-2:] != video_up.shape[-2:]:
                                       r_frame = torch.nn.functional.interpolate(r_frame.unsqueeze(0), size=video_up.shape[-2:], mode="bilinear", align_corners=False).squeeze(0)
                                current_ref_frames.append(r_frame)
                      else:
                           logger.error("[DualVal] No GT source available. Skipping GT pass.")
                           run_inference = False
            
            else:
                 # GT Logic (Original Tensor)
                 current_ref_frames = []
                 
                 # Logic if use_cached_gt was false (e.g. dir not set)
                 source_tensor = ref_video_gt if ref_video_gt is not None else ref_source # handle fallback
                 if isinstance(source_tensor, torch.Tensor):
                     for idx in ref_idx_list:
                          r_frame = source_tensor[idx].to(self.accelerator.device, dtype=self.components.vae.dtype)
                          
                          if r_frame.max() > 1.05:
                                r_frame = r_frame / 255.0
                                
                          if r_frame.min() >= 0.0 and r_frame.max() <= 1.0:
                                 r_frame = (r_frame * 2.0) - 1.0
                          if r_frame.dim() == 3 and r_frame.shape[-2:] != video_up.shape[-2:]:
                                 r_frame = torch.nn.functional.interpolate(r_frame.unsqueeze(0), size=video_up.shape[-2:], mode="bilinear", align_corners=False).squeeze(0)
                          current_ref_frames.append(r_frame)
                 else:
                      logger.warning("Unreachable state: GT source is not tensor and not cached.")
                      run_inference = False

            if not run_inference: 
                logger.info(f"[DualVal] Pass {ref_type} skipped.")
                continue
            


            logger.info(f"[DualVal] Pass {ref_type} starting inference...")

            # --- Inference Core (Shared) ---
            self.components.vae.to(self.accelerator.device)
            lq_latent = self.encode_video(video_up.to(self.accelerator.device, dtype=self.components.vae.dtype)) # LQ Latent
            
            B, C, F_lat, H_lat, W_lat = lq_latent.shape
            
            # --- Conditioned Reference Latent Preparation ---
            full_ref_latent = torch.zeros_like(lq_latent)
            mask_latent = torch.zeros((B, 1, F_lat, H_lat, W_lat), device=lq_latent.device, dtype=lq_latent.dtype)
            
            for i, idx_in_video in enumerate(ref_idx_list):
                if i >= len(current_ref_frames): break
                r_frame = current_ref_frames[i]
                
                # Encode (Clean)
                chunk = r_frame.unsqueeze(0).repeat(4, 1, 1, 1).unsqueeze(0) 
                chunk = chunk.transpose(1, 2) # [1, C, 4, H, W]
                lat = self.encode_video(chunk)
                
                # Map to target latent index
                target_idx = idx_in_video // 4
                
                if target_idx < F_lat:
                    full_ref_latent[:, :, target_idx, :, :] = lat[0, :, 0, :, :]
                    mask_latent[:, :, target_idx, :, :] = 1.0 
            
            do_classifier_free_guidance = self.args.ref_guidance_scale > 1.0
            
            # Prepare Input Latents
            if do_classifier_free_guidance:
                # Cond
                input_latent_cond = torch.cat([lq_latent, full_ref_latent], dim=1)
                # Uncond
                uncond_ref_latent = torch.zeros_like(full_ref_latent)
                input_latent_uncond = torch.cat([lq_latent, uncond_ref_latent], dim=1)
                
                input_latent = torch.cat([input_latent_uncond, input_latent_cond], dim=0) # [2*B, C*2, F, H, W]
            else:
                input_latent = torch.cat([lq_latent, full_ref_latent], dim=1) 
            
            patch_size_t = self.state.transformer_config.patch_size_t
            ncopy = 0
            
            # Handle Temporal Patch Padding
            if patch_size_t is not None:
                 ncopy = input_latent.shape[2] % patch_size_t
                 f_copy = input_latent[:, :, :1, :, :]
                 input_latent = torch.cat([f_copy.repeat(1, 1, ncopy, 1, 1), input_latent], dim=2)
                 
            latents = input_latent.permute(0, 2, 1, 3, 4) # [B (or 2B), F, 32, H, W]
            
            self.components.text_encoder.to(self.accelerator.device)
            prompt_emb = self.encode_text(prompt)
            prompt_emb = prompt_emb.view(B, prompt_emb.shape[1], -1).to(dtype=latents.dtype)
            
            if do_classifier_free_guidance:
                prompt_emb = torch.cat([prompt_emb, prompt_emb], dim=0)
            
            t = torch.full((latents.shape[0],), self.args.sr_noise_step, dtype=torch.long, device=self.accelerator.device)
            
            rotary_emb = self.prepare_rotary_positional_embeddings(
                height=H_lat * 2 ** (len(self.components.vae.config.block_out_channels) - 1), 
                width=W_lat * 2 ** (len(self.components.vae.config.block_out_channels) - 1),
                num_frames=latents.shape[1],
                transformer_config=self.state.transformer_config,
                vae_scale_factor_spatial=2 ** (len(self.components.vae.config.block_out_channels) - 1),
                device=self.accelerator.device
            ) if self.state.transformer_config.use_rotary_positional_embeddings else None
            
            if self.state.transformer_config.ofs_embed_dim is not None:
                ofs = torch.full((latents.shape[0],), fill_value=2.0, device=self.accelerator.device, dtype=latents.dtype)
            else:
                ofs = None

            predicted_noise = self.components.transformer(
                hidden_states=latents,
                encoder_hidden_states=prompt_emb,
                timestep=t,
                image_rotary_emb=rotary_emb,
                ofs=ofs,
                return_dict=False
            )[0]
            
            # Extract actual sample (LQ part)
            # latents is [2B, F, C_total, H, W]
            # lq_sample should base on real lq_latent
            lq_sample = latents[:, :, :16, :, :] # Noisy LQ [B or 2B, F, 16, H, W]
            
            # Predict noise for LQ part and transpose to [B, C, F, H, W]
            predicted_noise_slice = predicted_noise[:, :, :16, :, :].transpose(1, 2)
            
            # Apply Guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_cond = predicted_noise_slice.chunk(2)
                predicted_noise_slice = noise_pred_uncond + self.args.ref_guidance_scale * (noise_pred_cond - noise_pred_uncond)
                
                # Also split lq_sample to match batch size B for denoise
                lq_sample_cond = lq_sample.chunk(2)[1] # Take cond part (same as uncond anyway)
                lq_sample = lq_sample_cond
                
                # t needs to be size B now
                t = t.chunk(2)[0]

            # Denoise
            generated_latents = self.components.scheduler.get_velocity(predicted_noise_slice, lq_sample.transpose(1, 2), t)
            
            if ncopy > 0:
                generated_latents = generated_latents[:, :, ncopy:, :, :]
            
            video_gen = self.components.vae.decode(generated_latents / self.components.vae.config.scaling_factor).sample
            
            video_gen = (video_gen / 2 + 0.5).clamp(0, 1)
            video_gen = video_gen.cpu().permute(0, 2, 3, 4, 1).float().numpy() 
            video_gen = video_gen[0] 
            
            frames = [Image.fromarray((frame * 255).astype('uint8')) for frame in video_gen]
            
            # Append to results with specific key
            artifact_key = "video" if ref_type == "video" else "video_api"
            ret_artifacts.append((artifact_key, frames))
            logger.info(f"[DualVal] Pass {ref_type} completed (Scale={self.args.ref_guidance_scale}).")

        return ret_artifacts

    # Helper methods copied from S1 Trainer
    def get_resize_crop_region_for_grid(self, src, tgt_width, tgt_height):
        tw = tgt_width
        th = tgt_height
        h, w = src
        r = h / w
        if r > (th / tw):
            resize_height = th
            resize_width = int(round(th / h * w))
        else:
            resize_width = tw
            resize_height = int(round(tw / w * h))

        crop_top = int(round((th - resize_height) / 2.0))
        crop_left = int(round((tw - resize_width) / 2.0))

        return (crop_top, crop_left), (crop_top + resize_height, crop_left + resize_width)

    def prepare_rotary_positional_embeddings(
        self,
        height: int,
        width: int,
        num_frames: int,
        transformer_config: Dict,
        vae_scale_factor_spatial: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
    
        grid_height = height // (vae_scale_factor_spatial * transformer_config.patch_size)
        grid_width = width // (vae_scale_factor_spatial * transformer_config.patch_size)

        p = transformer_config.patch_size
        p_t = transformer_config.patch_size_t

        base_size_width = transformer_config.sample_width // p
        base_size_height = transformer_config.sample_height // p

        if p_t is None:
            grid_crops_coords = self.get_resize_crop_region_for_grid(
                (grid_height, grid_width), base_size_width, base_size_height
            )
            freqs_cos, freqs_sin = get_3d_rotary_pos_embed(
                embed_dim=transformer_config.attention_head_dim,
                crops_coords=grid_crops_coords,
                grid_size=(grid_height, grid_width),
                temporal_size=num_frames,
                device=device,
            )
        else:
            base_num_frames = (num_frames + p_t - 1) // p_t

            freqs_cos, freqs_sin = get_3d_rotary_pos_embed(
                embed_dim=transformer_config.attention_head_dim,
                crops_coords=None,
                grid_size=(grid_height, grid_width),
                temporal_size=base_num_frames,
                grid_type="slice",
                max_size=(base_size_height, base_size_width),
                device=device,
            )

        return freqs_cos, freqs_sin

register("sparkvsr-s1", "sft", KFVSRS1RefTrainer)
