from typing import Any, Dict, List
import random
import torch
import torch.nn.functional as F
from .real_sr_dataset import RealSRDataset
from .utils import load_file, save_file
import hashlib

class RefRealSRDataset(RealSRDataset):
    def __getitem__(self, index: int) -> Dict[str, Any]:
        if isinstance(index, list):
            return index

        prompt = self.prompts[index]
        is_empty_propmt = random.random() < self.trainer.args.empty_ratio
        if is_empty_propmt:
            prompt = ''

        video = self.videos[index]
        
        # Get processed frames [F, C, H, W]
        hq_frames, lq_frames = self.preprocess(video)
        
        # Reference frame selection logic
        num_frames = hq_frames.shape[0]
        # Randomly choose 1 to 0.4 * num_frames
        # Max frames allowing interval > 3 (gap >= 4)
        max_num_ref = (num_frames - 1) // 4 + 1
        upper_bound = max(1, max_num_ref)
        
        # Reference Dropout Logic
        if random.random() < self.trainer.args.ref_dropout_ratio:
             num_ref = 0
        else:
             num_ref = random.randint(1, upper_bound)
        
        if num_ref > 0:
            ref_indices = [0]
            if num_ref > 1:
                # Select remaining frames uniformly with gap constraint
                step = (num_frames - 1) / (num_ref - 1)
                for i in range(1, num_ref):
                    idx = int(i * step)
                    # Enforce strictly gap >= 4 just in case of rounding
                    if idx - ref_indices[-1] < 4:
                        idx = ref_indices[-1] + 4
                    idx = min(idx, num_frames - 1)
                    ref_indices.append(idx)
            ref_indices.sort()
            input_ref_frames = hq_frames[ref_indices]
        else:
            ref_indices = []
            # Empty tensor matching dimensions except batch/time
            # hq_frames is [F, C, H, W]
            # input_ref_frames should be [0, C, H, W]
            input_ref_frames = torch.empty((0, hq_frames.shape[1], hq_frames.shape[2], hq_frames.shape[3]), dtype=hq_frames.dtype)
        
        # Select reference frames [R, C, H, W]
        ref_frames = input_ref_frames
        
        # Resize LQ frames to match HQ dimensions for encoding if needed (RealSRDataset logic)
        H_, W_ = hq_frames.shape[2], hq_frames.shape[3]
        lq_frames_resize = F.interpolate(lq_frames, size=(H_, W_), mode="bilinear", align_corners=False)

        # Apply transformations [-1, 1]
        hq_frames = self.video_transform(hq_frames)
        lq_frames_resize = self.video_transform(lq_frames_resize)
        if ref_frames.shape[0] > 0:
            ref_frames = self.video_transform(ref_frames)

        # Convert to [B, C, F, H, W] (B=1 here, but will be batched later)
        # RealSRDataset unqueezes and permutes to [C, F, H, W] structure effectively for the output dict
        # Actually RealSRDataset returns:
        # hq_frames: [C, F, H, W] (after permute(0, 2, 1, 3, 4) it was [B, ...])
        # Wait, let's check RealSRDataset again.
        # hq_frames = hq_frames.unsqueeze(0) [1, F, C, H, W]
        # hq_frames = hq_frames.permute(0, 2, 1, 3, 4) [1, C, F, H, W]
        # Return hq_frames[0] -> [C, F, H, W]
        
        hq_frames = hq_frames.unsqueeze(0).permute(0, 2, 1, 3, 4).contiguous()
        lq_frames_resize = lq_frames_resize.unsqueeze(0).permute(0, 2, 1, 3, 4).contiguous()
        ref_frames = ref_frames.unsqueeze(0).permute(0, 2, 1, 3, 4).contiguous()

        # Handle prompt embeddings (same as base class)
        cache_dir = self.trainer.args.data_root / "cache"
        prompt_embeddings_dir = cache_dir / self.prompt_cache
        prompt_embeddings_dir.mkdir(parents=True, exist_ok=True)
        prompt_hash = str(hashlib.sha256(prompt.encode()).hexdigest())
        prompt_embedding_path = prompt_embeddings_dir / (prompt_hash + ".safetensors")

        if self.empty_prompt is not None:
            prompt_embedding = self.empty_prompt
        elif prompt_embedding_path.exists():
            prompt_embedding = load_file(prompt_embedding_path)["prompt_embedding"]
        else:
            prompt_embedding = self.encode_text(prompt)[0].to("cpu")
            if self.is_cache:
                save_file({"prompt_embedding": prompt_embedding.to("cpu")}, prompt_embedding_path)

        # No latent caching for Ref frames currently supported in this snippet as distinct logic
        # If caching is needed, we'd need to change how caching works to account for random ref frames.
        # For now, we will NOT use cached latents for this dynamic ref frame logic, or we assume online encoding.
        # The base class uses `encoded_hq_video` if `is_latent` is true.
        # Since we are modifying the input (adding ref frames), relying on `is_latent` for HQ/LQ is fine,
        # but Ref frames need to be encoded. If `is_latent` is set, the Trainer expects latents.
        # However, `Ref frames` are dynamic.
        # We will assume that for this new dataset, we provide the raw ref frames to the trainer, 
        # and the trainer will handle encoding them, OR we encode them here if we have access to the VAE.
        # `self.encode_video` is available.
        # Let's encode them here if `is_latent` is True to match the pipeline expectation.
        
        encoded_hq_video = None
        encoded_lq_video = None
        encoded_ref_video = None

        if self.is_latent:
            # We reuse base class logic for HQ/LQ latents caching if possible
            train_resolution_str = "x".join(str(x) for x in self.trainer.args.train_resolution)
            hq_video_latent_dir = cache_dir / "video_latent" / "hq"  / self.trainer.args.model_name / train_resolution_str
            lq_video_latent_dir = cache_dir / "video_latent" / "lq" / self.trainer.args.model_name / train_resolution_str
            
            hq_video_latent_dir.mkdir(parents=True, exist_ok=True)
            lq_video_latent_dir.mkdir(parents=True, exist_ok=True)
            
            encoded_hq_video_path = hq_video_latent_dir / (video.stem + ".safetensors")
            encoded_lq_video_path = lq_video_latent_dir / (video.stem + ".safetensors")

            if encoded_hq_video_path.exists():
                encoded_hq_video = load_file(encoded_hq_video_path)["encoded_hq_video"]
            else:
                encoded_hq_video = self.encode_video(hq_frames).to("cpu")
                if self.is_cache:
                    save_file({"encoded_hq_video": encoded_hq_video.to("cpu")}, encoded_hq_video_path)

            if encoded_lq_video_path.exists():
                encoded_lq_video = load_file(encoded_lq_video_path)["encoded_hq_video"]
            else:
                encoded_lq_video = self.encode_video(lq_frames_resize).to("cpu")
                if self.is_cache:
                    save_file({"encoded_hq_video": encoded_lq_video.to("cpu")}, encoded_lq_video_path)

            # Encode Ref Frames (Dynamic, so no caching by default)
            # Ref frames are small (1-3 frames), so encoding on the fly is cheap.
            encoded_ref_video = self.encode_video(ref_frames).to("cpu")

        return {
            "prompt": prompt,
            "hq_video": hq_frames[0],
            "lq_video": lq_frames_resize[0],
            "ref_video": ref_frames[0], # [C, R, H, W]
            "ref_indices": torch.tensor(ref_indices, dtype=torch.long),
            "prompt_embedding": prompt_embedding,
            "encoded_hq_video": encoded_hq_video,
            "encoded_lq_video": encoded_lq_video,
            "encoded_ref_video": encoded_ref_video,
            "video_metadata": {
                "num_frames": hq_frames.shape[2],
                "height": hq_frames.shape[3],
                "width": hq_frames.shape[4],
            },
            "encoded_video_metadata": (
                {
                    "num_frames": encoded_hq_video.shape[1],
                    "height": encoded_hq_video.shape[2],
                    "width": encoded_hq_video.shape[3],
                } if encoded_hq_video is not None else None
            ),
        }
