from typing import Any, Dict, List
import random
import torch
import torch.nn.functional as F
from .real_sr_image_video_dataset import RealSRImageVideoDataset
from .utils import load_file, save_file
import hashlib

class RefRealSRImageVideoDataset(RealSRImageVideoDataset):
    def __getitem__(self, index: int) -> Dict[str, Any]:
        if isinstance(index, list):
            return index
        prompt = self.prompts[index]
        is_empty_propmt = random.random() < self.trainer.args.empty_ratio
        if is_empty_propmt:
             prompt = ''

        train_resolution_str = "x".join(str(x) for x in self.trainer.args.train_resolution)

        # Image
        image_path = self.images[index % len(self.images)]
        # [B, C, F, H, W]
        image_lq_frames_resize, image_hq_frames = self.preprocess_image_video(image_path, 'image')

        # Video
        video_path = self.videos[index]
        video_lq_frames_resize, video_hq_frames = self.preprocess_image_video(video_path, 'video')
        
        # --- Reference Selection Logic ---
        
        # Image: No Ref (SFT)
        # Return dummy/empty ref for image batch consistency if handled uniformly, 
        # or we return None and let trainer handle it. 
        # Trainer expects dictionary items. 
        # We can return `ref_video` and `ref_indices` for Video component.
        # For Image component, we can return None or empty.
        
        # Video Ref Selection
        num_frames = video_hq_frames.shape[2] # [B, C, F, H, W] -> B=1. index 2 is F.
        
        # Logic from RefRealSRDataset
        # Max frames allowing interval > 3 (gap >= 4)
        max_num_ref = (num_frames - 1) // 4 + 1
        upper_bound = max(1, max_num_ref)
        
        # Reference Dropout Logic (S2)
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
        else:
            ref_indices = []
        
        ref_indices.sort()
        
        # Select Video Ref Frames
        # video_hq_frames[0] is [C, F, H, W]
        # We need [R, C, H, W] ? Or [C, R, H, W]?
        # Dataset typically prepares for Collation or Trainer.
        # Let's extract [1, C, R, H, W] to match struct
        
        if len(ref_indices) > 0:
            ref_frames_vid = video_hq_frames[:, :, ref_indices, :, :] # [1, C, R, H, W]
        else:
            # Empty tensor [1, C, 0, H, W]
            ref_frames_vid = torch.empty((1, video_hq_frames.shape[1], 0, video_hq_frames.shape[3], video_hq_frames.shape[4]), dtype=video_hq_frames.dtype)
        ref_indices_tensor = torch.tensor(ref_indices, dtype=torch.long)
        
        # Image Ref: None (or empty tensor).
        # Trainer logic will switch based on `is_image_batch`.
        # So populate valid data for both or handle in trainer?
        # Trainer `compute_loss` picks `is_image_batch` randomly.
        # Batch passed to `compute_loss` is collated.
        # `collate_fn` stacks everything.
        # So Image batch samples should have fields compatible with stacking?
        # Or we can stack dummy values.
        
        # Let's provide None for Image Ref and handle in Collate.
        # Or provide dummy empty tensor.
        # Since `ref_frames` size R varies per sample for Video, we can't stack `ref_video` easily!
        # Unless R is constant.
        # R varies [1, 0.25*F].
        # So `collate_fn` typically would fail to stack simple lists of tensors of different sizes.
        # Solution: Returns List[Tensor] in batch? Or pad?
        # Or we handle it in `collate_fn` override (which we must do).
        
        # Return detailed dict
        
        cache_dir = self.trainer.args.data_root / "cache"

        prompt_embeddings_dir = cache_dir / self.prompt_cache
        prompt_embeddings_dir.mkdir(parents=True, exist_ok=True)
        prompt_hash = str(hashlib.sha256(prompt.encode()).hexdigest())
        prompt_embedding_path = prompt_embeddings_dir / (prompt_hash + ".safetensors")

        if self.empty_prompt is not None:
            # print(f"Using empty prompt embedding")
            prompt_embedding = self.empty_prompt
        elif prompt_embedding_path.exists():
            prompt_embedding = load_file(prompt_embedding_path)["prompt_embedding"]
        else:
            # 不能多进程处理
            prompt_embedding = self.encode_text(prompt)[0].to("cpu")
            if self.is_cache:
                save_file({"prompt_embedding": prompt_embedding.to("cpu")}, prompt_embedding_path)

        encoded_hq_video = None
        encoded_lq_video = None
        
        # We assume no latent caching for this ref implementation for now (as discussed)
        
        # Return dictionaries
        return {
            "prompt": prompt,
            "hq_video": video_hq_frames[0], # [C, F, H, W]
            "lq_video": video_lq_frames_resize[0],
            "ref_video": ref_frames_vid[0],
            "ref_indices": ref_indices_tensor,
            
            "hq_image": image_hq_frames[0],
            "lq_image": image_lq_frames_resize[0],
            
            "prompt_embedding": prompt_embedding,
            "encoded_hq_video": encoded_hq_video,
            "encoded_lq_video": encoded_lq_video,
            "video_metadata": {
                "num_frames": video_hq_frames.shape[2],
                "height": video_hq_frames.shape[3],
                "width": video_hq_frames.shape[4],
            },
            "encoded_video_metadata": None,
            "video_name": video_path.stem,
        }
