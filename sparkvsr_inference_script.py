from pathlib import Path
import logging
from .finetune.utils.ref_utils import _select_indices
import torch
from torchvision import transforms
# from torchvision.io import write_video
# from tqdm import tqdm
import gc
from diffusers import (
    CogVideoXDPMScheduler,
    CogVideoXImageToVideoPipeline,
)
from diffusers.models import AutoencoderKLCogVideoX, CogVideoXTransformer3DModel
import folder_paths
from transformers import set_seed
from typing import Dict, Tuple, List
from diffusers.models.embeddings import get_3d_rotary_pos_embed
from safetensors.torch import load_file
from tqdm import tqdm
# import json
import os
import cv2
from PIL import Image
from .model_loader_utils import tensor2image
from pathlib import Path
from .model_loader_utils import  map_0_1_to_neg1_1,map_neg1_1_to_0_1
import imageio.v3 as iio
from contextlib import contextmanager
import sys
import time
# Must import after torch because this can sometimes lead to a nasty segmentation fault, or stack smashing error
# Very few bug reports but it happens. Look in decord Github issues for more relevant information.
import decord  # isort:skip
from contextlib import nullcontext
from accelerate import init_empty_weights
from diffusers.utils import is_accelerate_available
decord.bridge.set_bridge("torch")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add path for finetune utils
import sys
sys.path.append(os.getcwd())

#from .finetune.utils.ref_utils import get_ref_frames_api, save_ref_frames_locally


# 0 ~ 1
to_tensor = transforms.ToTensor()
video_exts = ['.mp4', '.avi', '.mov', '.mkv']
fr_metrics = ['psnr', 'ssim', 'lpips', 'dists']

@contextmanager
def temp_patch_module_attr(module_name: str, attr_name: str, new_obj):
    mod = sys.modules.get(module_name)
    if mod is None:
        yield
        return
    had = hasattr(mod, attr_name)
    orig = getattr(mod, attr_name, None)
    setattr(mod, attr_name, new_obj)
    try:
        yield
    finally:
        if had:
            setattr(mod, attr_name, orig)
        else:
            try:
                delattr(mod, attr_name)
            except Exception:
                pass

def no_grad(func):
    def wrapper(*args, **kwargs):
        with torch.no_grad():
            return func(*args, **kwargs)
    return wrapper


    return any(filename.lower().endswith(ext) for ext in video_exts)


def center_crop_to_aspect_ratio(tensor: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
    """
    Center sorts a tensor (C, H, W) to match the aspect ratio of target_h/target_w.
    """
    _, src_h, src_w = tensor.shape
    target_ar = target_w / target_h
    src_ar = src_w / src_h

    if abs(target_ar - src_ar) < 1e-3:
        return tensor

    if src_ar > target_ar:
        # Source is wider: crop width
        new_w = int(src_h * target_ar)
        start_w = (src_w - new_w) // 2
        return tensor[:, :, start_w : start_w + new_w]
    else:
        # Source is taller: crop height
        new_h = int(src_w / target_ar)
        start_h = (src_h - new_h) // 2
        return tensor[:, start_h : start_h + new_h, :]


def read_video_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(to_tensor(Image.fromarray(rgb)))
    cap.release()
    return torch.stack(frames)


def read_image_folder(folder_path):
    image_files = sorted([
        os.path.join(folder_path, f) for f in os.listdir(folder_path)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ])
    frames = [to_tensor(Image.open(p).convert("RGB")) for p in image_files]
    return torch.stack(frames)


def load_sequence(path):
    # return a tensor of shape [F, C, H, W] // 0, 1
    if os.path.isdir(path):
        return read_image_folder(path)
    elif os.path.isfile(path):
        if path.lower().endswith(video_exts):
            return read_video_frames(path)
        elif path.lower().endswith(('.png', '.jpg', '.jpeg')):
            # Treat image as a single-frame video
            img = to_tensor(Image.open(path).convert("RGB"))
            return img.unsqueeze(0)  # [1, C, H, W]
    raise ValueError(f"Unsupported input: {path}")

@no_grad
def compute_metrics(pred_frames, gt_frames, metrics_model, metric_accumulator, file_name):

    print(f"\n\n[{file_name}] Metrics:", end=" ")

    # Center crop GT to match pred resolution if misaligned
    if gt_frames is not None:
        pred_h, pred_w = pred_frames.shape[-2], pred_frames.shape[-1]
        gt_h, gt_w = gt_frames.shape[-2], gt_frames.shape[-1]
        if (pred_h, pred_w) != (gt_h, gt_w):
            crop_top = (gt_h - pred_h) // 2
            crop_left = (gt_w - pred_w) // 2
            gt_frames = gt_frames[:, :, crop_top:crop_top + pred_h, crop_left:crop_left + pred_w]
            print(f"[Align] GT {gt_h}x{gt_w} -> center crop to {pred_h}x{pred_w}", end=" ")

    for name, model in metrics_model.items():
        scores = []
        # Ensure lengths match
        min_len = min(pred_frames.shape[0], gt_frames.shape[0])
        for i in range(min_len):
            pred = pred_frames[i].unsqueeze(0)
            if gt_frames is not None:
                gt = gt_frames[i].unsqueeze(0)
            else:
                gt = None
                
            if name in fr_metrics and gt is not None:
                score = model(pred, gt).item()
            else:
                score = model(pred).item()
            scores.append(score)
        val = sum(scores) / len(scores)
        metric_accumulator[name].append(val)
        print(f"{name.upper()}={val:.4f}", end="  ")
    print()


def save_frames_as_png(video, output_dir, fps=8):
    video = video[0]  # Remove batch dimension
    video = video.permute(1, 2, 3, 0)  # C F H W --> [F, H, W, C] 

    os.makedirs(output_dir, exist_ok=True)
    frames = (video * 255).clamp(0, 255).to(torch.uint8).cpu().numpy()
    
    for i, frame in enumerate(frames):
        filename = os.path.join(output_dir, f"{i:03d}.png")
        Image.fromarray(frame).save(filename)


def save_video_with_imageio(video, output_path, fps=8, format='yuv444p'):
    video = video[0]
    video = video.permute(1, 2, 3, 0)

    frames = (video * 255).clamp(0, 255).to(torch.uint8).cpu().numpy()

    if format == 'yuv444p':
        iio.imwrite(
            output_path,
            frames,
            fps=fps,
            codec='libx264',
            pixelformat='yuv444p',
            macro_block_size=None,
            ffmpeg_params=['-crf', '0'],
        )
    else:
        iio.imwrite(
            output_path,
            frames,
            fps=fps,
            codec='libx264',
            pixelformat='yuv420p',
            macro_block_size=None,
            ffmpeg_params=['-crf', '10'],
        )


def preprocess_video_match(
    video_path ,
    is_match: bool = False,
) -> torch.Tensor:
    if isinstance(video_path, str):
        video_path = Path(video_path)
        video_reader = decord.VideoReader(uri=video_path.as_posix())
        video_num_frames = len(video_reader)
        frames = video_reader.get_batch(list(range(video_num_frames)))
    else:
        frames = video_path
    F, H, W, C = frames.shape
    original_shape = (F, H, W, C)
    
    pad_f = 0
    pad_h = 0
    pad_w = 0

    if is_match:
        remainder = (F - 1) % 8
        if remainder != 0:
            last_frame = frames[-1:]
            pad_f = 8 - remainder
            repeated_frames = last_frame.repeat(pad_f, 1, 1, 1)
            frames = torch.cat([frames, repeated_frames], dim=0)

        pad_h = (4 - H % 4) % 4
        pad_w = (4 - W % 4) % 4
        if pad_h > 0 or pad_w > 0:
            # pad = (w_left, w_right, h_top, h_bottom)
            frames = torch.nn.functional.pad(frames, pad=(0, 0, 0, pad_w, 0, pad_h))  # pad right and bottom

    # to F, C, H, W
    if   isinstance(video_path, str): # decord video is numpy array
        frames = torch.from_numpy(frames.asnumpy()) 
    return frames.float().permute(0, 3, 1, 2).contiguous(), pad_f, pad_h, pad_w, original_shape


def remove_padding_and_extra_frames(video, pad_F, pad_H, pad_W):
    if pad_F > 0:
        video = video[:, :, :-pad_F, :, :]
    if pad_H > 0:
        video = video[:, :, :, :-pad_H, :]
    if pad_W > 0:
        video = video[:, :, :, :, :-pad_W]
    
    return video


def make_temporal_chunks(F, chunk_len, overlap_t=8):
    if chunk_len == 0:
        return [(0, F)]

    effective_stride = chunk_len - overlap_t
    if effective_stride <= 0:
        raise ValueError("chunk_len must be greater than overlap")

    chunk_starts = list(range(0, F - overlap_t, effective_stride))
    if chunk_starts[-1] + chunk_len < F:
        chunk_starts.append(F - chunk_len)

    time_chunks = []
    for i, t_start in enumerate(chunk_starts):
        t_end = min(t_start + chunk_len, F)
        time_chunks.append((t_start, t_end))

    if len(time_chunks) >= 2 and time_chunks[-1][1] - time_chunks[-1][0] < chunk_len:
        last = time_chunks.pop()
        prev_start, _ = time_chunks[-1]
        time_chunks[-1] = (prev_start, last[1])

    return time_chunks


def make_spatial_tiles(H, W, tile_size_hw, overlap_hw=(32, 32)):
    tile_height, tile_width = tile_size_hw
    overlap_h, overlap_w = overlap_hw

    if tile_height == 0 or tile_width == 0:
        return [(0, H, 0, W)]

    tile_stride_h = tile_height - overlap_h
    tile_stride_w = tile_width - overlap_w

    if tile_stride_h <= 0 or tile_stride_w <= 0:
        raise ValueError("Tile size must be greater than overlap")

    h_tiles = list(range(0, H - overlap_h, tile_stride_h))
    if not h_tiles or h_tiles[-1] + tile_height < H:
        h_tiles.append(H - tile_height)
    
     # Merge last row if needed
    if len(h_tiles) >= 2 and h_tiles[-1] + tile_height > H:
        h_tiles.pop()

    w_tiles = list(range(0, W - overlap_w, tile_stride_w))
    if not w_tiles or w_tiles[-1] + tile_width < W:
        w_tiles.append(W - tile_width)
    
    # Merge last column if needed
    if len(w_tiles) >= 2 and w_tiles[-1] + tile_width > W:
        w_tiles.pop()

    spatial_tiles = []
    for h_start in h_tiles:
        h_end = min(h_start + tile_height, H)
        if h_end + tile_stride_h > H:
            h_end = H
        for w_start in w_tiles:
            w_end = min(w_start + tile_width, W)
            if w_end + tile_stride_w > W:
                w_end = W
            spatial_tiles.append((h_start, h_end, w_start, w_end))
    return spatial_tiles


def get_valid_tile_region(t_start, t_end, h_start, h_end, w_start, w_end,
                          video_shape, overlap_t, overlap_h, overlap_w):
    _, _, F, H, W = video_shape

    t_len = t_end - t_start
    h_len = h_end - h_start
    w_len = w_end - w_start

    valid_t_start = 0 if t_start == 0 else overlap_t // 2
    valid_t_end = t_len if t_end == F else t_len - overlap_t // 2
    valid_h_start = 0 if h_start == 0 else overlap_h // 2
    valid_h_end = h_len if h_end == H else h_len - overlap_h // 2
    valid_w_start = 0 if w_start == 0 else overlap_w // 2
    valid_w_end = w_len if w_end == W else w_len - overlap_w // 2

    out_t_start = t_start + valid_t_start
    out_t_end = t_start + valid_t_end
    out_h_start = h_start + valid_h_start
    out_h_end = h_start + valid_h_end
    out_w_start = w_start + valid_w_start
    out_w_end = w_start + valid_w_end

    return {
        "valid_t_start": valid_t_start, "valid_t_end": valid_t_end,
        "valid_h_start": valid_h_start, "valid_h_end": valid_h_end,
        "valid_w_start": valid_w_start, "valid_w_end": valid_w_end,
        "out_t_start": out_t_start, "out_t_end": out_t_end,
        "out_h_start": out_h_start, "out_h_end": out_h_end,
        "out_w_start": out_w_start, "out_w_end": out_w_end,
    }

# ==================== REF SPECIFIC LOGIC ====================

def get_resize_crop_region_for_grid(src, tgt_width, tgt_height):
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
        grid_crops_coords = get_resize_crop_region_for_grid(
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
            max_size=(max(base_size_height, grid_height), max(base_size_width, grid_width)),
            device=device,
        )

    return freqs_cos, freqs_sin

@no_grad
def process_video_ref_i2v(
    pipe: CogVideoXImageToVideoPipeline,
    video: torch.Tensor,
    prompt: str = '',
    ref_frames: List[torch.Tensor] = [],
    ref_indices: List[int] = [],
    chunk_start_idx: int = 0,
    noise_step: int = 0,
    sr_noise_step: int = 399,
    empty_prompt_embedding: torch.Tensor = None,
    ref_guidance_scale: float = 1.0,
):
    # Decode video
    # video: [B, C, F, H, W]
    video = video.to(pipe.device,pipe.dtype)
    pipe.vae.to(video.device, dtype=video.dtype)
   
    #print(f"Decoding video with shape {video.dtype},vae device: {pipe.vae.device}, vae dtype: {pipe.vae.dtype}")
    latent_dist = pipe.vae.encode(video).latent_dist
    lq_latent = latent_dist.sample() * pipe.vae.config.scaling_factor
    # lq_latent: [B, 16, F_lat, H_lat, W_lat]
    
    batch_size, num_channels, num_frames, height, width = lq_latent.shape
    device = lq_latent.device
    dtype = lq_latent.dtype

    # Prepare Ref Latent
    full_ref_latent = torch.zeros_like(lq_latent)
    
    for i, idx in enumerate(ref_indices):
        if i >= len(ref_frames): break
        
        # Calculate local index in this chunk
        local_frame_idx = idx - chunk_start_idx
        
        # If idx is outside this chunk, skip
        # Note: video F is in pixels. latent F is F_pix / 4. 
        # local_frame_idx is in pixels.
        
        # Map pixel index to latent index
        target_lat_idx = local_frame_idx // 4
        
        if 0 <= target_lat_idx < num_frames:
             # This reference frame belongs to this latent chunk
             r_frame = ref_frames[i].to(device, dtype=dtype) # [C, H, W]
             
             # Chunk for VAE [1, C, 4, H, W]
             chunk = r_frame.unsqueeze(0).unsqueeze(2).repeat(1, 1, 4, 1, 1)
             
             lat_dist = pipe.vae.encode(chunk).latent_dist
             lat = lat_dist.sample() * pipe.vae.config.scaling_factor
             
             full_ref_latent[:, :, target_lat_idx, :, :] = lat[0, :, 0, :, :]
             
    # --- Dual-Pass / CFG Logic ---
    do_classifier_free_guidance = abs(ref_guidance_scale - 1.0) > 1e-3
    
    if do_classifier_free_guidance:
        # Cond
        input_latent_cond = torch.cat([lq_latent, full_ref_latent], dim=1)
        # Uncond
        uncond_ref_latent = torch.zeros_like(full_ref_latent)
        input_latent_uncond = torch.cat([lq_latent, uncond_ref_latent], dim=1)
        
        # Concatenate batch for parallel forward pass
        input_latent = torch.cat([input_latent_uncond, input_latent_cond], dim=0) # [2*B, C*2, F, H, W]
    else:
        input_latent = torch.cat([lq_latent, full_ref_latent], dim=1) # [B, 32, F, H, W]

    # Handle Patch Size T
    patch_size_t = pipe.transformer.config.patch_size_t
    ncopy = 0
    if patch_size_t is not None:
        ncopy = input_latent.shape[2] % patch_size_t
        first_frame = input_latent[:, :, :1, :, :]
        input_latent = torch.cat([first_frame.repeat(1, 1, ncopy, 1, 1), input_latent], dim=2)
        del first_frame
        torch.cuda.empty_cache()

    # Encode Prompt
    if prompt == "" and empty_prompt_embedding is not None:
        prompt_embedding = empty_prompt_embedding.to(device, dtype=dtype)
        if prompt_embedding.shape[0] != batch_size:
            prompt_embedding = prompt_embedding.repeat(batch_size, 1, 1)
    else:
        prompt_token_ids = pipe.tokenizer(
            prompt,
            padding="max_length",
            max_length=pipe.transformer.config.max_text_seq_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        prompt_token_ids = prompt_token_ids.input_ids
        prompt_embedding = pipe.text_encoder(
            prompt_token_ids.to(device)
        )[0]
        _, seq_len, _ = prompt_embedding.shape
        prompt_embedding = prompt_embedding.view(batch_size, seq_len, -1).to(dtype=dtype)
        # 显式释放中间变量
        del prompt_token_ids
        torch.cuda.empty_cache()

    latents = input_latent.permute(0, 2, 1, 3, 4) # [B or 2B, F, C, H, W]
    
    # Expand prompt embedding for CFG
    if do_classifier_free_guidance:
         prompt_embedding = torch.cat([prompt_embedding, prompt_embedding], dim=0)

    # Add Noise
    if noise_step != 0:
        # Separating Lq part
        lq_part = latents[:, :, :16, :, :]
        ref_part = latents[:, :, 16:, :, :]
        
        noise = torch.randn_like(lq_part)
        add_timesteps = torch.full(
        (latents.shape[0],), # Batch size varies
        fill_value=noise_step,
        dtype=torch.long,
        device=device,
    )
        lq_part = pipe.scheduler.add_noise(lq_part.transpose(1, 2), noise.transpose(1, 2), add_timesteps).transpose(1, 2)
        latents = torch.cat([lq_part, ref_part], dim=2)
        # 显式释放中间变量
        del lq_part, ref_part, noise, add_timesteps
        torch.cuda.empty_cache()
    
    timesteps = torch.full(
        (latents.shape[0],), # Batch size varies
        fill_value=sr_noise_step,
        dtype=torch.long,
        device=device,
    )

    # RoPE
    vae_scale_factor_spatial = 2 ** (len(pipe.vae.config.block_out_channels) - 1)
    transformer_config = pipe.transformer.config
    rotary_emb = (
        prepare_rotary_positional_embeddings(
            height=height * vae_scale_factor_spatial,
            width=width * vae_scale_factor_spatial,
            num_frames=num_frames, # Use original num_frames (before cat) for PE logic? 
            # Wait, latents F dim is padded by ncopy.
            # PE logic usually handles effective F.
            # But let's check `latents.shape[1]`.
            # In trainer: `num_frames=latents.shape[1]`
            # So here: `num_frames=latents.shape[1]`
            transformer_config=transformer_config,
            vae_scale_factor_spatial=vae_scale_factor_spatial,
            device=device,
        )
        if pipe.transformer.config.use_rotary_positional_embeddings
        else None
    )
    
    # OFS
    ofs = None
    if pipe.transformer.config.ofs_embed_dim is not None:
         ofs = torch.full((latents.shape[0],), fill_value=2.0, device=device, dtype=dtype)

    # Predict
    predicted_noise = pipe.transformer(
        hidden_states=latents,
        encoder_hidden_states=prompt_embedding,
        timestep=timesteps,
        image_rotary_emb=rotary_emb,
        ofs=ofs,
        return_dict=False,
    )[0]

    # 显式释放中间变量
    del  prompt_embedding, rotary_emb, ofs
    torch.cuda.empty_cache()

    # Denoise
    predicted_noise_slice = predicted_noise[:, :, :16, :, :].transpose(1, 2)
    lq_sample = latents[:, :, :16, :, :].transpose(1, 2)

    # Apply Guidance
    if do_classifier_free_guidance:
        noise_pred_uncond, noise_pred_cond = predicted_noise_slice.chunk(2)
        predicted_noise_slice = noise_pred_uncond + ref_guidance_scale * (noise_pred_cond - noise_pred_uncond)
        
        # Split lq_sample and timesteps for scheduler step (take one half)
        lq_sample = lq_sample.chunk(2)[1] # Take cond part as base? Or uncond? Typically X_t is same.
        timesteps = timesteps.chunk(2)[0]

        # 显式释放中间变量
        del noise_pred_uncond, noise_pred_cond
        torch.cuda.empty_cache()

    latent_generate = pipe.scheduler.get_velocity(
        predicted_noise_slice, lq_sample, timesteps
    )

    # 显式释放中间变量
    del predicted_noise, predicted_noise_slice, lq_sample, timesteps
    torch.cuda.empty_cache()

    if patch_size_t is not None and ncopy > 0:
        latent_generate = latent_generate[:, :, ncopy:, :, :]

    # Decode
    pipe.maybe_free_model_hooks()
    latent_generate = pipe.vae.decode(latent_generate / pipe.vae.config.scaling_factor).sample
    latent_generate = (latent_generate * 0.5 + 0.5).clamp(0.0, 1.0)
    
    return latent_generate



def load_sparkvsr_model( args, device):
    # Setup
    #print(f"Loading Model from {args}")
    if args.dtype == "float16":
        dtype = torch.float16
    elif args.dtype == "bfloat16":
        dtype = torch.bfloat16
    else:
        dtype = torch.float32

    # Load vae
    vae_config=AutoencoderKLCogVideoX.load_config(os.path.join(args.repo,"vae"))
    vae=AutoencoderKLCogVideoX.from_config(vae_config,torch_dtype=dtype)
    vae_state_dict = load_file(args.vae_path)
    x,y=vae.load_state_dict(vae_state_dict, strict=False)
    #print(x,y,"vae")
    vae.eval().to(device)
    del vae_state_dict

    print(f"Loading Model from {args.model_path}")
    ctx = init_empty_weights if is_accelerate_available() else nullcontext
    dit_config=CogVideoXTransformer3DModel.load_config(os.path.join(args.repo,"transformer"))
    with ctx():
        transformer=CogVideoXTransformer3DModel.from_config(dit_config, torch_dtype=dtype)
    if args.model_path is not None:
        dit_state_dict = load_file(args.model_path)
        x,y=transformer.load_state_dict(dit_state_dict, strict=False, assign=True)
        #print(x,y,"dit")
    elif args.gguf_path is not None:
        dit_state_dict=load_gguf_checkpoint(args.gguf_path)
        set_gguf2meta_model(transformer,dit_state_dict,dtype,device)
    del dit_state_dict
    transformer.eval().to("cpu")

    # with temp_patch_module_attr("diffusers", "CogVideoXImageToVideoPipeline", CogVideoXImageToVideoPipeline):
    #     if dit_path is not None:
    #         CogVideo_transformer = CogVideoXImageToVideoPipeline.from_single_file(dit_path,config=os.path.join(args.repo, "transformer"),torch_dtype=torch.bfloat16)  
    #     elif gguf_path is not None:
    #         CogVideo_transformer = CogVideoXImageToVideoPipeline.from_single_file(
    #             gguf_path,
    #             config=os.path.join(args.repo, "transformer"),
    #             quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
    #             torch_dtype=torch.bfloat16,) 
    #     else:
    #         raise ValueError("Please provide either dit_path or gguf_path")
    pipe = CogVideoXImageToVideoPipeline.from_pretrained(args.repo,vae=vae, transformer=transformer,text_encoder=None,torch_dtype=dtype, low_cpu_mem_usage=True)
    
    if args.lora_path:
        print(f"Loading LoRA from {args.lora_path}")
        pipe.load_lora_weights(args.lora_path, adapter_name="dove_ref_i2v")
        pipe.fuse_lora(lora_scale=1.0)
        
    pipe.scheduler = CogVideoXDPMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
    
    
    return pipe

def per_video_refer(args, video_path,SR_model,sr_image, save_pisasr=True,empty_prompt_path=None,sr_embedding=None):
    empty_prompt_embedding=torch.load(empty_prompt_path, map_location="cpu",weights_only=False)
    sr_embedding= torch.load(sr_embedding, map_location="cpu",weights_only=False) if  sr_embedding is not None else None
    video, pad_f, pad_h, pad_w, original_shape = preprocess_video_match(video_path, is_match=True) # FHWC-->FCHW
    H_orig, W_orig = video.shape[2], video.shape[3]
    # Upscale Input
    if args.output_resolution is not None:
        target_h, target_w = args.output_resolution
        
        # Scale-and-Center-Crop: scale so both dims >= target, then crop center
        src_h, src_w = H_orig, W_orig
        scale_h = target_h / src_h
        scale_w = target_w / src_w
        scale_factor = max(scale_h, scale_w)  # Ensure both dims >= target
        
        scaled_h = int(src_h * scale_factor)
        scaled_w = int(src_w * scale_factor)
        
        print(f"Output Resolution Mode: {target_h}x{target_w}")
        print(f"  Source: {src_h}x{src_w} | Scale: {scale_factor:.4f} -> Scaled: {scaled_h}x{scaled_w}")
        
        # Step 1: Scale up
        video_up = torch.nn.functional.interpolate(
            video, 
            size=(scaled_h, scaled_w),
            mode=args.upscale_mode,
            align_corners=False
        )
        
        # Step 2: Center crop to target
        crop_top = (scaled_h - target_h) // 2
        crop_left = (scaled_w - target_w) // 2
        video_up = video_up[:, :, crop_top:crop_top + target_h, crop_left:crop_left + target_w]
        print(f"  Center crop: top={crop_top} left={crop_left} -> Final: {target_h}x{target_w}")
        
        # Step 3: Pad to VAE-compatible size (multiple of 8)
        pad_h_extra = (8 - target_h % 8) % 8
        pad_w_extra = (8 - target_w % 8) % 8
        if pad_h_extra > 0 or pad_w_extra > 0:
            video_up = torch.nn.functional.pad(video_up, (0, pad_w_extra, 0, pad_h_extra))
            print(f"  VAE pad: +{pad_h_extra}h +{pad_w_extra}w -> {target_h + pad_h_extra}x{target_w + pad_w_extra}")
        
        # Set effective upscale to 1 for downstream padding removal
        effective_upscale = 1
    else:
        video_up = torch.nn.functional.interpolate(
            video, 
            size=(H_orig * args.upscale, W_orig * args.upscale),
            mode=args.upscale_mode,
            align_corners=False
        )
        effective_upscale = args.upscale
     # Normalize to [-1, 1]
    if   isinstance(video_path,str):
        video_up = (video_up / 255.0 * 2.0) - 1.0 # From [0, 255] Tensor (preprocess returns 0-255 float range? wait)
    else:
        video_up=map_0_1_to_neg1_1(video_up) # From [0, 1] Tensor to [-1, 1]

    video_lr = video
    video = video_up.unsqueeze(0).permute(0, 2, 1, 3, 4).contiguous() # [B, C, F, H, W]

    # Retrieve References
    ref_frames_list = []
    if args.ref_mode != "no_ref":
        if args.ref_indices is not None:
            # Manually specified
            ref_indices = sorted(list(set(args.ref_indices)))
            # Validate interval
            if len(ref_indices) > 1:
                for i in range(len(ref_indices) - 1):
                    if ref_indices[i+1] - ref_indices[i] < 4:
                        raise ValueError(f"Reference frame interval must be > 3 (>= 4). Found interval {ref_indices[i+1] - ref_indices[i]} between {ref_indices[i]} and {ref_indices[i+1]}.")
            if not ref_indices:
                print(f"Using manually specified indices: NONE (0 reference frames)")
            else:
                print(f"Using manually specified indices: {ref_indices}")
        else:
            ref_indices = _select_indices(video.shape[2]) # Shape 2 is F now after permute
            print(f"Using auto-selected indices: {ref_indices}")
    else:
        ref_indices = []
    
    if args.ref_mode == "no_ref":
        print("Running in No-Ref mode (0 reference frames).")
        ref_frames_list = []
        ref_indices = []

    elif args.ref_mode == "pisasr":
        from .finetune.PiSASR.test_pisasr import  infer_pisasr
        prefiex=time.strftime("%Y%m%d-%H%M%S", time.localtime())
        ref_frames_list = infer_pisasr(SR_model,ref_indices, video_lr,sr_embedding,args.upscale,) #pli list
        ref_frames_list=resize_pli(ref_frames_list,video)
        if save_pisasr:
            for img,idx in zip(ref_frames_list,ref_indices):
                print(f"Saving SR image {idx},img.shape:{img.shape}")
                img=map_neg1_1_to_0_1(img.unsqueeze(0).permute(0, 2, 3, 1))
                tensor2image(img).save(os.path.join(folder_paths.get_output_directory(),f"sr_img_indices{idx}_{prefiex}.png"))

    elif args.ref_mode == "SRimg":
        ref_frames_list==resize_pli(ref_frames_list,video)
        if len(ref_frames_list)>len(ref_indices):
            print(f"[PiSA-SR] WARNING: SR image count ({len(ref_frames_list)}) > ref_indices count ({len(ref_indices)}). Truncating SR images to match ref_indices count.")
            ref_frames_list=ref_frames_list[:len(ref_indices)]
        else:
            print(f"[PiSA-SR] WARNING: ref_indices count ({len(ref_indices)}) > SR image count ({len(ref_frames_list)}). Truncating ref_indices to match SR image count.")
            ref_indices=ref_indices[:len(ref_frames_list)]

    output={
        "ref_frames_list": ref_frames_list,
        "pad_f": pad_f,
        "pad_h": pad_h,
        "pad_w": pad_w,
        "effective_upscale": effective_upscale,
        "video": video,
        "video_lr": video_lr,
        "ref_indices": ref_indices,
        "empty_prompt_embedding": empty_prompt_embedding,
        }
    return output

def resize_pli(pli_list,video):
    output_list=[]
    for img in pli_list:
        t_img = transforms.ToTensor()(img) # [0, 1]
        t_img = t_img * 2.0 - 1.0 # [-1, 1]
        
        target_h, target_w = video.shape[-2], video.shape[-1]
        
        orig_h, orig_w = t_img.shape[-2], t_img.shape[-1]
        print(f"[PiSA-SR] Generated HD reference resolution: {orig_w}x{orig_h}")
        print(f"[PiSA-SR] Target generated video resolution: {target_w}x{target_h}")
        
        if t_img.shape[-2:] != (target_h, target_w):
            t_img = torch.nn.functional.interpolate(
                t_img.unsqueeze(0),
                size=(target_h, target_w),
                mode="bilinear",
                align_corners=False
            ).squeeze(0)
            
        final_h, final_w = t_img.shape[-2], t_img.shape[-1]
        print(f"[PiSA-SR] Resized reference resolution: {final_w}x{final_h}")
        output_list.append(t_img)
    return output_list


def infer_sparkvsr(args, pipe, conds):
    if args.is_vae_st:
        pipe.vae.enable_slicing()
        pipe.vae.enable_tiling()
        
    set_seed(args.seed)

    ref_frames_list = conds["ref_frames_list"]
    ref_indices = conds["ref_indices"]
    video = conds["video"]
    empty_prompt_embedding = conds["empty_prompt_embedding"]
    # Tiling Setup
    B, C, F, H, W = video.shape
    overlap_t = args.overlap_t if args.chunk_len > 0 else 0
    overlap_hw = tuple(args.overlap_hw) if tuple(args.tile_size_hw) != (0,0) else (0,0)
    
    time_chunks = make_temporal_chunks(F, args.chunk_len, overlap_t)
    spatial_tiles = make_spatial_tiles(H, W, args.tile_size_hw, overlap_hw)
    
    output_video = torch.zeros_like(video)
    #write_count = torch.zeros_like(video, dtype=torch.int)
    print(f"Processing: F={F} H={H} W={W} | Chunks={len(time_chunks)} Tiles={len(spatial_tiles)}")
    print(spatial_tiles, time_chunks) #[(0, 544, 0, 960)] [(0, 33)]
    total_tiles = len(time_chunks) * len(spatial_tiles)
    overall_pbar = tqdm(total=total_tiles, desc="Overall progress")

    for t_idx, (t_start, t_end) in enumerate(time_chunks):
        # 为每个时间块创建空间tile进度条
        spatial_pbar = tqdm(total=len(spatial_tiles), desc=f"Time chunk {t_idx+1}/{len(time_chunks)}", leave=False)
        
        for h_start, h_end, w_start, w_end in spatial_tiles:
            video_chunk = video[:, :, t_start:t_end, h_start:h_end, w_start:w_end]
            
            # Check Refs for Spatial Tile (Optimization: Crop Refs)
            # We need to crop reference frames to match the current spatial tile
            current_ref_frames = []
            for rf in ref_frames_list:
                # rf is [C, VideoH, VideoW]
                rf_crop = rf[:, h_start:h_end, w_start:w_end]
                current_ref_frames.append(rf_crop)

            _video_generate = process_video_ref_i2v(
                pipe=pipe,
                video=video_chunk,
                prompt="",
                ref_frames=current_ref_frames,
                ref_indices=ref_indices,
                chunk_start_idx=t_start,
                noise_step=args.noise_step,
                sr_noise_step=args.sr_noise_step,
                empty_prompt_embedding=empty_prompt_embedding,
                ref_guidance_scale=args.ref_guidance_scale,
            )
            # 显式释放中间变量
            del video_chunk, current_ref_frames
            torch.cuda.empty_cache()
            
            region = get_valid_tile_region(
                t_start, t_end, h_start, h_end, w_start, w_end,
                video.shape, overlap_t, overlap_hw[0], overlap_hw[1]
            )
            
            output_video[:, :, region["out_t_start"]:region["out_t_end"],
                            region["out_h_start"]:region["out_h_end"],
                            region["out_w_start"]:region["out_w_end"]] = \
            _video_generate[:, :, region["valid_t_start"]:region["valid_t_end"],
                            region["valid_h_start"]:region["valid_h_end"],
                            region["valid_w_start"]:region["valid_w_end"]]
            
            # 显式释放中间变量
            del _video_generate, region
            torch.cuda.empty_cache()

            overall_pbar.update(1)
            spatial_pbar.update(1)
            overall_pbar.set_postfix({
                'Chunk': f'{t_start}-{t_end}',
                'Tile': f'{h_start}-{h_end}x{w_start}-{w_end}'
            })
            spatial_pbar.set_postfix({
                'Spatial': f'{h_start}-{h_end}x{w_start}-{w_end}'
            })
        
        spatial_pbar.close()

    overall_pbar.close()
    pipe.maybe_free_model_hooks()

    video_generate = output_video
    # Save
    video = remove_padding_and_extra_frames(video_generate, conds["pad_f"], conds["pad_h"]*conds["effective_upscale"], conds["pad_w"]*conds["effective_upscale"])

    video = video[0]
    video = video.permute(1, 2, 3, 0).float().to("cpu")#C F H W --> F, H, W, C
    if args.save_output:
        out_file_path = os.path.join(args.output_path, "check.mp4")
        save_video_with_imageio(video_generate, out_file_path, fps=16, format="yuv444p")
    del video_generate
    torch.cuda.empty_cache()
    return video
    

def load_gguf_checkpoint(gguf_checkpoint_path):

    from  diffusers.utils  import is_gguf_available, is_torch_available
    if is_gguf_available() and is_torch_available():
        import gguf
        from gguf import GGUFReader
        from diffusers.quantizers.gguf.utils import SUPPORTED_GGUF_QUANT_TYPES, GGUFParameter,dequantize_gguf_tensor
    else:
        logger.error(
            "Loading a GGUF checkpoint in PyTorch, requires both PyTorch and GGUF>=0.10.0 to be installed. Please see "
            "https://pytorch.org/ and https://github.com/ggerganov/llama.cpp/tree/master/gguf-py for installation instructions."
        )
        raise ImportError("Please install torch and gguf>=0.10.0 to load a GGUF checkpoint in PyTorch.")

    reader = GGUFReader(gguf_checkpoint_path)
    parsed_parameters = {}
  
    for i, tensor in enumerate(reader.tensors):
        name = tensor.name
        quant_type = tensor.tensor_type


        is_gguf_quant = quant_type not in [gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16]
        if is_gguf_quant and quant_type not in SUPPORTED_GGUF_QUANT_TYPES:
            _supported_quants_str = "\n".join([str(type) for type in SUPPORTED_GGUF_QUANT_TYPES])
            raise ValueError(
                (
                    f"{name} has a quantization type: {str(quant_type)} which is unsupported."
                    "\n\nCurrently the following quantization types are supported: \n\n"
                    f"{_supported_quants_str}"
                    "\n\nTo request support for this quantization type please open an issue here: https://github.com/huggingface/diffusers"
                )
            )

        weights = torch.from_numpy(tensor.data) #tensor.data.copy()
 
        parsed_parameters[name] = GGUFParameter(weights, quant_type=quant_type) if is_gguf_quant else weights
        del tensor,weights
        if i > 0 and i % 1000 == 0:  # 每1000个tensor执行一次gc
            logger.info(f"Processed {i}tensors...")
            gc.collect()
    del reader
    gc.collect()
    return parsed_parameters

def set_gguf2meta_model(meta_model,model_state_dict,dtype,device,lora_sd_and_strengths=None):
    from diffusers import GGUFQuantizationConfig
    from diffusers.quantizers.gguf import GGUFQuantizer

    g_config = GGUFQuantizationConfig(compute_dtype=dtype or torch.bfloat16)
    hf_quantizer = GGUFQuantizer(quantization_config=g_config)
    hf_quantizer.pre_quantized = True
    
    if lora_sd_and_strengths is not None:
        print("Applying LoRAs to GGUF model")
        model_state_dict=apply_loras_gguf(model_state_dict, lora_sd_and_strengths, dtype)

    hf_quantizer._process_model_before_weight_loading(
        meta_model,
        device_map={"": device} if device else None,
        state_dict=model_state_dict
    )
    from diffusers.models.model_loading_utils import load_model_dict_into_meta
    load_model_dict_into_meta(
        meta_model, 
        model_state_dict, 
        hf_quantizer=hf_quantizer,
        device_map={"": device} if device else None,
        dtype=dtype
    )

    hf_quantizer._process_model_after_weight_loading(meta_model)

    
    del model_state_dict
    gc.collect()
    return meta_model.to(dtype=dtype)

def apply_loras_gguf(
    model_sd,
    lora_sd_and_strengths,
    dtype: torch.dtype,
):
    sd = {}
    device = torch.device("meta")
    for key, weight in model_sd.items():
        if weight is None:
            continue
        device = weight.device
        #target_dtype = dtype if dtype is not None else weight.dtype
        #deltas_dtype = target_dtype if target_dtype not in [torch.float8_e4m3fn, torch.float8_e5m2] else torch.bfloat16
        deltas_dtype =  torch.bfloat16
        deltas = _prepare_deltas(lora_sd_and_strengths, key, deltas_dtype, device)
        if deltas is None:
            deltas = weight
        elif weight.dtype == torch.bfloat16:
            deltas.add_(weight)
        else:
            raise ValueError(f"Unsupported dtype: {weight.dtype}")
        sd[key] = deltas
        del weight,deltas
    del model_sd
    gc.collect()
    return sd

def _prepare_deltas(
    lora_sd_and_strengths, key: str, dtype: torch.dtype, device: torch.device
) -> torch.Tensor | None:
    deltas = []
    prefix = key[: -len(".weight")]
    key_a = f"{prefix}.lora_A.weight"
    key_b = f"{prefix}.lora_B.weight"
    for lsd, coef in lora_sd_and_strengths:
        if key_a not in lsd.sd or key_b not in lsd.sd:
            continue
        a = lsd.sd[key_a].to(device=device)
        b = lsd.sd[key_b].to(device=device)
        product = torch.matmul(b * coef, a)
        del a, b
        deltas.append(product.to(dtype=dtype))
    if len(deltas) == 0:
        return None
    elif len(deltas) == 1:
        return deltas[0]
    else:
        return torch.sum(torch.stack(deltas, dim=0), dim=0)
