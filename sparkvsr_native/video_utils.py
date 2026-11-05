"""Frame preprocessing and temporal/spatial tiling helpers.

Ported 1:1 from SparkVSR's `sparkvsr_inference_script.py` / the ComfyUI
wrapper's `preprocess_video_match`, but with no decord / torchvision
dependency: ComfyUI hands us an IMAGE batch [F,H,W,C] in [0,1].
"""

import torch
import torch.nn.functional as F


def parse_indices(text):
    """'0,16,32' / ' 0, ' / '' -> sorted unique list of ints."""
    if text is None:
        return []
    if isinstance(text, (list, tuple)):
        raw = [int(i) for i in text]
    else:
        raw = []
        for part in str(text).replace("，", ",").split(","):
            part = part.strip()
            if not part:
                continue
            try:
                raw.append(int(float(part)))
            except ValueError:
                continue
    return sorted(set(i for i in raw if i >= 0))


def preprocess_frames(images, upscale=4, output_resolution=None, upscale_mode="bilinear"):
    """ComfyUI IMAGE [F,H,W,C] in [0,1] -> video [1,3,F,H,W] in [-1,1].

    Returns (video, video_lr, pad_f, pad_h, pad_w, effective_upscale,
             original_shape). Matches the official wrapper's padding rules:
      * temporal: pad to F % 8 == 1 by repeating the last frame
      * spatial : pad H/W to multiples of 4 (on the LR frames)
    """
    if images.ndim != 4:
        raise ValueError(f"Expected IMAGE [F,H,W,C], got {tuple(images.shape)}")
    frames = images.float()  # [F,H,W,C] 0..1
    F0, H0, W0, C = frames.shape
    original_shape = (F0, H0, W0, C)

    # temporal pad: F-1 multiple of 8 (CogVideoX VAE temporal ratio 4 + patch_t 2)
    remainder = (F0 - 1) % 8
    pad_f = 0
    if remainder != 0:
        pad_f = 8 - remainder
        frames = torch.cat([frames, frames[-1:].repeat(pad_f, 1, 1, 1)], dim=0)

    pad_h = (4 - H0 % 4) % 4
    pad_w = (4 - W0 % 4) % 4
    if pad_h > 0 or pad_w > 0:
        # pad right/bottom on [F,H,W,C]
        frames = F.pad(frames, (0, 0, 0, pad_w, 0, pad_h))

    # [F,C,H,W] in 0..255 space (official wrapper works in 0..255 before normalizing)
    video_lr = frames.permute(0, 3, 1, 2).contiguous() * 255.0
    H, W = video_lr.shape[-2:]

    if output_resolution is not None:
        target_h, target_w = output_resolution
        if target_h <= 0 or target_w <= 0:
            raise ValueError("output_resolution must be positive")
        scale = max(target_h / H, target_w / W)
        scaled_h, scaled_w = int(H * scale), int(W * scale)
        up = F.interpolate(video_lr, size=(scaled_h, scaled_w),
                           mode=upscale_mode, align_corners=False)
        crop_top = (scaled_h - target_h) // 2
        crop_left = (scaled_w - target_w) // 2
        up = up[:, :, crop_top:crop_top + target_h, crop_left:crop_left + target_w]
        pad_he = (8 - target_h % 8) % 8
        pad_we = (8 - target_w % 8) % 8
        if pad_he > 0 or pad_we > 0:
            up = F.pad(up, (0, pad_we, 0, pad_he))
        effective_upscale = 1
    else:
        up = F.interpolate(video_lr, size=(H * upscale, W * upscale),
                           mode=upscale_mode, align_corners=False)
        effective_upscale = upscale

    # [0,255] -> [-1,1]; then [1,C,F,H,W]
    up = up / 255.0 * 2.0 - 1.0
    video = up.unsqueeze(0).permute(0, 2, 1, 3, 4).contiguous()
    return video, video_lr, pad_f, pad_h, pad_w, effective_upscale, original_shape


def remove_padding(video, pad_f, pad_h, pad_w):
    """video: [B,F,H,W,C] in [0,1] (ComfyUI VAE.decode layout)."""
    if pad_f > 0:
        video = video[:, :-pad_f]
    if pad_h > 0:
        video = video[:, :, :-pad_h]
    if pad_w > 0:
        video = video[:, :, :, :-pad_w]
    return video


def make_temporal_chunks(F_len, chunk_len, overlap_t=8):
    if chunk_len == 0:
        return [(0, F_len)]
    stride = chunk_len - overlap_t
    if stride <= 0:
        raise ValueError("chunk_len must be greater than overlap_t")
    starts = list(range(0, F_len - overlap_t, stride))
    if starts[-1] + chunk_len < F_len:
        starts.append(F_len - chunk_len)
    chunks = [(s, min(s + chunk_len, F_len)) for s in starts]
    if len(chunks) >= 2 and chunks[-1][1] - chunks[-1][0] < chunk_len:
        last = chunks.pop()
        chunks[-1] = (chunks[-1][0], last[1])
    return chunks


def make_spatial_tiles(H, W, tile_size_hw, overlap_hw=(32, 32)):
    th, tw = tile_size_hw
    oh, ow = overlap_hw
    if th == 0 or tw == 0:
        return [(0, H, 0, W)]
    sh, sw = th - oh, tw - ow
    if sh <= 0 or sw <= 0:
        raise ValueError("tile size must be greater than overlap")
    h_tiles = list(range(0, H - oh, sh))
    if not h_tiles or h_tiles[-1] + th < H:
        h_tiles.append(H - th)
    if len(h_tiles) >= 2 and h_tiles[-1] + th > H:
        h_tiles.pop()
    w_tiles = list(range(0, W - ow, sw))
    if not w_tiles or w_tiles[-1] + tw < W:
        w_tiles.append(W - tw)
    if len(w_tiles) >= 2 and w_tiles[-1] + tw > W:
        w_tiles.pop()
    out = []
    for hs in h_tiles:
        he = min(hs + th, H)
        if he + sh > H:
            he = H
        for ws in w_tiles:
            we = min(ws + tw, W)
            if we + sw > W:
                we = W
            out.append((hs, he, ws, we))
    return out


def get_valid_tile_region(t_start, t_end, h_start, h_end, w_start, w_end,
                          video_shape, overlap_t, overlap_h, overlap_w):
    _, _, F_len, H, W = video_shape
    t_len, h_len, w_len = t_end - t_start, h_end - h_start, w_end - w_start
    vt0 = 0 if t_start == 0 else overlap_t // 2
    vt1 = t_len if t_end == F_len else t_len - overlap_t // 2
    vh0 = 0 if h_start == 0 else overlap_h // 2
    vh1 = h_len if h_end == H else h_len - overlap_h // 2
    vw0 = 0 if w_start == 0 else overlap_w // 2
    vw1 = w_len if w_end == W else w_len - overlap_w // 2
    return {
        "vt0": vt0, "vt1": vt1, "vh0": vh0, "vh1": vh1, "vw0": vw0, "vw1": vw1,
        "ot0": t_start + vt0, "ot1": t_start + vt1,
        "oh0": h_start + vh0, "oh1": h_start + vh1,
        "ow0": w_start + vw0, "ow1": w_start + vw1,
    }
