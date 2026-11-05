"""SparkVSR native ComfyUI inference stack.

Zero hard dependency on diffusers / transformers / peft / accelerate / decord.
Everything runs on ComfyUI's own CogVideoX DiT core, VAE, model patcher and
scheduler. See README_NATIVE.md for the migration notes.
"""

from .gguf_loader import load_sparkvsr_model, remap_diffusers_cogvideox_keys
from .schedule import (
    ALPHAS_CUMPROD,
    RawVelocitySampling,
    add_noise,
    get_velocity,
    sigma_for_timestep,
)
from .video_utils import (
    make_spatial_tiles,
    make_temporal_chunks,
    get_valid_tile_region,
    parse_indices,
    preprocess_frames,
    remove_padding,
)
from .spark_core import infer_sparkvsr
from .pisa_core import infer_pisa

__all__ = [
    "load_sparkvsr_model",
    "remap_diffusers_cogvideox_keys",
    "ALPHAS_CUMPROD",
    "RawVelocitySampling",
    "add_noise",
    "get_velocity",
    "sigma_for_timestep",
    "make_spatial_tiles",
    "make_temporal_chunks",
    "get_valid_tile_region",
    "parse_indices",
    "preprocess_frames",
    "remove_padding",
    "infer_sparkvsr",
    "infer_pisa",
]
