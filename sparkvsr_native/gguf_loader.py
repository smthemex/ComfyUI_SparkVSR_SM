"""Load SparkVSR-S2 into ComfyUI's native CogVideoX DiT.

The GGUF shipped for SparkVSR is a diffusers CogVideoXTransformer3DModel
checkpoint (1028 tensors, mixed F32 / BF16 / Q8_0). Its keys are diffusers
names such as `transformer_blocks.0.attn1.to_q.weight`, while ComfyUI's
native `CogVideoXTransformer3DModel` uses `blocks.0.q.weight`.

ComfyUI_Dif_GGUF's `Dif_GGUF_Loader` cannot detect this file because
`model_config_from_unet` looks for `blocks.0.norm1.linear.weight`. That is
exactly why the old plugin README says "gguf maybe can't work".

Fix: remap the keys 1:1 (verified 1028/1028 unique, 0 collisions, 42 blocks)
BEFORE handing the state dict to Dif_GGUF's loader. Quantized tensors stay
GGMLTensor objects so Q8_0 / BF16 are dequantized lazily at forward time.
"""

import os
import sys
import logging

import torch


def remap_diffusers_cogvideox_keys(key):
    """diffusers CogVideoXTransformer3DModel key -> ComfyUI native key."""
    if key.startswith("transformer_blocks."):
        key = "blocks." + key[len("transformer_blocks."):]
    # attention projections
    key = key.replace("attn1.to_out.0", "attn_out")
    key = key.replace("attn1.to_q", "q")
    key = key.replace("attn1.to_k", "k")
    key = key.replace("attn1.to_v", "v")
    key = key.replace("attn1.norm_q", "norm_q")
    key = key.replace("attn1.norm_k", "norm_k")
    # feed forward
    key = key.replace("ff.net.0.proj", "ff_proj")
    key = key.replace("ff.net.2", "ff_out")
    # time / ofs embeddings
    key = key.replace("time_embedding.linear_1", "time_embedding_linear_1")
    key = key.replace("time_embedding.linear_2", "time_embedding_linear_2")
    key = key.replace("ofs_embedding.linear_1", "ofs_embedding_linear_1")
    key = key.replace("ofs_embedding.linear_2", "ofs_embedding_linear_2")
    return key


def _import_dif_gguf():
    """Import ComfyUI_Dif_GGUF as a package from the sibling custom_nodes dir."""
    custom_nodes = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    if custom_nodes not in sys.path:
        sys.path.insert(0, custom_nodes)
    try:
        import ComfyUI_Dif_GGUF.dif_gguf_node as dif
        from ComfyUI_Dif_GGUF.ggml_ops import GGMLOps, GGUFModelPatcher, is_quantized
    except ImportError as e:
        raise RuntimeError(
            "ComfyUI_Dif_GGUF is required for the .gguf model path. "
            "Install/enable it, or use a .safetensors SparkVSR model instead. "
            f"Original error: {e}"
        ) from e
    return dif, GGMLOps, GGUFModelPatcher, is_quantized


def load_sparkvsr_model(model_path, model_options=None):
    """Return a ComfyUI MODEL (ModelPatcher) for the SparkVSR DiT.

    .gguf      -> Dif_GGUF reader + key remap + GGMLOps lazy dequant
    .safetensors -> comfy.sd.load_diffusion_model (already comfy/diffusers names)
    """
    model_path = os.path.abspath(model_path)
    if not os.path.isfile(model_path):
        raise FileNotFoundError(model_path)

    if model_path.lower().endswith(".gguf"):
        dif, GGMLOps, GGUFModelPatcher, is_quantized = _import_dif_gguf()
        sd = dif.read_gguf(model_path)
        sd = {remap_diffusers_cogvideox_keys(k): v for k, v in sd.items()}
        if "blocks.0.norm1.linear.weight" not in sd:
            raise RuntimeError(
                "GGUF key remapping failed: not a diffusers CogVideoX checkpoint."
            )
        if len(sd) != len(set(sd)):
            raise RuntimeError("GGUF key remapping produced collisions.")
        logging.info(
            "[SparkVSR] remapped %d GGUF tensors -> ComfyUI CogVideoX names", len(sd)
        )
        has_quant = any(is_quantized(v) for v in sd.values())
        ops = GGMLOps() if has_quant else None
        patcher = dif.load_gguf_model(sd, ops)
        del sd
        if patcher is None:
            raise RuntimeError("Could not detect model type from remapped GGUF.")
        if has_quant:
            # Keep Dif_GGUF's size=0 behaviour: Q8_0/BF16 tensors are
            # dequantized lazily by GGMLOps at forward time, so model
            # management must not try to reserve 20 GB of fp weights.
            patcher = GGUFModelPatcher.clone(patcher)
            patcher.size = 0
    else:
        import comfy.sd
        patcher = comfy.sd.load_diffusion_model(
            model_path, model_options=model_options or {}
        )

    _validate_cogvideox(patcher)
    return patcher


def _validate_cogvideox(patcher):
    import comfy.model_base
    base = getattr(patcher, "model", None)
    if not isinstance(base, comfy.model_base.CogVideoX):
        raise RuntimeError(
            "Loaded model is not a native ComfyUI CogVideoX DiT "
            f"(got {type(base).__name__}). SparkVSR needs CogVideoX 1.5 I2V."
        )
    dit = base.diffusion_model
    if getattr(dit, "in_channels", None) != 32:
        raise RuntimeError(
            f"SparkVSR needs in_channels=32 (I2V), got {dit.in_channels}."
        )
    if getattr(dit, "patch_size_t", None) != 2:
        raise RuntimeError(
            f"SparkVSR needs CogVideoX 1.5 (patch_size_t=2), got {dit.patch_size_t}."
        )
    logging.info(
        "[SparkVSR] native DiT OK: layers=%d heads=%d in=%d ofs=%s rope=%s "
        "latent_scale=%s",
        len(dit.blocks), dit.num_attention_heads, dit.in_channels,
        dit.ofs_proj_dim, dit.use_rotary_positional_embeddings,
        getattr(base.latent_format, "scale_factor", None),
    )
