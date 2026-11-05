"""PiSA-SR single-image super-resolution on ComfyUI's native SD2.1 UNet.

The original plugin builds a diffusers UNet, adds six PEFT LoRA adapters
(pix/sem x encoder/decoder/others), merges pix, then merges sem, and finally
runs ONE step at timestep=1:

    encoded_control = vae.encode(c_t).latent_dist.sample() * 0.18215
    model_pred      = unet(encoded_control, 1, encoder_hidden_states=empty_prompt)
    x_denoised      = encoded_control - model_pred
    output          = vae.decode(x_denoised / 0.18215) * 0.5 + 0.5

So the merged model is just a plain SD2.1 UNet. `tools/merge_pisa_lora.py`
merges the six adapters offline (pure torch, no PEFT); at runtime we only need
ComfyUI's UNetModel + VAE, driven by RawVelocitySampling to get the raw
`model_pred` (ComfyUI's EPS.calculate_denoised would multiply it by sigma).
"""

import os
import importlib.util
import logging

import torch
import torch.nn.functional as F

from .schedule import RawVelocitySampling, sigma_for_timestep

SD_LATENT_SCALE = 0.18215
_PLUGIN_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_SD2_PROMPT_EMB = os.path.join(
    _PLUGIN_ROOT, "finetune", "sd2_pos_emptyemb_sm_.pt"
)


def load_sd2_prompt_emb(path=None):
    path = path or DEFAULT_SD2_PROMPT_EMB
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"SD2 empty prompt embedding not found: {path}. Keep "
            "finetune/sd2_pos_emptyemb_sm_.pt from the original plugin."
        )
    e = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(e, dict):
        e = next(iter(e.values()))
    return e.float()


def _gaussian_weights(tile, channels, device):
    import math
    m = (tile - 1) / 2
    xp = [math.exp(-((x - m) ** 2) / (2 * (tile ** 2) * 0.01)) /
          math.sqrt(2 * math.pi * 0.01) for x in range(tile)]
    yp = [math.exp(-((y - m) ** 2) / (2 * (tile ** 2) * 0.01)) /
          math.sqrt(2 * math.pi * 0.01) for y in range(tile)]
    w = torch.tensor([[a * b for b in xp] for a in yp], device=device)
    return w.view(1, 1, tile, tile).repeat(1, channels, 1, 1)


def _sdvae_encode_sample(vae, model_patcher, pixel, sample=True):
    """pixel: [B,C,H,W] in [-1,1] -> SCALED latent.

    CRITICAL: comfy's AutoencodingEngineLegacy.encode() is
        encoder(x) -> quant_conv(z) -> regularization(posterior.mode())
    Calling `fsm.encoder(x)` alone SKIPS quant_conv; that raw tensor is NOT
    a (mean, logvar) pair. Its second half reaches ~110, so
    exp(0.5 * logvar) explodes (posterior std ~75, latent std ~192) and the
    UNet saturates -> the colour-noise image seen in the first build.
    Always pass through quant_conv (or use comfy's public VAE.encode).
    """
    import comfy.model_management as mm
    bhwc = pixel.movedim(1, -1).clamp(-1.0, 1.0) * 0.5 + 0.5  # [B,H,W,C] 0..1
    mm.load_models_gpu(
        [model_patcher, vae.patcher],
        memory_required=vae.memory_used_encode(pixel.shape, vae.vae_dtype),
    )
    fsm = vae.first_stage_model
    z = None
    if hasattr(fsm, "quant_conv"):
        x = vae.process_input(bhwc.movedim(-1, 1)).to(
            device=vae.device, dtype=vae.vae_dtype)
        with torch.no_grad():
            moments = fsm.quant_conv(fsm.encoder(x))
        if moments.shape[1] == 8:
            mean, logvar = moments.chunk(2, dim=1)
            logvar = logvar.clamp(-30.0, 20.0)
            if sample:
                z = mean + torch.exp(0.5 * logvar) * torch.randn_like(mean)
            else:
                z = mean
    if z is None:
        z = vae.encode(bhwc)

    # Sanity guard: SD2.1 latent is O(1). Never let a broken path reach the
    # UNet silently (that is exactly what produced the noise image before).
    mag = float(z.detach().float().abs().mean())
    if not (0.0 < mag < 10.0):
        logging.warning(
            "[SparkVSR] SD VAE latent magnitude %.3f is abnormal; "
            "falling back to comfy VAE.encode()", mag
        )
        z = vae.encode(bhwc)
    return z * SD_LATENT_SCALE


def _unet_predict(base, model_patcher, z, prompt_emb, tile_size=96, overlap=32):
    """Raw model_pred; optional Gaussian-weighted latent tiling (official)."""
    import comfy.model_management as mm
    device = mm.get_torch_device()
    z = z.to(device)
    ms = base.model_sampling
    sigma = sigma_for_timestep(ms, 1, device).to(device).expand(z.shape[0])
    ctx = prompt_emb.to(device=device, dtype=z.dtype).repeat(z.shape[0], 1, 1)

    def run(tiles):
        orig_ms = base.model_sampling
        base.model_sampling = RawVelocitySampling(orig_ms)
        try:
            with torch.no_grad():
                return base.apply_model(
                    tiles, sigma.expand(tiles.shape[0]),
                    c_crossattn=ctx.repeat(tiles.shape[0], 1, 1),
                    transformer_options={},
                ).to(z.dtype)
        finally:
            base.model_sampling = orig_ms

    h, w = z.shape[-2:]
    if tile_size <= 0 or h * w <= tile_size * tile_size:
        return run(z)

    ts = min(tile_size, min(h, w))
    rows = 0
    cur = 0
    while cur < w:
        cur = max(rows * ts - overlap * rows, 0) + ts
        rows += 1
    cols = 0
    cur = 0
    while cur < h:
        cur = max(cols * ts - overlap * cols, 0) + ts
        cols += 1
    weights = _gaussian_weights(ts, z.shape[1], device).to(z.dtype)
    pred = torch.zeros_like(z)
    contrib = torch.zeros_like(z)
    for row in range(rows):
        for col in range(cols):
            ox = max(row * ts - overlap * row, 0)
            oy = max(col * ts - overlap * col, 0)
            if row == rows - 1:
                ox = w - ts
            if col == cols - 1:
                oy = h - ts
            tile = z[:, :, oy:oy + ts, ox:ox + ts]
            out = run(tile)
            pred[:, :, oy:oy + ts, ox:ox + ts] += out * weights
            contrib[:, :, oy:oy + ts, ox:ox + ts] += weights
    del weights
    return pred / contrib.clamp_min(1e-8)


def _color_fix(out_bhwc, source_bhwc, method):
    """Optional wavelet / adain colour fix, loaded lazily from the plugin's
    original PiSA-SR utility (only torch/torchvision/PIL, no HF deps)."""
    if method in (None, "", "none"):
        return out_bhwc
    path = os.path.join(
        _PLUGIN_ROOT, "finetune", "PiSASR", "src", "my_utils",
        "wavelet_color_fix.py",
    )
    if not os.path.isfile(path):
        logging.warning("[SparkVSR] colour-fix helper missing, skipping")
        return out_bhwc
    try:
        spec = importlib.util.spec_from_file_location("_spark_wavelet_color_fix", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        from PIL import Image
        import numpy as np
        fn = mod.wavelet_color_fix if method == "wavelet" else mod.adain_color_fix
        results = []
        for o, s in zip(out_bhwc, source_bhwc):
            to_img = lambda t: Image.fromarray(
                (t.detach().float().cpu().clamp(0, 1).numpy() * 255).astype(np.uint8)
            )
            fixed = fn(to_img(o), to_img(s))
            results.append(torch.from_numpy(
                np.asarray(fixed, dtype=np.float32) / 255.0
            ))
        return torch.stack(results).to(out_bhwc.device, out_bhwc.dtype)
    except Exception as e:
        logging.warning("[SparkVSR] colour-fix failed (%s), skipping", e)
        return out_bhwc


def infer_pisa(model, vae, image, *, upscale=4, prompt_emb=None,
               process_size=512, tile_size=96, tile_overlap=32,
               align_method="wavelet", sample_posterior=True, seed=666):
    """image: ComfyUI IMAGE [N,H,W,C] in [0,1] -> SR IMAGE [N,H',W',C]."""
    import comfy.model_management as mm

    if prompt_emb is None:
        prompt_emb = load_sd2_prompt_emb()
    elif isinstance(prompt_emb, str):
        prompt_emb = load_sd2_prompt_emb(prompt_emb)

    torch.manual_seed(int(seed))
    # Explicit sampling swap (scoped in _unet_predict) instead of
    # add_object_patch: load_models_gpu() does not call patch_model() here.
    work = model
    base = model.model
    mm.load_models_gpu([work], memory_required=0)

    outs = []
    srcs = []
    n = image.shape[0]
    src = image.float()
    H0, W0 = src.shape[1], src.shape[2]
    # official: upscale small inputs to process_size // rscale first
    if W0 < process_size // upscale or H0 < process_size // upscale:
        s = (process_size // upscale) / min(W0, H0)
        src = F.interpolate(src.movedim(-1, 1),
                            size=(int(H0 * s), int(W0 * s)),
                            mode="bilinear", align_corners=False).movedim(1, -1)
    H1, W1 = src.shape[1], src.shape[2]
    src = F.interpolate(src.movedim(-1, 1),
                        size=(H1 * upscale, W1 * upscale),
                        mode="bilinear", align_corners=False).movedim(1, -1)
    # trim to a multiple of 8 (official uses LANCZOS; bilinear is within 1 LSB)
    H2, W2 = src.shape[1] - src.shape[1] % 8, src.shape[2] - src.shape[2] % 8
    src = F.interpolate(src.movedim(-1, 1), size=(H2, W2),
                        mode="bilinear", align_corners=False).movedim(1, -1)

    for i in range(n):
        c_t = src[i:i + 1].movedim(-1, 1) * 2.0 - 1.0
        z = _sdvae_encode_sample(vae, work, c_t, sample=sample_posterior)
        pred = _unet_predict(base, work, z, prompt_emb, tile_size, tile_overlap)
        x_denoised = z - pred
        del z, pred
        with torch.no_grad():
            out = vae.decode(x_denoised / SD_LATENT_SCALE)
        del x_denoised
        outs.append(out[0].clamp(0.0, 1.0))
        srcs.append(src[i])
    del base
    mm.soft_empty_cache()
    out = torch.stack(outs)
    source = torch.stack(srcs)
    return _color_fix(out, source, align_method)
