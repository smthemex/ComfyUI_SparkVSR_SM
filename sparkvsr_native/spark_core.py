"""SparkVSR video super-resolution on ComfyUI's native CogVideoX DiT core.

Reproduces `sparkvsr_inference_script.py::process_video_ref_i2v` exactly, but
with the diffusers pipeline replaced by ComfyUI's own pieces:

    diffusers CogVideoXImageToVideoPipeline / transformer / vae / DPM scheduler
        -> comfy.model_base.CogVideoX.apply_model + comfy VAE + torch math

Key semantics that MUST be preserved (see README_NATIVE.md):
  1. ONE denoising step at timestep `sr_noise_step` (default 399), not a
     KSampler loop. The DiT predicts velocity v; x0 = sqrt(a)*x - sqrt(1-a)*v.
  2. The 32 input channels are [LQ latent (16) | HR reference latent (16)]
     concatenated on the CHANNEL axis (ComfyUI's c_concat). Reference frames
     are written into full_ref_latent at latent time index (frame_idx // 4).
  3. ref-CFG: uncond = reference latent all zeros, CFG is applied on the
     VELOCITY (not on x0), with the official `lq_sample = cond half` quirk.
  4. CogVideoX 1.5 patch_size_t=2: the official code duplicates the FIRST
     frame at the FRONT when F_latent is odd. ComfyUI's pad_to_patch_size pads
     at the END, so we pre-pad ourselves and crop the front afterwards.
  5. ofs = 2.0, empty T5 prompt embedding (1, 77, 4096).
  6. VAE latent is SCALED by 0.7 on the DiT side; VAE.decode takes the
     UNSCALED latent (x0 / 0.7) and returns IMAGE in [0,1].
  7. diffusers uses posterior.sample(); ComfyUI's CogVideoX VAE.encode()
     returns only the mode. We therefore sample the posterior ourselves.
"""

import os
import logging

import torch
import torch.nn.functional as F

from .schedule import (
    RawVelocitySampling,
    add_noise,
    get_velocity,
    sigma_for_timestep,
)
from .video_utils import (
    preprocess_frames,
    remove_padding,
    make_temporal_chunks,
    make_spatial_tiles,
    get_valid_tile_region,
)

LATENT_SCALE = 0.7          # CogVideoX1.5-5B VAE scaling_factor
OFS_VALUE = 2.0             # official: torch.full(..., 2.0)
TEMPORAL_RATIO = 4          # pixel frame index // 4 -> latent frame index

_PLUGIN_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_PROMPT_EMB = os.path.join(_PLUGIN_ROOT, "finetune", "SparkVSR_pos_sm.pt")


def load_prompt_emb(path=None):
    """Empty T5 prompt embedding, shape [1, 77, 4096] (fp32, CPU)."""
    path = path or DEFAULT_PROMPT_EMB
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Prompt embedding not found: {path}. Keep finetune/SparkVSR_pos_sm.pt "
            "from the original plugin, or pass prompt_emb_path."
        )
    emb = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(emb, dict):
        emb = next(iter(emb.values()))
    if not torch.is_tensor(emb):
        raise RuntimeError(f"Unsupported prompt embedding file: {path}")
    return emb.float()


def _evict_foreign_models(keep_patchers):
    """Unload every loaded model that is NOT in keep_patchers.

    This comfy build uses dynamic VRAM loading (ModelPatcherDynamic + VBAR
    staged weights).  Its free_memory() is called with for_dynamic=True
    whenever ALL models involved are dynamic, and that path REFUSES to
    unload dynamic models for other dynamic models ("works on-demand").
    When a workflow swaps the DiT checkpoint (bf16 -> int8), the stale
    fully-paged-in 10 GB DiT stays resident (node output caches keep the
    patcher referenced, so GC cleanup never fires), the new model pages in
    against the tiny remainder, and the pager thrashes on a 12 GB card --
    which looks like a hang.  Calling free_memory() with keep_loaded=ours
    takes the NON-dynamic path (for_dynamic=False) and properly unloads
    everything else, including dynamic models.
    """
    import comfy.model_management as mm
    keep = []
    stale = []
    for lm in list(mm.current_loaded_models):
        if lm.model is not None and lm.model in keep_patchers:
            keep.append(lm)
        elif not lm.is_dead():
            stale.append(lm.model.__class__.__name__)
    if not stale:
        return
    logging.info("[SparkVSR] evicting stale model(s) before load: %s", stale)
    mm.free_memory(1e30, mm.get_torch_device(), keep_loaded=keep)
    mm.soft_empty_cache()


def _ensure_loaded(vae, model_patcher, require_dit=True):
    """Keep the DiT and the VAE resident together -- WITHOUT re-entering
    comfy's memory manager on every call.

    load_models_gpu() ALWAYS runs free_memory() with a floor of
    minimum_inference_memory() (~0.8 GB + 0.6 GB Windows reserve).  With a
    ~10 GB DiT plus the VAE resident on a 12 GB card, free VRAM is almost
    always below that floor, so every load_models_gpu() call partially
    offloads DiT weights and streams them back from RAM (a hidden PCIe
    round trip per call).  Calling this per VAE encode was therefore a
    per-tile thrash.  We now check residency cheaply and only call
    load_models_gpu() when a model is genuinely missing (first call, or
    after an external eviction).

    require_dit=False keeps the DiT OUT of the required set: used during
    the encode phase, which deliberately runs with only the VAE resident.

    Do NOT pass vae.memory_used_encode(...) either: comfy's CogVideoX VAE
    estimate is ~43 GB for a 30-frame 960x536 clip on a 12 GB card. That
    makes load_models_gpu() try to free 43 GB, evicting the 10 GB DiT and
    forcing a full CPU->GPU reload on every encode/decode call.
    """
    import comfy.model_management as mm
    loaded = {lm.model for lm in mm.current_loaded_models}
    want = [vae.patcher] + ([model_patcher] if require_dit else [])
    if all(p in loaded for p in want):
        return
    _evict_foreign_models(want)
    mm.load_models_gpu(want, memory_required=0)


def _vae_decode(vae, latent, tile=48):
    """Explicit tiled 3D decode (tile size is in LATENT pixels).

    comfy's VAE.decode():
      1. computes memory_used_decode() ≈ 83 GB for our clip,
      2. attempts an UNTILED decode, OOMs,
      3. falls back to decode_tiled_3d() which averages THREE tiled passes,
      4. calls load_models_gpu([vae.patcher]) which EVICTS the DiT.
    Calling decode_tiled_3d() directly does one pass, never evicts the DiT,
    and keeps peak VRAM under the 12 GB limit (no WDDM shared-memory
    spill, which alone can slow things down 20x).
    """
    if tile <= 0:
        return vae.decode(latent)
    t = max(1, int(latent.shape[2]))
    ov = max(1, int(tile) // 8)
    # process_output() inside decode_tiled_3d is inplace (add_/div_/clamp_),
    # and tiled_scale_multidim allocates its output under inference mode,
    # so the whole call must stay inside torch.inference_mode().
    with torch.inference_mode():
        raw = vae.decode_tiled_3d(
            latent, tile_t=t, tile_x=int(tile), tile_y=int(tile),
            overlap=(1, ov, ov),
        )
    # decode_tiled_3d returns [B,C,F,H,W] already mapped to [0,1];
    # VAE.decode() applies this extra movedim(1,-1) -> [B,F,H,W,C].
    # clone() detaches the inference tensor before it is used in the stitch.
    return raw.clone().movedim(1, -1)


def _rolling_encode(fsm, x, frame_batch):
    """comfy's rolling encoder loop: causal conv cache carried across chunks."""
    t = x.shape[2]
    remainder = t % frame_batch
    conv_cache = None
    enc = []
    with torch.no_grad():
        if remainder > 0:
            chunk, conv_cache = fsm.encoder(x[:, :, :remainder], conv_cache=conv_cache)
            enc.append(chunk.to(x.device))
        for start in range(remainder, t, frame_batch):
            chunk, conv_cache = fsm.encoder(
                x[:, :, start:start + frame_batch], conv_cache=conv_cache
            )
            enc.append(chunk.to(x.device))
    del conv_cache
    return torch.cat(enc, dim=2)


# Diagnostics/test override for the temporal encode batch; None = auto.
_ENCODE_FB_OVERRIDE = None


def _encode_frame_batch(fsm, H, W):
    """Temporal batch for the rolling encoder.

    MEASURED on RTX 4070 (12 GB), 33 frames 544x960 bf16:
        fb=8 -> peak 12.9 GB, 70 s  (spills into WDDM shared memory)
        fb=4 -> peak  8.8 GB, 4.9 s
        fb=1 -> peak  1.4 GB, 3.8 s  but WRONG latent frame count
    fb must stay >= 4: the 4x temporal compression needs 4 frames per chunk,
    smaller batches emit one latent frame per chunk instead (T 9 -> 17/33).
    """
    if _ENCODE_FB_OVERRIDE:
        return int(_ENCODE_FB_OVERRIDE)
    default = int(getattr(fsm, "num_sample_frames_batch_size", 8))
    return default if H * W <= 256 * 256 else min(default, 4)


def cogvae_encode_posterior(vae, model_patcher, x, sample=True, require_dit=True):
    """ComfyUI CogVideoX VAE encode with the official posterior.sample().

    x: [B,3,F,H,W] in [-1,1]. Returns UNSCALED latent [B,16,F_lat,H/8,W/8].
    ComfyUI's own `VAE.encode()` returns only `mean` (vae.py: `mean, _ =
    enc.chunk(2, dim=1)`), so we reproduce the rolling encoder here and keep
    both halves. The temporal batch is reduced on big frames so the encoder
    working set stays inside 12 GB (see _encode_frame_batch).
    """
    _ensure_loaded(vae, model_patcher, require_dit=require_dit)
    fsm = vae.first_stage_model
    if not hasattr(fsm, "num_sample_frames_batch_size"):
        # Not a CogVideoX 3D VAE -> fall back to comfy's public encode.
        return vae.encode(x.movedim(1, -1) if x.ndim == 5 else x)

    x = x.to(device=vae.device, dtype=vae.vae_dtype)
    H, W = x.shape[-2:]
    frame_batch = _encode_frame_batch(fsm, H, W)
    enc = _rolling_encode(fsm, x, frame_batch)
    mean, logvar = enc.chunk(2, dim=1)
    del enc
    if not sample:
        return mean
    # diffusers DiagonalGaussianDistribution: logvar = clamp(logvar, -30, 20)
    std = torch.exp(0.5 * logvar.clamp(-30.0, 20.0))
    return mean + std * torch.randn_like(mean)


def _build_ref_latent(vae, model_patcher, ref_frames, ref_indices, chunk_start,
                      lq_latent, sample=True, lat_cache=None, tile_key=None,
                      require_dit=True):
    """Full reference latent in SCALED space, zeros everywhere except the
    selected keyframe latent time positions. Matches the official loop.

    The ENCODED latent of a reference frame depends only on (ref, spatial
    tile) -- never on the time chunk. The official script re-encodes it for
    every time chunk; we cache it per (ref, tile) instead so a multi-chunk
    video pays one VAE encode per reference per tile, not per chunk.
    (The posterior sample being reused across chunks changes the RNG stream
    vs. re-encoding; it is plain gaussian noise, not a semantic change.)
    """
    ref = torch.zeros_like(lq_latent)
    num_frames = lq_latent.shape[2]
    for i, idx in enumerate(ref_indices):
        if i >= len(ref_frames):
            break
        target = (idx - chunk_start) // TEMPORAL_RATIO
        if not (0 <= target < num_frames):
            continue
        # ref_frames[i]: [3,H,W] in [-1,1], already cropped to this spatial tile
        if lat_cache is not None:
            key = (i, tile_key, sample)
            lat = lat_cache.get(key)
            if lat is None:
                lat = _encode_ref_single(vae, model_patcher, ref_frames[i],
                                         sample, require_dit=require_dit)
                lat_cache[key] = lat
        else:
            lat = _encode_ref_single(vae, model_patcher, ref_frames[i], sample,
                                     require_dit=require_dit)
        ref[:, :, target, :, :] = lat[0, :, 0, :, :] * LATENT_SCALE
        del lat
    return ref


def _encode_ref_single(vae, model_patcher, ref_frame, sample=True,
                       require_dit=True):
    """One reference frame [3,H,W] -> unscaled latent [1,16,1,H/8,W/8]."""
    r = ref_frame.to(device=vae.device, dtype=vae.vae_dtype)
    chunk = r.unsqueeze(0).unsqueeze(2).repeat(1, 1, 4, 1, 1)
    lat = cogvae_encode_posterior(vae, model_patcher, chunk, sample=sample,
                                  require_dit=require_dit)
    del chunk
    return lat


def _encode_chunk(vae, model_patcher, video_chunk, ref_frames, ref_indices,
                  chunk_start, sample_posterior=True, lat_cache=None,
                  tile_key=None, timings=None):
    """Phase A of a tile: LQ latent + reference latents, DiT kept OUT of VRAM.

    Returns (lq, ref) in SCALED latent space. On a 12 GB card a fully paged
    bf16 DiT leaves <1 GB for the VAE encode working set, which made the
    encode crawl in WDDM shared memory (60.9s for one 33-frame tile);
    encoding everything up front with only the 0.4 GB VAE resident removes
    that. NOTE: the posterior-sampling RNG stream order changes vs. the old
    interleaved encode-per-tile flow (plain gaussian noise, no semantic
    change).
    """
    import time
    t0 = time.perf_counter()
    # LQ latent in SCALED space (official: * vae.config.scaling_factor)
    lq = cogvae_encode_posterior(
        vae, model_patcher, video_chunk, sample=sample_posterior,
        require_dit=False,
    ) * LATENT_SCALE

    # HR reference latent, also SCALED
    ref = _build_ref_latent(vae, model_patcher, ref_frames, ref_indices,
                            chunk_start, lq, sample=sample_posterior,
                            lat_cache=lat_cache, tile_key=tile_key,
                            require_dit=False)
    if timings is not None:
        timings["encode"] = timings.get("encode", 0.0) + (time.perf_counter() - t0)
    return lq, ref


def _dit_step(base, lq, ref, chunk_start, prompt_emb, sr_noise_step, noise_step,
              ref_guidance_scale, patch_size_t, timings=None):
    """Phase B of a tile: noise -> single DiT step -> CFG -> x0 (scaled)."""
    import time
    import comfy.model_management as mm
    # All DiT inputs must live on the device comfy put the model on. The
    # model_sampling buffers (sigmas/log_sigmas) follow that device too, so
    # mixing CPU inputs with GPU buffers breaks `emb + ofs_emb` in the DiT.
    device = mm.get_torch_device()
    t0 = time.perf_counter()
    do_cfg = abs(ref_guidance_scale - 1.0) > 1e-3
    if do_cfg:
        x_in = torch.cat([lq, lq], dim=0)
        c_concat = torch.cat([torch.zeros_like(ref), ref], dim=0)
    else:
        x_in = lq
        c_concat = ref
    bsz = x_in.shape[0]

    # optional extra noise on the LQ part only (official noise_step)
    if noise_step != 0:
        noise = torch.randn_like(x_in)
        x_in = add_noise(x_in, noise, [noise_step] * bsz)
        del noise

    # FRONT pad for patch_size_t (ComfyUI pads at the END, official at FRONT)
    ncopy = x_in.shape[2] % patch_size_t if patch_size_t else 0
    if ncopy:
        x_in = torch.cat([x_in[:, :, :1].repeat(1, 1, ncopy, 1, 1), x_in], dim=2)
        c_concat = torch.cat(
            [c_concat[:, :, :1].repeat(1, 1, ncopy, 1, 1), c_concat], dim=2
        )

    # native DiT forward through BaseModel.apply_model.
    #    RawVelocitySampling makes it return v instead of a denoised sample.
    x_in = x_in.to(device)
    c_concat = c_concat.to(device)
    ms = base.model_sampling
    sigma = sigma_for_timestep(ms, sr_noise_step, device).to(device).expand(bsz)
    ctx = prompt_emb.to(device=device, dtype=x_in.dtype).repeat(bsz, 1, 1)
    ofs = torch.full((bsz,), OFS_VALUE, device=device, dtype=x_in.dtype)
    orig_ms = base.model_sampling
    base.model_sampling = RawVelocitySampling(orig_ms)
    try:
        with torch.no_grad():
            v = base.apply_model(
                x_in, sigma,
                c_concat=c_concat,
                c_crossattn=ctx,
                ofs=ofs,
                transformer_options={},
            )
    finally:
        base.model_sampling = orig_ms
    v = v.to(x_in.dtype)
    del ctx, ofs, c_concat
    t_dit = time.perf_counter() - t0
    if timings is not None:
        timings["dit"] = timings.get("dit", 0.0) + t_dit
    t0 = time.perf_counter()

    # official CFG: applied on velocity; x_t taken from the COND half
    if do_cfg:
        v_u, v_c = v.chunk(2, dim=0)
        v = v_u + ref_guidance_scale * (v_c - v_u)
        x_for_x0 = x_in.chunk(2, dim=0)[1]
        del v_u, v_c
    else:
        x_for_x0 = x_in
    del x_in

    # x0 = sqrt(a)*x_t - sqrt(1-a)*v   (scaled latent space)
    x0 = get_velocity(v, x_for_x0, [sr_noise_step] * x_for_x0.shape[0])
    if ncopy:
        x0 = x0[:, :, ncopy:, :, :]
    del v, x_for_x0
    if timings is not None:
        timings["dit"] = timings.get("dit", 0.0) + (time.perf_counter() - t0)
    return x0


def _log_vram(tag):
    """Log real VRAM numbers + which models are resident. Never guess."""
    import comfy.model_management as mm
    try:
        dev = mm.get_torch_device()
        free = mm.get_free_memory(dev) / 1e9
        total = mm.get_total_memory(dev) / 1e9
        alloc = torch.cuda.memory_allocated(dev) / 1e9
        names = []
        for lm in mm.current_loaded_models:
            if lm.model is not None:
                try:
                    names.append(lm.model.model.__class__.__name__)
                except Exception:
                    names.append("?")
        logging.info(
            "[SparkVSR] VRAM %s: free=%.2f/%.2f GB (alloc %.2f) resident=%s",
            tag, free, total, alloc, names or "none",
        )
    except Exception:
        logging.exception("[SparkVSR] VRAM log failed")


def _force_offload(patcher, why=""):
    """Page a dynamic patcher's weights FULLY out of VRAM, then verify.

    free_memory()'s heuristics are not reliable for ModelPatcherDynamic:
    the dynamic-model exemption can skip a model, and detach() only releases
    the VBAR when unpatch_weights is honoured. partially_unload(..., 1e32)
    calls vbar.free_memory(1e32), which really gives the pages back. We then
    log the delta so an "evicted but still resident" state cannot pass
    unnoticed.
    """
    import comfy.model_management as mm
    try:
        if patcher is None:
            return
        before = mm.get_free_memory(mm.get_torch_device()) / 1e9
        size_before = patcher.loaded_size() / 1e6
        if size_before > 0:
            patcher.partially_unload(patcher.offload_device, 1e32)
        mm.soft_empty_cache()
        after = mm.get_free_memory(mm.get_torch_device()) / 1e9
        logging.info(
            "[SparkVSR] offload %s (%s): loaded %.0fMB -> %.0fMB, free VRAM "
            "%.2f -> %.2f GB",
            why or "model", patcher.model.__class__.__name__,
            size_before, patcher.loaded_size() / 1e6, before, after,
        )
    except Exception:
        logging.exception("[SparkVSR] force offload failed")


def _phase_a_prepare(vae, work):
    """VRAM setup for the encode phase: the DiT must NOT be on the GPU.

    Encode runs a full-frame 8 x 544 x 960 x 128ch working set; if the DiT
    keeps any pages in VRAM the encoder crawls through WDDM shared memory
    (observed 44-68s vs 6s warm). We page the DiT out explicitly (verified
    by loaded_size) and load only the 0.4 GB VAE.
    """
    import comfy.model_management as mm
    _force_offload(work, "DiT (before encode)")
    _evict_foreign_models([vae.patcher])
    mm.load_models_gpu([vae.patcher], memory_required=0)
    _log_vram("phase A encode")



def infer_sparkvsr(model, vae, images, *, ref_images=None, ref_indices=None,
                   upscale=4, output_resolution=None, upscale_mode="bilinear",
                   chunk_len=0, overlap_t=8, tile_size_hw=(0, 0),
                   overlap_hw=(32, 32), noise_step=0, sr_noise_step=399,
                   ref_guidance_scale=1.0, seed=0, prompt_emb=None,
                   sample_posterior=True, vae_tile_size=48,
                   progress_callback=None):
    """Full SparkVSR inference. Returns ComfyUI IMAGE [F,H,W,C] in [0,1]."""
    import comfy.model_management as mm

    if prompt_emb is None:
        prompt_emb = load_prompt_emb()
    if isinstance(prompt_emb, str):
        prompt_emb = load_prompt_emb(prompt_emb)

    ref_indices = list(ref_indices or [])
    ref_frames = []
    if ref_images is not None and len(ref_indices) > 0:
        # IMAGE [N,H,W,C] in [0,1] -> list of [3,H,W] in [-1,1]
        ri = ref_images.float()
        if ri.shape[0] != len(ref_indices):
            raise ValueError(
                f"ref_images count {ri.shape[0]} != ref_indices count "
                f"{len(ref_indices)}; they must match one-to-one."
            )
        ref_frames = [ri[i].permute(2, 0, 1) * 2.0 - 1.0 for i in range(ri.shape[0])]

    video, _, pad_f, pad_h, pad_w, eff_up, _ = preprocess_frames(
        images, upscale=upscale, output_resolution=output_resolution,
        upscale_mode=upscale_mode,
    )

    # CogVideoX 1.5 patch_size_t=2 requires latent H/W to be even.
    # ComfyUI's pad_to_patch_size pads at the END (different RoPE grid), so
    # make the HR size a multiple of 16 here and crop it away at the end.
    hr_pad_h = (16 - video.shape[-2] % 16) % 16
    hr_pad_w = (16 - video.shape[-1] % 16) % 16
    if hr_pad_h or hr_pad_w:
        video = F.pad(video, (0, hr_pad_w, 0, hr_pad_h), value=-1.0)

    torch.manual_seed(int(seed))

    # NOTE: on this ComfyUI build load_models_gpu() does NOT call
    # ModelPatcher.patch_model(), so add_object_patch("model_sampling", ...)
    # never takes effect when we bypass KSampler. We swap the sampling object
    # explicitly, scoped to each single apply_model() call (see
    # _dit_step), and restore it immediately afterwards.
    work = model
    base = model.model
    # NOTE: the workflow is split into three phases so the DiT is NEVER
    # resident while the VAE encodes (phase A) -- on a 12 GB card a paged-in
    # bf16 DiT leaves <1 GB for the encode working set, which crawled in
    # WDDM shared memory. The DiT loads for phase B only, after every latent
    # is already on the GPU.
    dit = base.diffusion_model
    patch_size_t = getattr(dit, "patch_size_t", None) or 1

    B, C, F_len, H, W = video.shape
    use_t = overlap_t if chunk_len > 0 else 0
    use_hw = tuple(overlap_hw) if tuple(tile_size_hw) != (0, 0) else (0, 0)
    time_chunks = make_temporal_chunks(F_len, chunk_len, use_t)
    spatial_tiles = make_spatial_tiles(H, W, tuple(tile_size_hw), use_hw)
    logging.info(
        "[SparkVSR] F=%d H=%d W=%d chunks=%d tiles=%d t=%d cfg=%.2f (phase-split)",
        F_len, H, W, len(time_chunks), len(spatial_tiles),
        sr_noise_step, ref_guidance_scale,
    )

    output = torch.zeros_like(video)
    tile_jobs = max(1, len(time_chunks) * len(spatial_tiles))
    total = tile_jobs * 3          # encode + dit + decode per tile
    done = 0
    # Reference images are loop-invariant: resize each one to the padded HR
    # size ONCE instead of re-running F.interpolate inside every (chunk x
    # tile) iteration.
    if ref_frames:
        resized = []
        for rf in ref_frames:
            if rf.shape[-2:] != (H, W):
                rf = F.interpolate(rf.unsqueeze(0), size=(H, W),
                                   mode="bilinear",
                                   align_corners=False).squeeze(0)
            resized.append(rf)
        ref_frames = resized
    # Encoded reference latents depend only on (ref, spatial tile), so they
    # are shared across time chunks. Key: (ref_i, tile_key, sample_flag).
    ref_lat_cache = {}
    timings = {"encode": 0.0, "dit": 0.0, "decode": 0.0}
    import time as _time
    _t_all = _time.perf_counter()

    # ---- PHASE A: encode ALL LQ chunks + reference latents, VAE only. ----
    _phase_a_prepare(vae, work)
    enc_store = {}                 # (t0,t1,h0,h1,w0,w1) -> (lq, ref) scaled
    for t_start, t_end in time_chunks:
        for h_start, h_end, w_start, w_end in spatial_tiles:
            tile_key = (h_start, h_end, w_start, w_end)
            chunk = video[:, :, t_start:t_end, h_start:h_end, w_start:w_end]
            cropped_refs = []
            if ref_frames:
                # reference images are full-size [3,H,W]; crop the same tile
                for rf in ref_frames:
                    cropped_refs.append(rf[:, h_start:h_end, w_start:w_end])
            enc_store[(t_start, t_end) + tile_key] = _encode_chunk(
                vae, work, chunk, cropped_refs, ref_indices, t_start,
                sample_posterior, lat_cache=ref_lat_cache, tile_key=tile_key,
                timings=timings,
            )
            del chunk, cropped_refs
            done += 1
            if progress_callback is not None:
                progress_callback(done, total)

    # ---- PHASE B: single DiT step per stored latent (DiT now resident). ----
    _ensure_loaded(vae, work)
    _log_vram("phase B dit")
    x0_store = {}
    for key, (lq, ref) in enc_store.items():
        x0_store[key] = _dit_step(
            base, lq, ref, key[0], prompt_emb, sr_noise_step, noise_step,
            ref_guidance_scale, patch_size_t, timings=timings,
        )
        done += 1
        if progress_callback is not None:
            progress_callback(done, total)
    del enc_store

    # ---- PHASE C: tiled decode of every x0 latent and stitch. ----
    # The DiT goes back to CPU first: decode only needs the VAE, and any
    # resident DiT pages push the tiled decode into WDDM shared memory.
    # x0 latents are tiny; the DiT re-pages from the staged mmap on demand.
    _force_offload(work, "DiT (before decode)")
    _evict_foreign_models([vae.patcher])
    mm.load_models_gpu([vae.patcher], memory_required=0)
    _log_vram("phase C decode")
    for (t_start, t_end, h_start, h_end, w_start, w_end), x0 in x0_store.items():
        t0 = _time.perf_counter()
        with torch.no_grad():
            gen = _vae_decode(vae, x0 / LATENT_SCALE, vae_tile_size)
        del x0
        timings["decode"] = timings.get("decode", 0.0) + (_time.perf_counter() - t0)
        gen = gen.permute(0, 4, 1, 2, 3).contiguous()  # [B,C,F,H,W] in [0,1]
        r = get_valid_tile_region(
            t_start, t_end, h_start, h_end, w_start, w_end,
            video.shape, use_t, use_hw[0], use_hw[1],
        )
        output[:, :, r["ot0"]:r["ot1"], r["oh0"]:r["oh1"], r["ow0"]:r["ow1"]] = \
            gen[:, :, r["vt0"]:r["vt1"], r["vh0"]:r["vh1"], r["vw0"]:r["vw1"]]
        del gen
        done += 1
        if progress_callback is not None:
            progress_callback(done, total)
    del x0_store

    _wall = _time.perf_counter() - _t_all
    logging.info(
        "[SparkVSR] tiles=%d wall=%.1fs | encode=%.1fs dit=%.1fs decode=%.1fs "
        "(residual %.1fs = stitching/transfer)",
        done // 3, _wall, timings["encode"], timings["dit"], timings["decode"],
        max(0.0, _wall - timings["encode"] - timings["dit"] - timings["decode"]),
    )

    # [B,C,F,H,W] -> [B,F,H,W,C] -> crop pads -> [F,H,W,C]
    out = output.permute(0, 2, 3, 4, 1).contiguous()
    out = remove_padding(
        out, pad_f, pad_h * eff_up + hr_pad_h, pad_w * eff_up + hr_pad_w
    )
    del output, video
    mm.soft_empty_cache()
    return out[0].clamp(0.0, 1.0)
