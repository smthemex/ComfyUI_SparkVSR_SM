"""ComfyUI nodes for the native SparkVSR stack.

All nodes use the modern `comfy_api.latest.io` API, so no ComfyUI source file
needs to be modified. Heavyweight old wrapper modules are never imported at
module load time -> transformers 5.x / diffusers 0.42 / peft 0.21 cannot make
these nodes disappear.
"""

import base64
import io
import logging
import os
import sys

import folder_paths
import numpy as np
import torch
from typing_extensions import override

from comfy_api.latest import ComfyExtension, io, ui, Types

from .spark_core import infer_sparkvsr
from .pisa_core import infer_pisa
from .video_utils import parse_indices

MAX_SEED = np.iinfo(np.int32).max

# No model-loading node lives in this plugin. SparkVSR checkpoints are loaded
# with ComfyUI's stock UNetLoader / Dif_GGUF_Loader after the one-time key
# conversion done by tools/convert_sparkvsr_keys.py.


def _video_files():
    input_dir = folder_paths.get_input_directory()
    files = [f for f in os.listdir(input_dir)
             if os.path.isfile(os.path.join(input_dir, f))]
    return sorted(folder_paths.filter_files_content_types(files, ["video"]))


def _parse_hw(text, default=(0, 0)):
    try:
        a, b = str(text).replace("，", ",").split(",", 1)
        return (int(a.strip()), int(b.strip()))
    except Exception:
        return default


class SparkVSR_Native_VideoSR(io.ComfyNode):
    """One-step CogVideoX super-resolution (the SparkVSR S2 pipeline)."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SparkVSR_Native_VideoSR",
            display_name="SparkVSR Native Video SR",
            category="SparkVSR_Native",
            description=(
                "Single-step SparkVSR video super-resolution on ComfyUI's "
                "native CogVideoX DiT. MODEL comes from the STOCK UNetLoader "
                "using 'SparkVSR-S2-bf16-comfy.safetensors' (converted once by "
                "tools/convert_sparkvsr_keys.py), or a Dif_GGUF_Loader GGUF. "
                "Connect a VAE (CogVideoX1.5-5B-I2V-VAE) and the LQ frames. "
                "ref_indices are SOURCE VIDEO frame numbers (e.g. '0,16,32'); "
                "ref_images must be the SAME count, HR keyframes (from PiSA-SR). "
                "SPEED: for ~960x536 use vae_tile_size=48 and "
                "tile_size_hw='256,256'. The '0,0' default runs one huge "
                "~7920-token DiT pass and is much slower on a 12 GB card."
            ),
            inputs=[
                io.Model.Input("model"),
                io.Vae.Input("vae"),
                io.Image.Input("images"),
                io.Image.Input("ref_images", optional=True),
                io.String.Input("ref_indices", default="", multiline=False,
                                tooltip="Comma-separated source frame indices"),
                io.Int.Input("upscale", default=4, min=1, max=16),
                io.Int.Input("target_width", default=0, min=0, max=16384, step=32),
                io.Int.Input("target_height", default=0, min=0, max=16384, step=32),
                io.Combo.Input("upscale_mode",
                               options=["bilinear", "bicubic", "nearest-exact"],
                               default="bilinear"),
                io.Int.Input("chunk_len", default=0, min=0, max=4096, step=8),
                io.Int.Input("overlap_t", default=8, min=0, max=512, step=8),
                io.Combo.Input("tile_size_hw",
                               options=["0,0", "256,256", "384,384", "512,512",
                                        "768,768", "1024,1024"],
                               default="0,0"),
                io.Combo.Input("overlap_hw",
                               options=["32,32", "48,48", "64,64", "96,96"],
                               default="32,32"),
                io.Int.Input("noise_step", default=0, min=0, max=999),
                io.Int.Input("sr_noise_step", default=399, min=1, max=999),
                io.Float.Input("ref_guidance_scale", default=1.0,
                               min=0.0, max=100.0, step=0.1),
                io.Int.Input("seed", default=0, min=0, max=MAX_SEED),
                io.Boolean.Input("posterior_sample", default=True,
                                 tooltip="Reproduce diffusers latent_dist.sample(); "
                                         "off uses the deterministic VAE mode."),
                io.Int.Input("vae_tile_size", default=48, min=0, max=256, step=8,
                             tooltip="CogVideoX VAE decode tile in LATENT pixels "
                                     "(48 ~ 384px). Explicit tiling is MUCH faster "
                                     "than 0/comfy-auto on a 12 GB card."),
            ],
            outputs=[io.Image.Output(display_name="images")],
        )

    @classmethod
    def execute(cls, model, vae, images, ref_indices, upscale, target_width,
                target_height, upscale_mode, chunk_len, overlap_t, tile_size_hw,
                overlap_hw, noise_step, sr_noise_step, ref_guidance_scale, seed,
                posterior_sample, vae_tile_size, ref_images=None) -> io.NodeOutput:
        idxs = parse_indices(ref_indices)
        if ref_images is None:
            idxs = []
        elif len(idxs) != ref_images.shape[0]:
            raise ValueError(
                f"ref_images has {ref_images.shape[0]} frames but ref_indices "
                f"lists {len(idxs)}: {idxs}. They must match 1:1."
            )
        if len(idxs) > 1:
            for a, b in zip(idxs, idxs[1:]):
                if b - a < 4:
                    raise ValueError(
                        "Reference frame interval must be >= 4 (official "
                        f"constraint); got {a} and {b}."
                    )
        out_res = None
        if target_width > 0 and target_height > 0:
            out_res = (target_height, target_width)

        import comfy.utils
        pbar = [None]

        def cb(done, total):
            if pbar[0] is None:
                pbar[0] = comfy.utils.ProgressBar(total)
            pbar[0].update(1)

        result = infer_sparkvsr(
            model, vae, images,
            ref_images=ref_images,
            ref_indices=idxs,
            upscale=upscale,
            output_resolution=out_res,
            upscale_mode=upscale_mode,
            chunk_len=chunk_len,
            overlap_t=overlap_t,
            tile_size_hw=_parse_hw(tile_size_hw),
            overlap_hw=_parse_hw(overlap_hw, (32, 32)),
            noise_step=noise_step,
            sr_noise_step=sr_noise_step,
            ref_guidance_scale=ref_guidance_scale,
            seed=seed,
            prompt_emb=None,          # built-in finetune/SparkVSR_pos_sm.pt
            sample_posterior=posterior_sample,
            vae_tile_size=vae_tile_size,
            progress_callback=cb,
        )
        return io.NodeOutput(result)


class SparkVSR_Native_PiSA_SR(io.ComfyNode):
    """PiSA-SR single-image super-resolution (produces HR reference frames)."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SparkVSR_Native_PiSA_SR",
            display_name="SparkVSR Native PiSA-SR",
            category="SparkVSR_Native",
            description=(
                "PiSA-SR single-image super-resolution. IMPORTANT: UNetLoader "
                "MUST point to 'pisa_sr_unet_merged.safetensors' (the SD2.1 UNet "
                "WITH the six PiSA LoRA adapters already merged). Loading plain "
                "'sd21fp16.safetensors' produces colour noise - the node now "
                "detects and rejects it. VAE = sd21vae.safetensors. Output can "
                "be fed to VideoSR as ref_images."
            ),
            inputs=[
                io.Model.Input("unet"),
                io.Vae.Input("vae"),
                io.Image.Input("image"),
                io.Int.Input("upscale", default=4, min=1, max=8),
                io.Combo.Input("align_method",
                               options=["wavelet", "adain", "none"],
                               default="wavelet"),
                io.Int.Input("process_size", default=512, min=64, max=2048, step=64),
                io.Int.Input("latent_tile_size", default=96, min=0, max=512, step=16),
                io.Int.Input("latent_tile_overlap", default=32, min=0, max=256, step=8),
                io.Int.Input("seed", default=666, min=0, max=MAX_SEED),
            ],
            outputs=[io.Image.Output(display_name="images")],
        )

    @classmethod
    def execute(cls, unet, vae, image, upscale, align_method, process_size,
                latent_tile_size, latent_tile_overlap, seed) -> io.NodeOutput:
        # Guard against the un-merged SD2.1 base UNet: PiSA-SR would output
        # colour noise because x_denoised = z - model_pred is dominated by a
        # base-model eps prediction with ~5x the magnitude of a scaled latent.
        try:
            from safetensors import safe_open
            base = folder_paths.get_full_path("diffusion_models",
                                             "sd21fp16.safetensors")
            if base and os.path.isfile(base):
                w = unet.model.diffusion_model.input_blocks[0][0].weight \
                    .detach().float().cpu().flatten()
                with safe_open(base, framework="pt", device="cpu") as f:
                    ref = f.get_tensor("conv_in.weight").float().flatten()
                if w.shape == ref.shape and torch.allclose(w, ref, atol=1e-4):
                    raise RuntimeError(
                        "PiSA-SR needs the LoRA-MERGED UNet: switch UNetLoader "
                        "to 'pisa_sr_unet_merged.safetensors'. The connected "
                        "model is the un-merged sd21fp16 base, which produces "
                        "colour noise. (Regenerate with tools/merge_pisa_lora.py.)"
                    )
        except RuntimeError:
            raise
        except Exception:
            pass

        out = infer_pisa(
            unet, vae, image,
            upscale=upscale,
            align_method=align_method,
            process_size=process_size,
            tile_size=latent_tile_size,
            tile_overlap=latent_tile_overlap,
            seed=seed,
            prompt_emb=None,          # built-in finetune/sd2_pos_emptyemb_sm_.pt
        )
        return io.NodeOutput(out)


class SparkVSR_Native_LoadVideoKeyframes(io.ComfyNode):
    """LoadVideo-style node with a clickable keyframe picker.

    The web extension (web/js/sparkvsr_keyframes.js) renders a thumbnail grid
    from /sparkvsr/frames; clicking writes the SOURCE frame numbers into the
    `selected_indices` widget, which is also exported downstream.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SparkVSR_Native_LoadVideoKeyframes",
            display_name="SparkVSR Load Video + Pick Keyframes",
            category="SparkVSR_Native",
            has_intermediate_output=True,
            description=(
                "Loads a video like ComfyUI's standard LoadVideo node, then lets "
                "you click any number of keyframes in the thumbnail grid. "
                "`indices` is a comma-separated list of SOURCE frame numbers "
                "for SparkVSR Native Video SR; `selected_images` are the picked "
                "frames (feed them through PiSA-SR first for best results)."
            ),
            inputs=[
                io.Combo.Input("video_file", options=_video_files(),
                               upload=io.UploadType.video),
                io.String.Input("selected_indices", default="0", multiline=False),
                io.Int.Input("max_frames", default=0, min=0, max=100000,
                             tooltip="0 = load all frames"),
            ],
            outputs=[
                io.Image.Output(display_name="images"),
                io.Image.Output(display_name="selected_images"),
                io.String.Output(display_name="indices"),
                io.Int.Output(display_name="frame_count"),
                io.Video.Output(display_name="video"),
            ],
        )

    @classmethod
    def execute(cls, video_file, selected_indices, max_frames) -> io.NodeOutput:
        from comfy_api.input_impl import VideoFromFile
        path = folder_paths.get_annotated_filepath(video_file)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Invalid video file: {video_file}")
        source = VideoFromFile(path)
        images = source.get_components().images.float()
        if max_frames > 0:
            images = images[:max_frames]
        count = images.shape[0]

        idxs = [i for i in parse_indices(selected_indices) if i < count]
        if not idxs:
            idxs = [0]
        picked = images[idxs]
        text = ",".join(str(i) for i in idxs)

        name, _ = folder_paths.annotated_filepath(video_file)
        subfolder, _, filename = name.replace("\\", "/").rpartition("/")
        preview = ui.PreviewVideo(
            [ui.SavedResult(filename, subfolder, io.FolderType.input)]
        )
        return io.NodeOutput(images, picked, text, count, source, ui=preview)

    @classmethod
    def validate_inputs(cls, video_file, **kwargs):
        if not folder_paths.exists_annotated_filepath(video_file):
            return "Invalid video file: {}".format(video_file)
        return True

    @classmethod
    def fingerprint_inputs(cls, **kwargs):
        return ""


# ---------------------------------------------------------------------------
# HTTP endpoint used by the keyframe picker UI. Registered only when the
# ComfyUI server is available (headless import / smoke tests stay safe).
# ---------------------------------------------------------------------------
def _register_routes():
    # ComfyUI creates PromptServer.instance BEFORE loading custom_nodes, so
    # `server` is already in sys.modules at that point. Looking it up instead
    # of importing it keeps headless/unit-test imports free of the server's
    # (and transformers') import graph.
    server_mod = sys.modules.get("server")
    if server_mod is None:
        return
    PromptServer = getattr(server_mod, "PromptServer", None)
    srv = getattr(PromptServer, "instance", None)
    if srv is None or not hasattr(srv, "routes"):
        return
    try:
        from aiohttp import web
    except Exception as e:
        logging.info("[SparkVSR] frame endpoint unavailable: %s", e)
        return

    @srv.routes.get("/sparkvsr/frames")
    async def _sparkvsr_frames(request):
        try:
            import av
            name = request.query.get("file", "")
            max_thumbs = int(request.query.get("max", "120"))
            path = folder_paths.get_annotated_filepath(name)
            if not path or not os.path.isfile(path):
                return web.json_response({"error": "file not found"}, status=404)
            container = av.open(path)
            stream = container.streams.video[0]
            total = int(getattr(stream, "frames", 0) or 0)
            step = max(1, (total + max_thumbs - 1) // max_thumbs) if total else 1
            thumbs = []
            for i, frame in enumerate(container.decode(stream)):
                if i % step:
                    continue
                img = frame.to_image().convert("RGB")
                img.thumbnail((160, 100))
                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=70)
                thumbs.append({
                    "index": i,
                    "data": base64.b64encode(buf.getvalue()).decode("ascii"),
                })
                if len(thumbs) >= max_thumbs:
                    break
            container.close()
            if not total:
                total = (len(thumbs) - 1) * step + 1
            return web.json_response({
                "frames": thumbs, "count": total, "step": step,
            })
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)


_register_routes()


NODE_LIST = [
    SparkVSR_Native_VideoSR,
    SparkVSR_Native_PiSA_SR,
    SparkVSR_Native_LoadVideoKeyframes,
]


class SparkVSRNativeExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return NODE_LIST
