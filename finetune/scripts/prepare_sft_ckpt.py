import shutil
import json
import argparse
import subprocess
import sys
from safetensors.torch import load_file
import torch
import os

def run_zero_to_fp32(checkpoint_dir, output_dir):
    zero_to_fp32_script = os.path.join(checkpoint_dir, "zero_to_fp32.py")
    subprocess.run([
        sys.executable,
        zero_to_fp32_script,
        checkpoint_dir,
        output_dir,
        "--safe_serialization"
    ], check=True)

def rename_weights(output_dir):
    # Rename index file and update content
    index_file = os.path.join(output_dir, "model.safetensors.index.json")
    if os.path.exists(index_file):
        with open(index_file, "r") as f:
            data = json.load(f)
        
        new_map = {}
        if "weight_map" in data:
            for k, v in data["weight_map"].items():
                new_map[k] = v.replace("model-", "diffusion_pytorch_model-")
            data["weight_map"] = new_map
        
        new_index_file = os.path.join(output_dir, "diffusion_pytorch_model.safetensors.index.json")
        with open(new_index_file, "w") as f:
            json.dump(data, f, indent=2)
        os.remove(index_file)

    # Rename weight files
    for file in os.listdir(output_dir):
        if file.startswith("model-") and file.endswith(".safetensors"):
            old_path = os.path.join(output_dir, file)
            new_path = os.path.join(output_dir, file.replace("model-", "diffusion_pytorch_model-"))
            os.rename(old_path, new_path)

def prepare_ckpt_structure(output_dir, weights_source_dir, ckpt_output_dir):
    if os.path.exists(ckpt_output_dir):
        shutil.rmtree(ckpt_output_dir)
    shutil.copytree(weights_source_dir, ckpt_output_dir)

    transformer_dir = os.path.join(ckpt_output_dir, "transformer")
    
    # Clean transformer dir but keep config.json
    if os.path.exists(transformer_dir):
        for item in os.listdir(transformer_dir):
            item_path = os.path.join(transformer_dir, item)
            if os.path.basename(item_path) == "config.json":
                continue
            if os.path.isdir(item_path):
                shutil.rmtree(item_path)
            else:
                os.remove(item_path)
    else:
        os.makedirs(transformer_dir)

    # Copy new weights
    for file in os.listdir(output_dir):
        src_path = os.path.join(output_dir, file)
        if os.path.isfile(src_path) and file != "config.json":
            shutil.copy(src_path, os.path.join(transformer_dir, file))

    # --- AUTO-UPDATE CONFIG LOGIC ---
    print("Checking for channel mismatch in config.json...")
    config_path = os.path.join(transformer_dir, "config.json")
    
    # Try to find the main model file to inspect
    model_file = None
    for f in os.listdir(transformer_dir):
        if f.endswith(".safetensors") and "diffusion_pytorch_model" in f:
            model_file = os.path.join(transformer_dir, f)
            break
            
    if model_file and os.path.exists(config_path):
        try:
            # Load only the first layer or metadata if possible, but safetensors.torch.load_file loads all
            # Ideally we use safetensors.safe_open but let's just load the specific key if we know it
            # or just load the file (it's fast enough for one file usually)
            
            # CogVideoX Transformer input layer is usually 'patch_embed.proj.weight'
            # Shape: [Out, In, K_h, K_w] or [Out, In, T, H, W] depending on 2D/3D
            
            with torch.no_grad():
                state_dict = load_file(model_file)
                if "patch_embed.proj.weight" in state_dict:
                     weight = state_dict["patch_embed.proj.weight"]
                     # Shape is typically [Out, In, T, H, W] for Conv3d or [Out, In, H, W] for Conv2d
                     # OR for Linear it might be [Out, In_features] where In_features = Channels * PatchVolume
                     
                     detected_in_channels = weight.shape[1]
                     
                     # Check if it's Conv3d or Conv2d
                     if len(weight.shape) == 5: # Conv3d [Out, In, T, H, W]
                         # For Conv3d, dim 1 IS usually in_channels. 
                         # UNLESS it's some other implementation?
                         # Wait, in PyTorch Conv3d: [Out, In, kT, kH, kW]
                         # So weight.shape[1] SHOULD be in_channels.
                         pass
                     elif len(weight.shape) == 2: # Linear [Out, In_features]
                         # Using 3D Patch Embed via Linear Layer?
                         # CogVideoX-Fun uses 3D VAE -> Patch Embed
                         # If it is Linear, In_features = Channels * Patch_T * Patch_H * Patch_W
                         # Assuming Patch Size ~ 2x2x2 = 8
                         print(f"Detected Linear patch_embed with in_features={detected_in_channels}")
                         if detected_in_channels % 8 == 0:
                             detected_in_channels = detected_in_channels // 8
                             print(f"Assuming Patch Volume 8 (2x2x2). Calculated in_channels={detected_in_channels}")
                         elif detected_in_channels % 4 == 0:
                             # Maybe 1x2x2?
                             print(f"Warning: in_features {detected_in_channels} not divisible by 8. Trying /4...")
                             # Let's not guess too wildly.
                             # If we got 288, 288/8 = 36. 288/32 = 9.
                             # If input was 32, we expect 256 (32*8). User got 256 for 32ch model.
                             # If input was 36, we expect 288 (36*8). User got 288 for 36ch model.
                             # perfect match for factor 8.
                             pass
                     
                     # Read Config
                     with open(config_path, 'r') as f:
                         config = json.load(f)
                     
                     current_in_channels = config.get("in_channels", 32)
                     
                     if detected_in_channels != current_in_channels:
                         print(f"Detected weight implies in_channels={detected_in_channels}, but config has {current_in_channels}.")
                         if detected_in_channels == 36:
                             print(f"Updating config.json in_channels to {detected_in_channels} (Mask Supported).")
                             config["in_channels"] = detected_in_channels
                             with open(config_path, 'w') as f:
                                 json.dump(config, f, indent=2)
                         elif detected_in_channels == 32:
                             # Config might correspond to something else? 
                             # If config matches, we do nothing.
                             pass
                         else:
                             print(f"Warning: Unusual channel count {detected_in_channels}. Updating config to match.")
                             config["in_channels"] = detected_in_channels
                             with open(config_path, 'w') as f:
                                 json.dump(config, f, indent=2)
                     else:
                         print(f"Config in_channels ({current_in_channels}) matches weights.")
                else:
                    print("Could not find 'patch_embed.proj.weight' to verify channels.")
        except Exception as e:
            print(f"Error checking/updating config: {e}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", required=True, help="Input checkpoint folder")
    parser.add_argument("--mid_output_dir", default="", help="Intermediate FP32 folder")
    parser.add_argument("--weights_source", default=os.path.expanduser("../../pretrained_models/CogVideoX1.5-5B"), help="Original weights source")
    parser.add_argument("--ckpt_output_dir", default="", help="Final output folder")

    args = parser.parse_args()

    if args.mid_output_dir == "":
        mid_output_dir = args.checkpoint_dir + '-fp32'
    else:
        mid_output_dir = args.mid_output_dir

    if args.ckpt_output_dir == "":
        ckpt_output_dir = args.checkpoint_dir.replace('/checkpoint-', '/ckpt-') + '-sft'
    else:
        ckpt_output_dir = args.ckpt_output_dir
    
    if os.path.exists(ckpt_output_dir):
        print(f"Skipping {ckpt_output_dir}, already exists.")
        return

    run_zero_to_fp32(args.checkpoint_dir, mid_output_dir)
    rename_weights(mid_output_dir)
    prepare_ckpt_structure(mid_output_dir, args.weights_source, ckpt_output_dir)

    if os.path.exists(mid_output_dir):
        shutil.rmtree(mid_output_dir)

if __name__ == "__main__":
    main()
