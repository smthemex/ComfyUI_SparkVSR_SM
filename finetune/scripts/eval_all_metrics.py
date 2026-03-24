import os
import sys
import argparse
import json
import cv2
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
import pyiqa
import imageio.v3 as iio
import glob

# Add script directory to sys.path to find local modules if needed
script_path = os.path.abspath(sys.argv[0])
script_directory = os.path.dirname(script_path)
# Assuming script is in finetune/scripts/, and metrics is in metrics/ (project root level)
metrics_dir = os.path.abspath(os.path.join(script_directory, "../../metrics"))

if not os.path.exists(metrics_dir):
    print(f"Warning: Metrics directory not found at {metrics_dir}")
    # Fallback or just proceed

sys.path.append(script_directory)

# Helper for suppressing warnings
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="timm")

# Try imports for optional metrics
HAS_DOVER = False
try:
    # Attempt to import DOVER from metrics directory
    dover_path = os.path.join(metrics_dir, "DOVER")
    sys.path.append(dover_path)
    
    # Just check if 'dover' package is importable since we use a wrapper now
    import dover
    HAS_DOVER = True
except ImportError as e:
    HAS_DOVER = False
    print(f"Warning: DOVER not found. DOVER metric will be skipped. Error: {e}")
except Exception as e:
    print(f"Warning: Unexpected error during DOVER import setup: {e}")

try:
    # Attempt to import RAFT/Ewarp from metrics directory
    raft_path = os.path.join(metrics_dir, "RAFT")
    sys.path.append(raft_path)
    sys.path.append(os.path.join(raft_path, "core"))
    
    # Check if RAFT code structure exists (e.g. core/raft.py)
    if os.path.exists(os.path.join(raft_path, "core", "raft.py")):
        from raft import RAFT
        from utils import flow_viz
        from utils.utils import InputPadder
        import torch.nn.functional as F

        class Ewarp:
            def __init__(self, args, device):
                self.args = args
                self.device = device
                self.model = torch.nn.DataParallel(RAFT(args))
                self.model.load_state_dict(torch.load(args.model, map_location=self.device))
                self.model = self.model.module
                self.model.to(self.device)
                self.model.eval()

            def process_video(self, video_path):
                import cv2
                cap = cv2.VideoCapture(video_path)
                frames = []
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame)
                cap.release()
                
                if len(frames) < 2:
                    return 0.0
                
                # Stack frames: [T, H, W, 3] -> [T, 3, H, W]
                frames_np = np.stack(frames)
                frames_tensor = torch.from_numpy(frames_np).permute(0, 3, 1, 2).float().to(self.device) / 255.0
                
                errors = []
                for i in range(len(frames_tensor) - 1):
                    img1 = frames_tensor[i] # Target frame (t)
                    img2 = frames_tensor[i+1] # Source frame (t+1) to be warped to t
                    # We compute flow from t to t+1: flow(t->t+1)
                    # Warp(I_{t+1}, flow) to approx I_t
                    
                    # Note: RAFT expects input in [0, 255] range for flow estimation usually?
                    # Let's check calculate_warping_error implementation
                    
                    err = self.calculate_warping_error(img1.unsqueeze(0), img2.unsqueeze(0))
                    errors.append(err)
                
                if not errors: return 0.0
                return np.mean(errors)

            def calculate_warping_error(self, image1, image2):
                # image1, image2: [B, 3, H, W] range [0, 1]
                
                # RAFT expects [0, 255]. So convert back for flow estimation
                image1_255 = image1 * 255.0
                image2_255 = image2 * 255.0
                
                padder = InputPadder(image1.shape)
                image1_pad, image2_pad = padder.pad(image1_255, image2_255)

                with torch.no_grad():
                    _, flow_up = self.model(image1_pad, image2_pad, iters=20, test_mode=True)
                
                # Warping logic
                B, C, H, W = image1_pad.shape
                grid_y, grid_x = torch.meshgrid(torch.arange(0, H), torch.arange(0, W), indexing='ij')
                grid = torch.stack((grid_x, grid_y), 2).float().to(self.device) # [H, W, 2]
                grid = grid.unsqueeze(0).repeat(B, 1, 1, 1) # [B, H, W, 2]
                
                # flow_up is [B, 2, H, W]. permute to [B, H, W, 2]
                vgrid = grid + flow_up.permute(0, 2, 3, 1)
                
                # Normalize grid to [-1, 1]
                vgrid[:, :, :, 0] = 2.0 * vgrid[:, :, :, 0] / max(W - 1, 1) - 1.0
                vgrid[:, :, :, 1] = 2.0 * vgrid[:, :, :, 1] / max(H - 1, 1) - 1.0
                
                # For warping, we want to warp image2 (source) using flow to match image1 (target)
                # We need padded image to extract from
                # image2_pad is [0, 255]. But we want error on [0, 1].
                # So we warp image2_pad / 255.0 ? Or warp image2_pad then divide?
                # Let's use image2_pad directly which is [0, 255], then divide.
                # Actually, image1 passed in is [0, 1].
                # So calculate_warping_error receives [0, 1].
                
                # Pad the ORIGINAL [0, 1] image2 for sampling
                # padder.pad expects tensor.
                image1_norm_pad, image2_norm_pad = padder.pad(image1, image2)
                
                warped_image2 = F.grid_sample(image2_norm_pad, vgrid, align_corners=True)
                
                # Unpad
                warped_image2 = padder.unpad(warped_image2)
                
                # Calculate Error (MSE) on [0, 1] scale
                diff = (image1 - warped_image2) ** 2
                mse = diff.mean()
                return mse.item()

        HAS_EWARP = True
    else:
        HAS_EWARP = False
except ImportError:
    HAS_EWARP = False
    print("Warning: Ewarp/RAFT imports failed. E-Warp metric will be skipped.")
except Exception as e:
    HAS_EWARP = False
    print(f"Warning: Ewarp/RAFT init failed: {e}")

try:
    # Attempt to import VBench from metrics directory
    sys.path.append(os.path.join(metrics_dir, "VBench"))
    from evaluate import calculate_final as Vbench_eval
    HAS_VBENCH = True
except ImportError:
    HAS_VBENCH = False
    print("Warning: VBench not found. VBench metric will be skipped.")

# Fast-VQA import logic
HAS_FASTVQA = False
try:
    if os.path.exists(os.path.join(metrics_dir, "FastVQA")):
         sys.path.append(os.path.join(metrics_dir, "FastVQA"))
         try:
             import fastvqa
             HAS_FASTVQA = True
         except ImportError:
             pass
except ImportError:
    pass

if not HAS_FASTVQA:
    print("Warning: FastVQA not found. FastVQA metric will be skipped.")


# Metrics definitions
FR_METRICS = ['psnr', 'ssim', 'lpips', 'dists']
NR_METRICS = ['clipiqa', 'niqe', 'musiq', 'dover', 'fastvqa', 'vbench']
TEMPORAL_METRICS = ['ewarp']

video_exts = ['.mp4', '.avi', '.mov', '.mkv']

def is_video_file(filename):
    return any(filename.lower().endswith(ext) for ext in video_exts)

to_tensor = transforms.ToTensor()

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
    if not frames:
        return None
    return torch.stack(frames)

def read_image_folder(folder_path):
    image_files = sorted([
        os.path.join(folder_path, f) for f in os.listdir(folder_path)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ])
    if not image_files:
        return None
    frames = [to_tensor(Image.open(p).convert("RGB")) for p in image_files]
    return torch.stack(frames)

def load_sequence(path):
    if os.path.isdir(path):
        return read_image_folder(path)
    elif os.path.isfile(path):
        if is_video_file(path):
            return read_video_frames(path)
        elif path.lower().endswith(('.png', '.jpg', '.jpeg')):
            img = to_tensor(Image.open(path).convert("RGB"))
            return img.unsqueeze(0)
    return None

def match_resolution(gt_frames, pred_frames, item_name=""):
    t_gt_orig = gt_frames.shape[0]
    t_pred_orig = pred_frames.shape[0]
    t = min(t_gt_orig, t_pred_orig)
    gt_frames = gt_frames[:t]
    pred_frames = pred_frames[:t]
    _, _, h_g, w_g = gt_frames.shape
    _, _, h_p, w_p = pred_frames.shape
    
    print(f"[{item_name}] Pre-alignment - GT: {t_gt_orig} frames, {w_g}x{h_g} | Input: {t_pred_orig} frames, {w_p}x{h_p}")
    
    target_h = min(h_g, h_p)
    target_w = min(w_g, w_p)
    
    # Center crop GT
    gt_top = (h_g - target_h) // 2
    gt_left = (w_g - target_w) // 2
    gt_frames = gt_frames[:, :, gt_top:gt_top + target_h, gt_left:gt_left + target_w]
    
    # Center crop Pred
    pred_top = (h_p - target_h) // 2
    pred_left = (w_p - target_w) // 2
    pred_frames = pred_frames[:, :, pred_top:pred_top + target_h, pred_left:pred_left + target_w]
    
    return gt_frames, pred_frames

def img2video(subfolder_path, output_path, fps=25):
    img_tensor = read_image_folder(subfolder_path)
    if img_tensor is None:
        return False
    img_tensor = img_tensor.permute(0, 2, 3, 1)
    frames = (img_tensor * 255).clamp(0, 255).to(torch.uint8).cpu().numpy()
    iio.imwrite(
        output_path,
        frames,
        fps=fps,
        codec='libx264rgb',
        pixelformat='rgb24',
        macro_block_size=None,
        ffmpeg_params=['-crf', '18'],
    )
    return True

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt', type=str, default='', help='Path to GT folder')
    parser.add_argument('--pred', type=str, required=True, help='Path to predicted results folder')
    parser.add_argument('--out', type=str, default='metrics_results', help='Path to save JSON output')
    parser.add_argument('--metrics', type=str, default='psnr,ssim,lpips,dists,clipiqa,niqe,musiq,dover,ewarp,vbench,fastvqa',
                        help='Comma-separated list of metrics')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--filename', type=str, default='all_metrics_results.json', help='Filename for the output JSON')
    args = parser.parse_args()

    metrics_list = [m.strip().lower() for m in args.metrics.split(',')]
    device = torch.device(args.device)
    
    # Initialize basic PyIQA metrics
    iqa_models = {}
    for m in metrics_list:
        if m in FR_METRICS or m in ['clipiqa', 'niqe', 'musiq']:
            try:
                iqa_models[m] = pyiqa.create_metric(m).to(device).eval()
            except Exception as e:
                print(f"Failed to initialize basic metric {m}: {e}")

    pred_root = args.pred
    gt_root = args.gt
    has_gt = bool(gt_root and os.path.exists(gt_root))
    
    if not os.path.exists(pred_root):
        print(f"Prediction path {pred_root} does not exist.")
        return

    items = sorted(os.listdir(pred_root))
    items = [x for x in items if os.path.isdir(os.path.join(pred_root, x)) or is_video_file(x)]
    
    results = {}
    
    print("Computing Per-Sample Metrics...")
    for item_name in tqdm(items):
        item_path = os.path.join(pred_root, item_name)
        key_name = os.path.splitext(item_name)[0]
        
        pred_seq = load_sequence(item_path)
        if pred_seq is None:
            continue
            
        res = {}
        
        gt_seq = None
        if has_gt:
            possible_gt_paths = [
                os.path.join(gt_root, item_name),
                os.path.join(gt_root, key_name),
                os.path.join(gt_root, key_name + '.mp4'),
                os.path.join(gt_root, key_name + '.mkv')
            ]
            for p in possible_gt_paths:
                if os.path.exists(p):
                    gt_seq = load_sequence(p)
                    if gt_seq is not None:
                        break
        
        if gt_seq is not None and pred_seq is not None:
            gt_seq, pred_seq_aligned = match_resolution(gt_seq.clone(), pred_seq.clone(), item_name)
        else:
            pred_seq_aligned = pred_seq
            
        pred_batch = pred_seq_aligned.to(device)
        gt_batch = gt_seq.to(device) if gt_seq is not None else None
        
        for m_name, model in iqa_models.items():
            try:
                if m_name in FR_METRICS:
                    if gt_batch is not None:
                        # Print resolution and frame number once per sequence
                        if m_name == FR_METRICS[0]:
                            _, c_gt, h_gt, w_gt = gt_batch.shape
                            t_gt = gt_batch.shape[0]
                            _, c_pred, h_pred, w_pred = pred_batch.shape
                            t_pred = pred_batch.shape[0]
                            print(f"{key_name} - GT => frames: {t_gt}, resolution: {w_gt}x{h_gt}")
                            print(f"{key_name} - Input => frames: {t_pred}, resolution: {w_pred}x{h_pred}")
                            
                        # Compute per-frame to avoid CUDA 32-bit index overflow
                        # on high-resolution videos
                        frame_vals = []
                        for fi in range(pred_batch.shape[0]):
                            fv = model(pred_batch[fi:fi+1], gt_batch[fi:fi+1]).item()
                            frame_vals.append(fv)
                        val = float(np.mean(frame_vals))
                    else:
                        val = None
                else:
                    # NR metrics: also compute per-frame for safety
                    frame_vals = []
                    for fi in range(pred_batch.shape[0]):
                        fv = model(pred_batch[fi:fi+1]).item()
                        frame_vals.append(fv)
                    val = float(np.mean(frame_vals))
                
                if val is not None:
                    res[m_name] = round(val, 4)
            except Exception as e:
                print(f"Error computing {m_name} for {key_name}: {e}")
        
        if 'fastvqa' in metrics_list and HAS_FASTVQA:
            # Placeholder for actual FastVQA call
            pass 

        results[key_name] = res

    need_dover = 'dover' in metrics_list and HAS_DOVER
    need_ewarp = 'ewarp' in metrics_list and HAS_EWARP
    need_vbench = 'vbench' in metrics_list and HAS_VBENCH
    need_fastvqa = 'fastvqa' in metrics_list and HAS_FASTVQA
    
    if need_dover or need_ewarp or need_vbench or need_fastvqa:
        print("Preparing temporary input folder for batch metrics...")
        temp_input_dir = os.path.join(args.out, "temp_for_eval")
        os.makedirs(temp_input_dir, exist_ok=True)
        
        valid_temp_files = []
        for item_name in items:
            key_name = os.path.splitext(item_name)[0]
            src_path = os.path.join(pred_root, item_name)
            dst_path = os.path.join(temp_input_dir, key_name + ".mp4")
            
            if os.path.exists(dst_path):
                 valid_temp_files.append(dst_path)
                 continue

            if os.path.isdir(src_path):
                if img2video(src_path, dst_path):
                    valid_temp_files.append(dst_path)
            elif is_video_file(src_path):
                if not os.path.exists(dst_path):
                    os.symlink(os.path.abspath(src_path), dst_path)
                valid_temp_files.append(dst_path)
        
        if need_dover:
            try:
                print("Running DOVER...")
                # Define DOVER wrapper internally since the repo lacks a direct API
                import yaml
                from dover.models import DOVER
                from dover.datasets import ViewDecompositionDataset
                
                class DOVERWrapper:
                    def __init__(self, metrics_dir, device, mode="standard"):
                        self.device = device
                        self.dover_dir = os.path.join(metrics_dir, "DOVER")
                        self.mode = mode
                        
                        if mode == "mobile":
                            yml_path = os.path.join(self.dover_dir, "dover-mobile.yml")
                            print("Using DOVER-Mobile model.")
                        else:
                            yml_path = os.path.join(self.dover_dir, "dover.yml")
                            print("Using DOVER model.")

                        # Load config
                        with open(yml_path, "r") as f:
                            self.opt = yaml.safe_load(f)
                        
                        # Load model
                        self.model = DOVER(**self.opt["model"]["args"]).to(self.device)
                        
                        if mode == "mobile":
                             pretrained_path = os.path.join(self.dover_dir, "pretrained_weights/DOVER-Mobile.pth")
                        else:
                             pretrained_path = os.path.join(self.dover_dir, "pretrained_weights/DOVER.pth")
                        
                        try:
                            self.model.load_state_dict(torch.load(pretrained_path, map_location=self.device))
                        except Exception as e:
                            print(f"Failed to load DOVER weights from {pretrained_path}: {e}")
                            raise e
                        
                        self.model.eval()

                    def evaluate(self, input_dir):
                        # Configure dataset options
                        dopt = self.opt["data"]["val-l1080p"]["args"]
                        dopt["anno_file"] = None
                        dopt["data_prefix"] = input_dir
                        
                        # Fix sampling params for short videos (e.g. 32 frames).
                        # Default technical config (num_clips=3, frame_interval=2)
                        # generates 96 indices spanning 0~82; after mod(N_frames),
                        # this causes severe frame repetition and breaks temporal
                        # quality assessment. Reduce to num_clips=1, frame_interval=1
                        # so that all frames are used without excessive wrapping.
                        dopt["sample_types"]["technical"]["num_clips"] = 1
                        dopt["sample_types"]["technical"]["frame_interval"] = 1
                        
                        dataset = ViewDecompositionDataset(dopt)
                        dataloader = torch.utils.data.DataLoader(
                            dataset, batch_size=1, num_workers=4, pin_memory=True
                        )
                        
                        results = {}
                        sample_types = ["aesthetic", "technical"]

                        for i, data in enumerate(tqdm(dataloader, desc="DOVER Inference")):
                            if len(data.keys()) == 1: continue # processing failed
                            
                            video = {}
                            for key in sample_types:
                                if key in data:
                                    video[key] = data[key].to(self.device)
                                    b, c, t, h, w = video[key].shape
                                    video[key] = (
                                        video[key]
                                        .reshape(b, c, data["num_clips"][key], t // data["num_clips"][key], h, w)
                                        .permute(0, 2, 1, 3, 4, 5)
                                        .reshape(b * data["num_clips"][key], c, t // data["num_clips"][key], h, w)
                                    )
                            
                            with torch.no_grad():
                                res = self.model(video, reduce_scores=False)
                                res = [np.mean(l.cpu().numpy()) for l in res]
                            
                            # Fuse results (from dover's evaluate_a_set_of_videos.py)
                            t, a = (res[1] - 0.1107) / 0.07355, (res[0] + 0.08285) / 0.03774
                            x = t * 0.6104 + a * 0.3896
                            final_scores = {
                                "aesthetic": 1 / (1 + np.exp(-a)),
                                "technical": 1 / (1 + np.exp(-t)),
                                "overall": 1 / (1 + np.exp(-x)),
                            }
                            
                            name = data["name"][0] # e.g., "video_name" (no ext if ViewDecompositionDataset handles it)
                            # DOVER dataset might return relative path or just name
                            key = os.path.splitext(os.path.basename(name))[0]
                            results[key] = final_scores
                            
                        return results

                # User requested standard DOVER
                wrapper = DOVERWrapper(metrics_dir, device, mode="standard")
                dover_scores_dict = wrapper.evaluate(temp_input_dir)
                
                for k, scores in dover_scores_dict.items():
                    if k in results:
                        results[k]['dover_overall'] = scores['overall']
                        results[k]['dover_aesthetic'] = scores['aesthetic']
                        results[k]['dover_technical'] = scores['technical']
                        results[k]['dover'] = scores['overall'] # Default for summary
                
            except Exception as e:
                print(f"Error running DOVER: {e}")
                import traceback
                traceback.print_exc()

        if need_ewarp:
            try:
                print("Running E-Warp...")
                class EwarpArgs:
                    def __init__(self):
                        self.dropout = 0
                        self.alternate_corr = False
                    def __contains__(self, key):
                        return hasattr(self, key)
                ewarp_args = EwarpArgs()
                ewarp_args.small = False
                ewarp_args.mixed_precision = False
                ewarp_args.alternate_corr = False
                ewarp_args.dropout = 0
                ewarp_args.model = os.path.join(metrics_dir, "RAFT/models/raft-things.pth")

                ewarp_evaluator = Ewarp(ewarp_args, device)
                
                input_videos = sorted(glob.glob(os.path.join(temp_input_dir, '*')))
                for vid_path in input_videos:
                     score = ewarp_evaluator.process_video(vid_path)
                     # Score is MSE on [0, 1] scale (e.g. 0.0017)
                     # Paper often reports x1e3? Or x1e2?
                     # User said 1.77. 0.0017 * 1000 = 1.7.
                     # Let's store scaled score matching user expectation?
                     # Or store raw and let user know.
                     # Let's store scaled x1000 to be helpful if it matches the magnitude.
                     # But strictly, E_warp unit is arbitrary unless specified.
                     # Let's save raw score but multiply by 1000 in the final JSON key 'ewarp_x1000' as well?
                     
                     name = os.path.basename(vid_path)
                     key = os.path.splitext(name)[0]
                     
                     if key not in results: results[key] = {}
                     results[key]['ewarp'] = score
                     results[key]['ewarp_scaled_1000'] = score * 1000.0
                     
            except Exception as e:
                print(f"Error running E-Warp: {e}")
                import traceback
                traceback.print_exc()

        if need_fastvqa:
            try:
                print("Running Fast-VQA...")
                from fastvqa.datasets import get_spatial_fragments, SampleFrames, FragmentSampleFrames
                from fastvqa.models import DiViDeAddEvaluator
                import decord
                import yaml
                
                class FastVQAWrapper:
                    def __init__(self, metrics_dir, device):
                        self.device = device
                        self.fastvqa_dir = os.path.join(metrics_dir, "FastVQA")
                        # Default to FasterVQA (3D) config as matched by downloaded weights
                        self.opt_path = os.path.join(self.fastvqa_dir, "options/fast/f3dvqa-b.yml")
                        
                        with open(self.opt_path, "r") as f:
                            self.opt = yaml.safe_load(f)
                        
                        self.model = DiViDeAddEvaluator(**self.opt["model"]["args"]).to(self.device)
                        
                        pretrained_path = os.path.join(self.fastvqa_dir, "pretrained_weights/FAST_VQA_3D_1_1.pth")
                        if not os.path.exists(pretrained_path):
                             print(f"FastVQA weights not found at {pretrained_path}. Please download them.")
                             return

                        # Load weights
                        checkpoint = torch.load(pretrained_path, map_location=self.device)
                        if "state_dict" in checkpoint:
                            self.model.load_state_dict(checkpoint["state_dict"])
                        else:
                            self.model.load_state_dict(checkpoint)
                        self.model.eval()
                        
                        self.mean_score = 0.14759505
                        self.std_score = 0.03613452

                    def sigmoid_rescale(self, score):
                        x = (score - self.mean_score) / self.std_score
                        return 1 / (1 + np.exp(-x))

                    def process_video(self, video_path):
                        if not os.path.exists(video_path): return 0.0
                        
                        # Use config from options
                        t_data_opt = self.opt["data"]["val-kv1k"]["args"]
                        s_data_opt = self.opt["data"]["val-kv1k"]["args"]["sample_types"]
                        
                        video_reader = decord.VideoReader(video_path)
                        vsamples = {}
                        
                        for sample_type, sample_args in s_data_opt.items():
                            # Sample Temporally
                            if t_data_opt.get("t_frag",1) > 1:
                                sampler = FragmentSampleFrames(fsize_t=sample_args["clip_len"] // sample_args.get("t_frag",1),
                                                               fragments_t=sample_args.get("t_frag",1),
                                                               num_clips=sample_args.get("num_clips",1),
                                                              )
                            else:
                                sampler = SampleFrames(clip_len = sample_args["clip_len"], num_clips = sample_args["num_clips"])
                            
                            num_clips = sample_args.get("num_clips",1)
                            frames = sampler(len(video_reader))
                            frame_dict = {idx: video_reader[idx] for idx in np.unique(frames)}
                            imgs = [frame_dict[idx] for idx in frames]
                            video = torch.stack(imgs, 0)
                            video = video.permute(3, 0, 1, 2)

                            # Sample Spatially
                            sampled_video = get_spatial_fragments(video, **sample_args)
                            mean, std = torch.FloatTensor([123.675, 116.28, 103.53]), torch.FloatTensor([58.395, 57.12, 57.375])
                            sampled_video = ((sampled_video.permute(1, 2, 3, 0) - mean) / std).permute(3, 0, 1, 2)
                            
                            sampled_video = sampled_video.reshape(sampled_video.shape[0], num_clips, -1, *sampled_video.shape[2:]).transpose(0,1)
                            vsamples[sample_type] = sampled_video.to(self.device)
                            
                        with torch.no_grad():
                            result = self.model(vsamples)
                            score = self.sigmoid_rescale(result.mean().item())
                            return score

                fastvqa_evaluator = FastVQAWrapper(metrics_dir, device)
                
                print("Computing Fast-VQA scores...")
                # We can reuse input_videos list if available, or list temp_input_dir again
                input_videos = sorted(glob.glob(os.path.join(temp_input_dir, '*')))
                for vid_path in input_videos:
                     score = fastvqa_evaluator.process_video(vid_path)
                     
                     name = os.path.basename(vid_path)
                     key = os.path.splitext(name)[0]
                     
                     if key not in results: results[key] = {}
                     results[key]['fastvqa'] = score

            except Exception as e:
                print(f"Error running Fast-VQA: {e}")
                import traceback
                traceback.print_exc()

        if need_vbench:
            try:
                print("Running VBench...")
                v_results, v_avg, v_dim_results, v_dim_avg = Vbench_eval(temp_input_dir)
                
                for k, score in v_results.items():
                     if k not in results: results[k] = {}
                     results[k]['vbench_score'] = score
                
                for k, dim_res in v_dim_results.items():
                    if k in results:
                         results[k].update(dim_res)
            except Exception as e:
                print(f"Error running VBench: {e}")

    print("Aggregating results...")
    final_output = {
        "per_sample": results,
        "average": {}
    }
    
    all_keys = list(results.keys())
    if all_keys:
        # Collect all metric keys present in the first result (or union of all)
        # Using union is safer
        all_metrics = set()
        for k in all_keys:
            all_metrics.update(results[k].keys())
            
        for m in all_metrics:
            vals = [results[k][m] for k in all_keys if m in results[k] and results[k][m] is not None]
            if vals:
                final_output["average"][m] = float(np.mean(vals))
    
    def convert_to_serializable(obj):
        if isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(element) for element in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return convert_to_serializable(obj.tolist())
        return obj

    final_output = convert_to_serializable(final_output)

    os.makedirs(args.out, exist_ok=True)
    out_file = os.path.join(args.out, args.filename)
    # Ensure filename ends with .json if not provided
    if not out_file.endswith('.json'):
        out_file += '.json'
        
    with open(out_file, 'w') as f:
        json.dump(final_output, f, indent=4)
        
    print(f"Evaluation complete. Results saved to {out_file}")
    print("Average Metrics:")
    print(json.dumps(final_output["average"], indent=2))

if __name__ == "__main__":
    main()
