import os
import cv2
import shutil
import requests
#import fal_client
from PIL import Image, UnidentifiedImageError
from io import BytesIO
from tqdm import tqdm
import torch
from torchvision import transforms

# ================= 配置区域 =================
os.environ['FAL_KEY'] = 'your_fal_key'

# 模型ID
FAL_MODEL_ID = "fal-ai/nano-banana-pro/edit"

# 提示词
TASK_PROMPT = "Super-Resolution and Restoration Task: Upscale this low-resolution image to high definition. The primary goal is to restore sharpness and clarity by effectively removing all types of degradation, including blur, heavy digital noise, grain, and JPEG compression artifacts."


# ==========================================

to_tensor = transforms.ToTensor()

def is_valid_image(path):
    """检查图片是否完整有效"""
    try:
        with Image.open(path) as img:
            img.verify()  # 验证文件完整性
        return True
    except (IOError, SyntaxError, UnidentifiedImageError):
        return False

def save_ref_frames_locally(video_path=None, output_dir=None, video_id=None, target_frames=None, is_match=False, specific_indices=None):
    """
    Extract frames from video and save them locally without API call.
    Used for caching GT frames.
    
    Args:
        video_path (str): Path to input video file.
        output_dir (str): Directory to save frames.
        video_id (str): Optional identifier for the video.
        target_frames (int): Target number of frames (e.g. 17). If provided, indices are based on this.
        is_match (bool): If True, apply (F-1)%8 padding logic (same as trainer's raw_test padding).
        specific_indices (List[int]): Optional list of specific indices to extract. Overrides target_frames/_select_indices logic.
        
    Returns:
        List[(int, str)]: List of (index, saved_path).
    """
    if output_dir is None:
        raise ValueError("output_dir must be provided")

    os.makedirs(output_dir, exist_ok=True)
    frame_list = []
    
    # Prefix for filenames
    prefix = f"{video_id}_" if video_id else ""
    
    if video_path is None:
        return []

    print(f"Opening video for GT extraction: {video_path}")
    
    # Try to find GT video if input is LQ
    # Assuming standard structure: .../LQ-Video/... -> .../GT-Video/...
    if "LQ-Video" in video_path:
        gt_path = video_path.replace("LQ-Video", "GT-Video")
        if os.path.exists(gt_path):
            print(f"Switched to GT video source: {gt_path}")
            video_path = gt_path
    
    # Try using Decord which is more robust
    try:
        from decord import VideoReader, cpu
        vr = VideoReader(video_path, ctx=cpu(0))
        total_frames = len(vr)
        use_decord = True
        print(f"Using Decord for frame extraction. Total frames: {total_frames}")
    except ImportError:
        use_decord = False
        print("Decord not found, falling back to OpenCV.")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
             print(f"Error: Could not open video {video_path}")
             return []
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    effective_frames = total_frames
    if is_match:
        remainder = (total_frames - 1) % 8
        if remainder != 0:
            effective_frames = total_frames + (8 - remainder)
    
    if specific_indices is not None:
        indices = specific_indices
    else:
        num_frames_to_use = target_frames if target_frames is not None else effective_frames
        indices = _select_indices(num_frames_to_use)
    
    for idx in indices:
        # If idx is beyond raw video length, we use the last frame (padding behavior)
        read_idx = min(idx, total_frames - 1)
        
        if use_decord:
             frame_obj = vr[read_idx]
             if hasattr(frame_obj, "asnumpy"):
                 frame_np = frame_obj.asnumpy()
             else:
                 # Assume torch tensor or similar
                 frame_np = frame_obj.cpu().numpy()
                 
             # Decord returns RGB
             frame = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR) # Convert to BGR for cv2.imwrite
             ret = True
        else:
             cap.set(cv2.CAP_PROP_POS_FRAMES, read_idx)
             ret, frame = cap.read()
             
        if ret:
            # Use prefix in filename
            frame_filename = os.path.join(output_dir, f"{prefix}frame_{idx:05d}.png")
            cv2.imwrite(frame_filename, frame)
            frame_list.append((idx, frame_filename))
            
    if not use_decord:
        cap.release()
    return frame_list

def get_ref_frames_api(video_path=None, output_dir=None, video_tensor=None, video_id=None, target_frames=None, is_match=False, specific_indices=None, prompt=None, ref_prompt_mode='fixed', resolution='1K'):
    """
    Unified function to get reference frames via API.
    Can accept video_path OR video_tensor.
    Now processes frames with **FACE REFERENCE STRATEGY (First Frame)**:
    1. Process Frame 0 (Independent SR -> HR Initial).
    2. Detect & Crop Face from Frame 0 HR.
    3. If Face Found: Use as Static Reference for Frames 1-N.
    4. If No Face: Use Independent SR (No Reference) for Frames 1-N.
    
    Args:
        video_path (str): Path to input video file.
        output_dir (str): Directory to save intermediate files.
        video_tensor (torch.Tensor): [F, C, H, W] in 0-1 range (optional, if video_path not provided).
        video_id (str): Optional identifier for the video.
        target_frames (int): Target number of frames (e.g. 17). If provided, indices are based on this.
        is_match (bool): If True, apply (F-1)%8 padding logic (same as trainer's raw_test padding).
        specific_indices (List[int]): Optional list of specific indices to use. Overrides target_frames.
        prompt (str): Optional prompt for the API call. Defaults to PROMPT_TEXT.
        ref_prompt_mode (str): 'fixed' or 'dynamic'. 'fixed' uses static prompt for all frames. 'dynamic' uses VLM analysis.
        
    Returns:
        List[(int, torch.Tensor)]: List of (index, tensor).
    """
    import fal_client
    import numpy as np
    import cv2
    from PIL import Image

    if output_dir is None:
        raise ValueError("output_dir must be provided")

    TEMP_INPUT_DIR = os.path.join(output_dir, "temp_frames_input")
    TEMP_OUTPUT_DIR = os.path.join(output_dir, "temp_frames_output")
    TEMP_AUG_DIR = os.path.join(output_dir, "temp_frames_aug") # For Face Crop
    
    # 1. 目录准备
    if os.path.exists(TEMP_INPUT_DIR): 
        shutil.rmtree(TEMP_INPUT_DIR)
    if os.path.exists(TEMP_AUG_DIR): 
        shutil.rmtree(TEMP_AUG_DIR)
    os.makedirs(TEMP_INPUT_DIR)
    os.makedirs(TEMP_OUTPUT_DIR, exist_ok=True)
    os.makedirs(TEMP_AUG_DIR, exist_ok=True)

    frame_list = []
    
    # Prefix for filenames
    prefix = f"{video_id}_" if video_id else ""
    
    # 2. Extract Frames
    if video_path is not None:
        print(f"Opening video: {video_path}")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Error: Could not open video {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        effective_frames = total_frames
        if is_match:
            remainder = (total_frames - 1) % 8
            if remainder != 0:
                effective_frames = total_frames + (8 - remainder)
        
        if specific_indices is not None:
            indices = specific_indices
        else:
            num_frames_to_use = target_frames if target_frames is not None else effective_frames
            indices = _select_indices(num_frames_to_use)
        
        for idx in indices:
            # If idx is beyond raw video length, we use the last frame (padding behavior)
            read_idx = min(idx, total_frames - 1)
            cap.set(cv2.CAP_PROP_POS_FRAMES, read_idx)
            ret, frame = cap.read()
            if ret:
                # Use prefix in filename
                frame_filename = os.path.join(TEMP_INPUT_DIR, f"{prefix}frame_{idx:05d}.png")
                cv2.imwrite(frame_filename, frame)
                frame_list.append((idx, frame_filename))
        cap.release()

    elif video_tensor is not None:
        # Preprocess tensor to list of frames
        if video_tensor.ndim == 5: 
            video_tensor = video_tensor[0].permute(1, 0, 2, 3)
        elif video_tensor.ndim == 4 and video_tensor.shape[0] < video_tensor.shape[1]: 
            video_tensor = video_tensor.permute(1, 0, 2, 3)
            
        total_frames = video_tensor.shape[1] if video_tensor.ndim == 5 else video_tensor.shape[0]
        
        effective_frames = total_frames
        if is_match:
            remainder = (total_frames - 1) % 8
            if remainder != 0:
                effective_frames = total_frames + (8 - remainder)

        if specific_indices is not None:
            indices = specific_indices
        else:
            num_frames_to_use = target_frames if target_frames is not None else effective_frames
            indices = _select_indices(num_frames_to_use)
        
        for idx in indices:
            read_idx = min(idx, total_frames - 1)
            frame = video_tensor[read_idx] if video_tensor.ndim == 4 else video_tensor[0, :, read_idx]
            # 0-1 float to 0-255 uint8
            if frame.dtype == torch.float32 or frame.dtype == torch.float16 or frame.dtype == torch.bfloat16:
                if frame.max() > 1.0:
                    # Assume [0, 255]
                    frame = frame.float().clamp(0, 255).to(torch.uint8)
                elif frame.min() < 0.0:
                    # Assume [-1, 1]
                    frame = ((frame.float() * 0.5 + 0.5).clamp(0, 1) * 255).to(torch.uint8)
                else:
                    # Assume [0, 1]
                    frame = (frame.float().clamp(0, 1) * 255).to(torch.uint8)
            
            frame_np = frame.permute(1, 2, 0).cpu().numpy() 
            img = Image.fromarray(frame_np)
            frame_filename = os.path.join(TEMP_INPUT_DIR, f"{prefix}frame_{idx:05d}.png")
            img.save(frame_filename)
            frame_list.append((idx, frame_filename))

    processed_ref_frames = []
    
    # Sort frames by index
    frame_list.sort(key=lambda x: x[0])
    

    # -------------------------------------------------------------
    # 0. Helper: Caption Generation & Degradation Analysis
    # -------------------------------------------------------------
    def _generate_caption(image_path):
        """Generate caption using local Qwen-VL service."""
        try:
            from openai import OpenAI
            import base64
            
            client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")
            
            def encode_image(path):
                with open(path, "rb") as image_file:
                    return base64.b64encode(image_file.read()).decode('utf-8')
            
            base64_image = encode_image(image_path)
            
            completion = client.chat.completions.create(
                model="Qwen/Qwen2-VL-7B-Instruct",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Describe this image in extreme detail, focusing on textures, lighting, facial features, and background elements. Keep it concise but comprehensive."},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                        ],
                    }
                ],
                max_tokens=300,
            )
            return completion.choices[0].message.content
        except Exception as e:
            print(f"[Warning] Caption generation failed: {e}")
            return ""

    def _analyze_degradation(image_path):
        """Analyze degradation using local Qwen-VL service."""
        try:
            from openai import OpenAI
            import base64
            
            client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")
            
            def encode_image(path):
                with open(path, "rb") as image_file:
                    return base64.b64encode(image_file.read()).decode('utf-8')
            
            base64_image = encode_image(image_path)
            
            completion = client.chat.completions.create(
                model="Qwen/Qwen2-VL-7B-Instruct",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Analyze the image quality issues (blur, noise, compression artifacts, low resolution, etc.) and write a concise, effective prompt for an image restoration model to fix these specific issues. Return ONLY the prompt, starting with 'Restore image, ...'"},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                        ],
                    }
                ],
                max_tokens=200,
            )
            return completion.choices[0].message.content
        except Exception as e:
            print(f"[Warning] Degradation analysis failed: {e}")
            return TASK_PROMPT # Fallback

    # -------------------------------------------------------------
    # 0. Helper: Caption Generation & Degradation Analysis
    # -------------------------------------------------------------
    
    print(f"[{ref_prompt_mode.upper()}] Mode Selected.")
    if ref_prompt_mode == 'fixed':
        print(f"FAL_MODEL_ID: {FAL_MODEL_ID}")
        print(f"Prompt Config: Fixed. Using base TASK_PROMPT.")
    else:
        print(f"FAL_MODEL_ID: {FAL_MODEL_ID}")
        print(f"Prompt Config: Dynamic. Using VLM analysis.")
    
    # -------------------------------------------------------------
    # Main Loop
    # -------------------------------------------------------------
    
    print(f"Starting Qwen-Prompt SR for {len(frame_list)} frames...")
    
    content_description = ""
    
    for i, (idx, local_path) in enumerate(tqdm(frame_list, desc="SR Processing")):
        base_name = os.path.basename(local_path)
        save_path = os.path.join(TEMP_OUTPUT_DIR, base_name)
        
        # Check Cache
        if os.path.exists(save_path) and is_valid_image(save_path):
             try:
                img = Image.open(save_path).convert("RGB")
                t_img = to_tensor(img) * 2.0 - 1.0
                processed_ref_frames.append((idx, t_img))
                # Try to load existing content description if we are skipping frame 0
                if i == 0:
                    caption_path = os.path.join(output_dir, "caption.txt")
                    if os.path.exists(caption_path):
                        with open(caption_path, "r") as f:
                            content_description = f.read()
                continue
             except Exception:
                pass
        
        # Logic
        api_prompt = ""
        
        if ref_prompt_mode == 'fixed':
            # Mode 1: Fixed SR Prompt
            api_prompt = TASK_PROMPT
            if prompt:
                 api_prompt += f" Target Scene Description: {prompt}"
            image_urls = [fal_client.upload_file(local_path)]
            
        else:
            # Mode 2: Dynamic (VLM Analysis)
            # 1. Analyze Degradation for CURRENT frame
            print(f"Analyzing degradation for Frame {idx}...")
            current_task_prompt = _analyze_degradation(local_path)
            
            # Save Task Prompt
            task_prompt_path = os.path.join(output_dir, f"task_prompt_frame_{idx:05d}.txt")
            with open(task_prompt_path, "w") as f:
                f.write(current_task_prompt)

            api_prompt = current_task_prompt
            if prompt:
                 api_prompt += f" Target Scene Description: {prompt}"
            
            image_urls = []
            
            # 1. Upload Target (Current LQ)
            target_url = fal_client.upload_file(local_path)
            
            if i == 0:
                 # Frame 0: Independent SR
                 image_urls = [target_url]
            else:
                 # Frame 1-N:
                 # Enhance Prompt with Content Description ONLY IF DYNAMIC MODE
                 # In fixed mode, we stay fixed.
                 if ref_prompt_mode == 'dynamic' and content_description:
                      api_prompt += f" The image content is: {content_description}"
                      
                 image_urls = [target_url]

        # Call API
        try:
            result = fal_client.run(
                FAL_MODEL_ID,
                arguments={
                    "prompt": api_prompt,
                    "image_urls": image_urls,
                    "num_images": 1, 
                    "aspect_ratio": "auto",
                    "output_format": "png",
                    "resolution": resolution
                },
            )
            output_image_url = result['images'][0]['url']
            
            # Download
            response = requests.get(output_image_url)
            response.raise_for_status()
            img = Image.open(BytesIO(response.content))
            img.save(save_path)
            
            # Save Input
            input_save_path = os.path.join(TEMP_OUTPUT_DIR, f"input_{base_name}")
            shutil.copy(local_path, input_save_path)
            
            t_img = to_tensor(img.convert("RGB")) * 2.0 - 1.0
            processed_ref_frames.append((idx, t_img))
            
            # After Frame 0:
            if i == 0:
                 # 1. Generate Caption (ONLY IN DYNAMIC MODE)
                 if ref_prompt_mode == 'dynamic':
                     print("Generating Content Description for Frame 0...")
                     content_description = _generate_caption(save_path)
                     if content_description:
                         print(f"Content Description: {content_description}")
                         
                         # SAVE CAPTION TO FILE
                         caption_path = os.path.join(output_dir, "caption.txt")
                         with open(caption_path, "w") as f:
                             f.write(content_description)
                 else:
                     print("Skipping Content Description Generation (Fixed Mode).")
                         
            elif i == 1:
                 # Print Full Prompt once for Frame 1 (first frame using caption)
                 print(f"Full Prompt for Frame {idx}: {api_prompt}")
                 pass1_prompt_path = os.path.join(output_dir, f"prompt_frame{idx:05d}.txt")
                 with open(pass1_prompt_path, "w") as f:
                     f.write(api_prompt)
            
        except Exception as e:
            print(f"[Error] API Failed for frame {idx}: {e}")
            continue

    return processed_ref_frames

def _select_indices(total_frames):
    """Select 3 frames: first, middle, last."""
    if total_frames <= 0:
        return []
    if total_frames == 1:
        return [0]
    if total_frames == 2:
        return [0, 1]
    return [0, total_frames // 2, total_frames - 1]
