# !/usr/bin/env python
# -*- coding: UTF-8 -*-
import os
import torch
import gc
from PIL import Image
import numpy as np
import comfy.model_management as mm
from comfy.utils import common_upscale

cur_path = os.path.dirname(os.path.abspath(__file__))

def clear_comfyui_cache():
    cf_models=mm.loaded_models()
    try:
        for pipe in cf_models:
            pipe.unpatch_model(device_to=torch.device("cpu"))
    except: pass
    mm.soft_empty_cache()
    torch.cuda.empty_cache()
    max_gpu_memory = torch.cuda.max_memory_allocated()
    print(f"After Max GPU memory allocated: {max_gpu_memory / 1000 ** 3:.2f} GB")

def gc_cleanup():
    gc.collect()
    torch.cuda.empty_cache()


def phi2narry(img):
    img = torch.from_numpy(np.array(img).astype(np.float32) / 255.0).unsqueeze(0)
    return img

def tensor2image(tensor):
    tensor = tensor.cpu()
    image_np = tensor.squeeze().mul(255).clamp(0, 255).byte().numpy()
    image = Image.fromarray(image_np, mode='RGB')
    return image

def tensor2pillist(tensor_in):
    d1, _, _, _ = tensor_in.size()
    if d1 == 1:
        img_list = [tensor2image(tensor_in)]
    else:
        tensor_list = torch.chunk(tensor_in, chunks=d1)
        img_list=[tensor2image(i) for i in tensor_list]
    return img_list

def tensor2pillist_upscale(tensor_in,width,height):
    d1, _, _, _ = tensor_in.size()
    if d1 == 1:
        img_list = [nomarl_upscale(tensor_in,width,height)]
    else:
        tensor_list = torch.chunk(tensor_in, chunks=d1)
        img_list=[nomarl_upscale(i,width,height) for i in tensor_list]
    return img_list

def tensor2list(tensor_in,width,height):
    if tensor_in is None:
        return None
    d1, _, _, _ = tensor_in.size()
    if d1 == 1:
        tensor_list = [tensor_upscale(tensor_in,width,height)]
    else:
        tensor_list_ = torch.chunk(tensor_in, chunks=d1)
        tensor_list=[tensor_upscale(i,width,height) for i in tensor_list_]
    return tensor_list


def tensor_upscale(tensor, width, height):
    samples = tensor.movedim(-1, 1)
    samples = common_upscale(samples, width, height, "nearest-exact", "center")
    samples = samples.movedim(1, -1)
    return samples

def nomarl_upscale(img, width, height):
    samples = img.movedim(-1, 1)
    img = common_upscale(samples, width, height, "nearest-exact", "center")
    samples = img.movedim(1, -1)
    img = tensor2image(samples)
    return img

def map_neg1_1_to_0_1(t):
    """
    接受 torch.Tensor 或可转 torch.Tensor 的输入。
    可处理形状: H,W,C 或 B,H,W,C 或 B,T,H,W,C（会按元素处理）。
    返回 float tensor，范围 0..1。
    """
    if not torch.is_tensor(t):
        t = torch.tensor(t)
    t = t.float()
    # map -1..1 -> 0..1
    t = (t + 1.0) * 0.5
    # 限幅到 [0,1]
    t = t.clamp(0.0, 1.0)
    # 保持在 cpu 端，调用方可决定是否转 device/dtype
    return t.cpu()

def map_0_1_to_neg1_1(t):
    """
    接受 torch.Tensor 或可转 torch.Tensor 的输入。
    可处理形状: H,W,C 或 B,H,W,C 或 B,T,H,W,C（会按元素处理）。
    确保 float，0..255 -> 0..1，再把 0..1 -> -1..1（如果已经在 -1..1 则不变）。
    """
    if not torch.is_tensor(t):
        t = torch.tensor(t)
    t = t.float()
    # 处理 0..255 的情况
    try:
        vmax = float(t.max())
    except Exception:
        vmax = 1.0
    if vmax > 2.0:
        t = t / 255.0
    # 若当前处于 0..1 范围，则映射到 -1..1
    try:
        vmin = float(t.min())
        vmax = float(t.max())
    except Exception:
        vmin, vmax = -1.0, 1.0
    if vmin >= 0.0 and vmax <= 1.1:
        t = t * 2.0 - 1.0
    return t