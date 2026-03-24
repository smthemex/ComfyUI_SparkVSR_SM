import os
import argparse
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
import torchvision.transforms.functional as F

from .pisasr import PiSASR_eval
from .src.my_utils.wavelet_color_fix import adain_color_fix, wavelet_color_fix

import glob


def load_pisasr_model(args):
    # Initialize the model
    model = PiSASR_eval(args)
    model.set_eval()
    return model

def infer_pisasr(model, ref_indices, video_lr,prompt_embeds,upscale):
    args=model.args
    image_list=[]
    time_records = []
    for idx in ref_indices:
        # Ensure the input image is a multiple of 8
        lr_frame = (video_lr[idx].cpu()*255).permute(1, 2, 0).numpy() # [H, W, C] in [0, 255]
        input_image = Image.fromarray(lr_frame.astype('uint8')).convert('RGB')
        #input_image.save(f'temp{idx}.png')
        #input_image = Image.open(image_name).convert('RGB')

        ori_width, ori_height = input_image.size
        rscale = upscale
        resize_flag = False

        if ori_width < args.process_size // rscale or ori_height < args.process_size // rscale:
            scale = (args.process_size // rscale) / min(ori_width, ori_height)
            input_image = input_image.resize((int(scale * ori_width), int(scale * ori_height)))
            resize_flag = True

        input_image = input_image.resize((input_image.size[0] * rscale, input_image.size[1] * rscale))
        new_width = input_image.width - input_image.width % 8
        new_height = input_image.height - input_image.height % 8
        input_image = input_image.resize((new_width, new_height), Image.LANCZOS)
        #bname = os.path.basename(image_name)

        # Get caption (you can add the text prompt here)
        #prompt_embeds = ''

        # Translate the image
        with torch.no_grad():
            c_t = F.to_tensor(input_image).unsqueeze(0).cuda() * 2 - 1
            inference_time, output_image = model(args.default, c_t, prompt_embeds=prompt_embeds)

        print(f"Inference time: {inference_time:.4f} seconds")
        time_records.append(inference_time)

        output_image = output_image * 0.5 + 0.5
        output_image = torch.clip(output_image, 0, 1)
        output_pil = transforms.ToPILImage()(output_image[0].cpu())

        if args.align_method == 'adain':
            output_pil = adain_color_fix(target=output_pil, source=input_image)
        elif args.align_method == 'wavelet':
            output_pil = wavelet_color_fix(target=output_pil, source=input_image)

        # if resize_flag:
        #     output_pil = output_pil.resize((int(args.upscale * ori_width), int(args.upscale * ori_height)))
        # output_pil.save(os.path.join(args.output_dir, bname))
        image_list.append(output_pil)
    # Calculate the average inference time, excluding the first few for stabilization
    if len(time_records) > 3:
        average_time = np.mean(time_records[3:])
    else:
        average_time = np.mean(time_records)
    print(f"Average inference time: {average_time:.4f} seconds")
    return image_list


def pisa_sr(args):
    # Initialize the model
    model = PiSASR_eval(args)
    model.set_eval()

    # Get all input images
    if os.path.isdir(args.input_image):
        image_names = sorted(glob.glob(f'{args.input_image}/*.png'))
    else:
        image_names = [args.input_image]

    # Make the output directory
    os.makedirs(args.output_dir, exist_ok=True)
    print(f'There are {len(image_names)} images.')

    time_records = []
    for image_name in image_names:
        # Ensure the input image is a multiple of 8
        input_image = Image.open(image_name).convert('RGB')
        ori_width, ori_height = input_image.size
        rscale = args.upscale
        resize_flag = False

        if ori_width < args.process_size // rscale or ori_height < args.process_size // rscale:
            scale = (args.process_size // rscale) / min(ori_width, ori_height)
            input_image = input_image.resize((int(scale * ori_width), int(scale * ori_height)))
            resize_flag = True

        input_image = input_image.resize((input_image.size[0] * rscale, input_image.size[1] * rscale))
        new_width = input_image.width - input_image.width % 8
        new_height = input_image.height - input_image.height % 8
        input_image = input_image.resize((new_width, new_height), Image.LANCZOS)
        bname = os.path.basename(image_name)

        # Get caption (you can add the text prompt here)
        validation_prompt = ''

        # Translate the image
        with torch.no_grad():
            c_t = F.to_tensor(input_image).unsqueeze(0).cuda() * 2 - 1
            inference_time, output_image = model(args.default, c_t, prompt=validation_prompt)

        print(f"Inference time: {inference_time:.4f} seconds")
        time_records.append(inference_time)

        output_image = output_image * 0.5 + 0.5
        output_image = torch.clip(output_image, 0, 1)
        output_pil = transforms.ToPILImage()(output_image[0].cpu())

        if args.align_method == 'adain':
            output_pil = adain_color_fix(target=output_pil, source=input_image)
        elif args.align_method == 'wavelet':
            output_pil = wavelet_color_fix(target=output_pil, source=input_image)

        if resize_flag:
            output_pil = output_pil.resize((int(args.upscale * ori_width), int(args.upscale * ori_height)))
        output_pil.save(os.path.join(args.output_dir, bname))

    # Calculate the average inference time, excluding the first few for stabilization
    if len(time_records) > 3:
        average_time = np.mean(time_records[3:])
    else:
        average_time = np.mean(time_records)
    print(f"Average inference time: {average_time:.4f} seconds")


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--input_image', '-i', type=str, default='preset/test_datasets', help="path to the input image")
#     parser.add_argument('--output_dir', '-o', type=str, default='experiments/test', help="the directory to save the output")
#     parser.add_argument("--pretrained_model_path", type=str, default='preset/models/stable-diffusion-2-1-base')
#     parser.add_argument('--pretrained_path', type=str, default='preset/models/pisa_sr.pkl', help="path to a model state dict to be used")
#     parser.add_argument('--seed', type=int, default=42, help="Random seed to be used")
#     parser.add_argument("--process_size", type=int, default=512)
#     parser.add_argument("--upscale", type=int, default=4)
#     parser.add_argument("--align_method", type=str, choices=['wavelet', 'adain', 'nofix'], default="adain")
#     parser.add_argument("--lambda_pix", default=1.0, type=float, help="the scale for pixel-level enhancement")
#     parser.add_argument("--lambda_sem", default=1.0, type=float, help="the scale for sementic-level enhancements")
#     parser.add_argument("--vae_decoder_tiled_size", type=int, default=224)
#     parser.add_argument("--vae_encoder_tiled_size", type=int, default=1024)
#     parser.add_argument("--latent_tiled_size", type=int, default=96) 
#     parser.add_argument("--latent_tiled_overlap", type=int, default=32) 
#     parser.add_argument("--mixed_precision", type=str, default="fp16")
#     parser.add_argument("--default",  action="store_true", help="use default or adjustale setting?") 

#     args = parser.parse_args()

#     # Call the processing function
#     pisa_sr(args)
