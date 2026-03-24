 # !/usr/bin/env python
# -*- coding: UTF-8 -*-

import numpy as np
import torch
import os
import argparse
from .model_loader_utils import  clear_comfyui_cache,tensor2pillist
from .sparkvsr_inference_script import infer_sparkvsr,load_sparkvsr_model,per_video_refer,preprocess_video_match
from .finetune.PiSASR.test_pisasr import load_pisasr_model,infer_pisasr
import folder_paths
from typing_extensions import override
from comfy_api.latest import ComfyExtension, io
import nodes

MAX_SEED = np.iinfo(np.int32).max
node_cr_path = os.path.dirname(os.path.abspath(__file__))

device = torch.device(
    "cuda:0") if torch.cuda.is_available() else torch.device(
    "mps") if torch.backends.mps.is_available() else torch.device(
    "cpu")

weigths_GGUF_current_path = os.path.join(folder_paths.models_dir, "gguf")
if not os.path.exists(weigths_GGUF_current_path):
    os.makedirs(weigths_GGUF_current_path)
folder_paths.add_model_folder_path("gguf", weigths_GGUF_current_path) #  gguf dir


class SparkVSR_SM_Model(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        
        return io.Schema(
            node_id="SparkVSR_SM_Model",
            display_name="SparkVSR_SM_Model",
            category="SparkVSR_SM",
            inputs=[
                io.Combo.Input("dit",options= ["none"] + folder_paths.get_filename_list("diffusion_models") ),
                io.Combo.Input("gguf",options= ["none"] + folder_paths.get_filename_list("gguf") ),
                io.Combo.Input("vae",options= ["none"] + folder_paths.get_filename_list("vae") ),
                io.Combo.Input("lora",options= ["none"] + folder_paths.get_filename_list("loras") ),
                io.Combo.Input("dtype",options= ["bfloat16","float16","float32"] ),
                
            ],
            outputs=[
                io.Model.Output(display_name="model"),
                ],
            )
    @classmethod
    def execute(cls, dit,gguf,vae,lora,dtype) -> io.NodeOutput:
        model_path=folder_paths.get_full_path("diffusion_models", dit) if dit != "none" else None
        lora_path=folder_paths.get_full_path("loras", lora) if lora != "none" else None
        vae_path=folder_paths.get_full_path("vae",vae) if vae != "none" else None
        gguf_path=folder_paths.get_full_path("gguf", gguf) if gguf != "none" else None
        args = argparse.Namespace(
            model_path=model_path,
            dtype=dtype,
            lora_path=lora_path,
            vae_path=vae_path,
            gguf_path=gguf_path,
            repo=os.path.join(node_cr_path, "CogVideoX1.5-5B-I2V")),
        model=load_sparkvsr_model( args, device)
        return io.NodeOutput(model)
    

class SparkVSR_SM_KSampler(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SparkVSR_SM_KSampler",
            display_name="SparkVSR_SM_KSampler",
            category="SparkVSR_SM",
            inputs=[
                io.Model.Input("model"),   
                io.Conditioning.Input("conds"),  
                io.Int.Input("seed", default=0, min=0, max=MAX_SEED),
                io.Int.Input("overlap_t", default=8, min=1, max=nodes.MAX_RESOLUTION,),
                io.Int.Input("chunk_len", default=0, min=0, max=nodes.MAX_RESOLUTION),
                io.Combo.Input("overlap_hw", options= [[32,32],[64,64],[128,128],[256,256],[512,512]] ),
                io.Combo.Input("tile_size_hw",options= [[0,0],[8,8],[16,16],[32,32],[64,64]] ),
                io.Int.Input("noise_step", default=0, min=0, max=nodes.MAX_RESOLUTION),
                io.Int.Input("sr_noise_step", default=399, min=1, max=nodes.MAX_RESOLUTION),
                io.Float.Input("ref_guidance_scale", default=1.0, min=0.0, max=100.0, step=0.1,),
                io.Boolean.Input("offload", default=True),
                io.Int.Input("num_blocks_per_group", default=1, min=1, max=64, step=1, ),
            ],
            outputs=[
                io.Image.Output(display_name="images"),
            ],
        )
    @classmethod
    def execute(cls, model,conds,seed,overlap_t, chunk_len,overlap_hw,tile_size_hw,noise_step,sr_noise_step,ref_guidance_scale,offload,num_blocks_per_group) -> io.NodeOutput:
        if offload:
            from diffusers.hooks import apply_group_offloading
            apply_group_offloading(model.transformer, onload_device=torch.device("cuda"), offload_type="block_level", num_blocks_per_group=num_blocks_per_group,)
        else:
            model.to(device)
        clear_comfyui_cache()
        
        print(f"start inference with seed:{seed}")
        args = argparse.Namespace(
            seed=seed,
            is_vae_st=True,
            output_path=folder_paths.get_output_directory(),
            chunk_len=chunk_len,
            overlap_t=overlap_t,
            overlap_hw=overlap_hw,
            tile_size_hw=tile_size_hw,
            noise_step=noise_step,
            sr_noise_step=sr_noise_step,
            ref_guidance_scale=ref_guidance_scale,
            save_output=False,
            )
        images=infer_sparkvsr(args,model,conds)

        return io.NodeOutput(images)
    
class SparkVSR_SM_PreRefer(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SparkVSR_SM_PreRefer",
            display_name="SparkVSR_SM_PreRefer",
            category="SparkVSR_SM",
            inputs=[
                io.Image.Input("image"),
                io.String.Input("ref_indices", default=" 0,16,32", multiline=False),
                io.Int.Input("targe_width", default=0, min=0, max=nodes.MAX_RESOLUTION,step=32,display_mode=io.NumberDisplay.number),
                io.Int.Input("targe_height", default=0, min=0, max=nodes.MAX_RESOLUTION,step=32,display_mode=io.NumberDisplay.number),
                io.Int.Input("upscale", default=4, min=1, max=16),
                io.Model.Input("model",optional=True),
                io.Image.Input("sr_image",optional=True),
            ],
            outputs=[
                io.Conditioning.Output(display_name="conds"),
                ],
            )
    @classmethod
    def execute(cls, image,ref_indices,targe_width,targe_height,upscale,model=None,sr_image=None) -> io.NodeOutput:
        output_resolution=(targe_height,targe_width)

        # F,H,W,C=image.shape
        sr_embedding=None
        ref_mode="no_ref" 
        if sr_image is not None:
            ref_mode="SRimg_in"
            sr_image=tensor2pillist(sr_image)
        elif model is not  None:
            ref_mode= "pisasr"
            sr_embedding=os.path.join(node_cr_path,"finetune/sd2_pos_emptyemb_sm_.pt")
       
        args = argparse.Namespace(
            upscale=upscale,
            upscale_mode="bilinear",
            output_resolution=None if output_resolution == (0,0) else output_resolution,
            ref_mode=ref_mode,
            ref_indices=[int(i) for i in ref_indices.split(",")],
            output_path=folder_paths.get_output_directory(),
            pisa_python_executable="",
            pisa_script_path="",
            pisa_sd_model_path="",
            pisa_chkpt_path="",
            pisa_gpu=0,
            ref_api_cache_dir="",
            )
        conds=per_video_refer(args, image,model,sr_image,os.path.join(node_cr_path,"finetune/SparkVSR_pos_sm.pt"),sr_embedding)
        return io.NodeOutput(conds)

class SparkVSR_SM_SRModel(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        
        return io.Schema(
            node_id="SparkVSR_SM_SRModel",
            display_name="SparkVSR_SM_SRModel",
            category="SparkVSR_SM",
            inputs=[
                io.Combo.Input("unet",options= ["none"] + folder_paths.get_filename_list("diffusion_models") ),
                io.Combo.Input("vae",options= ["none"] + folder_paths.get_filename_list("vae") ),
                io.Combo.Input("pkl",options= ["none"] + folder_paths.get_filename_list("loras") ),
                io.Combo.Input("dtype",options= ["bfloat16","float16","float32"] ),
            ],
            outputs=[
                io.Model.Output(display_name="model"),
                ],
            )
    @classmethod
    def execute(cls, unet,vae,pkl,dtype) -> io.NodeOutput:
        unet_path=folder_paths.get_full_path("diffusion_models", unet) if unet != "none" else None
        pkl_path=folder_paths.get_full_path("loras", pkl) if pkl != "none" else None
        vae_path=folder_paths.get_full_path("vae",vae) if vae != "none" else None
        args = argparse.Namespace(
            unet_path=unet_path,
            mixed_precision=dtype,
            pretrained_path=pkl_path,
            vae_path=vae_path,
            pretrained_model_path=os.path.join(node_cr_path, "stable-diffusion-2-1-base"),
            vae_encoder_tiled_size=1024,
            vae_decoder_tiled_size=224,
            lambda_sem=1.0,
            lambda_pix=1.0,
            align_method="wavelet",
            process_size=512,
            latent_tiled_size=96,
            latent_tiled_overlap=32,
            default=True,
            upscale=4,
            seed=666,
            )
        model=load_pisasr_model( args)
        return io.NodeOutput(model)

# class SparkVSR_SM_SaveCond(io.ComfyNode):
#     @classmethod
#     def define_schema(cls):
#         return io.Schema(
#             node_id="SparkVSR_SM_SaveCond",
#             display_name="SparkVSR_SM_SaveCond",
#             category="SparkVSR_SM",
#             inputs=[
#                 io.Image.Input("image"),
#                 io.Model.Input("model",optional=True),  
#                 io.Conditioning.Input("pos",optional=True),
#             ],
#             outputs=[
#                 io.String.Output(display_name="done"),
#                 ],
#             )
#     @classmethod
#     def execute(cls,image,model=None, pos=None) -> io.NodeOutput:
        # if pos is not None:
        #     default_path=os.path.join(folder_paths.get_output_directory(),"sd2_pos_emptyemb_sm_.pt")
        #     from transformers import AutoTokenizer, CLIPTextModel
            
        #     text_encoder = CLIPTextModel.from_pretrained("D:/Downloads/sd2", subfolder="text_encoder").cuda()
        #     text_encoder.requires_grad_(False)
        #     text_encoder.eval()
        #     tokenizer = AutoTokenizer.from_pretrained('F:/ComfyUI311/ComfyUI/custom_nodes/ComfyUI_SparkVSR_SM/stable-diffusion-2-1-base', subfolder="tokenizer")
        #     def encode_prompt(prompt_batch,text_encoder,tokenizer):
        #         """Encode text prompts into embeddings."""
        #         with torch.no_grad():
        #             prompt_embeds = [
        #                 text_encoder(
        #                     tokenizer(
        #                         caption, max_length=tokenizer.model_max_length,
        #                         padding="max_length", truncation=True, return_tensors="pt"
        #                     ).input_ids.to(text_encoder.device)
        #                 )[0]
        #                 for caption in prompt_batch
        #             ]
        #         return torch.concat(prompt_embeds, dim=0)
        #     prompt_batch=[""]
        #     pos=encode_prompt(prompt_batch,text_encoder,tokenizer)
        #     print("pos shape:",pos.shape)
        #     torch.save(pos,default_path)
        
        # if model is not None:
        #     ref_indices=[0,10]
        #     prompt_embeds=torch.load(os.path.join(node_cr_path,"finetune/sd2_pos_emptyemb_sm_.pt"), map_location="cpu",weights_only=False)
        #     video, pad_f, pad_h, pad_w, original_shape = preprocess_video_match(image, is_match=True)
        #     pli_list=infer_pisasr(model,ref_indices, video,prompt_embeds )
        #     for img,idx in zip(pli_list,ref_indices):
        #         img.save(os.path.join(folder_paths.get_output_directory(),"sr_img_{}.png".format(idx)))
        # return io.NodeOutput("done")
    
class SparkVSR_SM_Extension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            SparkVSR_SM_Model,
            SparkVSR_SM_KSampler,
            SparkVSR_SM_PreRefer,
            SparkVSR_SM_SRModel,
            #SparkVSR_SM_SaveCond,
        ]
async def comfy_entrypoint() -> SparkVSR_SM_Extension:  # ComfyUI calls this to load your extension and its nodes.
    return SparkVSR_SM_Extension()



