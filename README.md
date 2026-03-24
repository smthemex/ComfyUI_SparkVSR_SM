<div align="center">
  <p><img src="assets/logo2.png" width="360px"></p>
  <h1>SparkVSR: Interactive Video Super-Resolution via Sparse Keyframe Propagation</h1>
  <p>
    Jiongze Yu<sup>1</sup>, Xiangbo Gao<sup>1</sup>, Pooja Verlani<sup>2</sup>, Akshay Gadde<sup>2</sup>,
    Yilin Wang<sup>2</sup>, Balu Adsumilli<sup>2</sup>, Zhengzhong Tu<sup>†,1</sup>
  </p>
  <p>
    <sup>1</sup>Texas A&amp;M University &nbsp;&nbsp; <sup>2</sup>YouTube, Google
    <br>
    <sup>†</sup>Corresponding author
  </p>
  <p>
    <a href="https://sparkvsr.github.io/"><img src="https://img.shields.io/badge/Project-Page-Green"></a>
    &nbsp;
    <a href="https://huggingface.co/JiongzeYu/SparkVSR"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue"></a>
    &nbsp;
    <a href="https://arxiv.org/abs/2603.16864"><img src="https://img.shields.io/badge/arXiv-2603.16864-b31b1b.svg"></a>
  </p>
</div>

> 💡 **Your ⭐ star means a lot to us and helps support the continuous development of this project!**

# ComfyUI_SparkVSR_SM
-----
[SparkVSR](https://github.com/taco-group/SparkVSR): Interactive Video Super-Resolution via Sparse Keyframe Propagation

Update
-----
* gguf mabe can't work,use safetensors . Preprocess the image node to connect to the pisa_sr node to load the model, or use other methods to upscale a single-frame image, that is, enable reference image mode; otherwise, it is non-reference image mode.  ref_indices are the sequence frame numbers of the reference image in the source video. For example,' 0, 'is the first frame. You can also input 0 to only reference the first frame to save resources and achieve better results.
* gguf模式目前还未测试是否可用。预处理图片节点接入pisa_sr 节点加载的模型  或者你用其他方法放大的单帧图片，即开启图模式，否则是非垫图模式，ref_indices是垫图在源视频的序列帧序号，比如0就是首帧，你也可以输入'0,'(注意有逗号) 只参考首帧以节省资源获得更好的效果.

1.Installation  
-----
  In the ./ComfyUI/custom_nodes directory, run the following:   
```
git clone https://github.com/smthemex/ComfyUI_SparkVSR_SM
```
2.requirements  
----

```
pip install -r requirements.txt
```

3.checkpoints 
----
* cog unet merged  [links](https://huggingface.co/smthem/SparkVSR-GGUF/tree/main) #f32 or bf16
*  cog vae [links](https://huggingface.co/JiongzeYu/SparkVSR/tree/main)
* if use pisa_sr need [stable-diffusion-2-1-base](https://www.modelscope.cn/models/stabilityai/stable-diffusion-2-1-base) vae and unet,and  pisa_sr.pkl[google](https://drive.google.com/drive/folders/1oLetijWNd59xwJE5oU-eXylQBifxWdss)

```
├── ComfyUI/models/
|     ├── diffusion_models/
|        ├──sd21base-f32.safetensors  # 3.22G or fb16  if use  pisa_sr to refer image  optional 可选，使用 pisa_sr超分参考图用
|        ├──SparkVSR-S2-F32.safetensors # 20.7G or fb16 
|     ├── vae/
|        ├──CogVideoX1.5-5B-I2V-VAE.safetensors #822M
|        ├──sd21vae.safetensors # 319M  if use  pisa_sr to refer image  optional 可选，使用 pisa_sr超分参考图用
|     ├── loras
|        ├──pisa_sr.pkl # 32M  if use  pisa_sr to refer image optional 可选,使用 pisa_sr超分参考图用

```

4.Example
----
![](https://github.com/smthemex/ComfyUI_SparkVSR_SM/blob/main/example_workflows/exampS2.png)


5.Citation
----
```
@misc{yu2026sparkvsrinteractivevideosuperresolution,
      title={SparkVSR: Interactive Video Super-Resolution via Sparse Keyframe Propagation}, 
      author={Jiongze Yu and Xiangbo Gao and Pooja Verlani and Akshay Gadde and Yilin Wang and Balu Adsumilli and Zhengzhong Tu},
      year={2026},
      eprint={2603.16864},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2603.16864}, 
}
```
