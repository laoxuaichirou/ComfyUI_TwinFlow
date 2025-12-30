# ComfyUI_TwinFlow
[TwinFlow](https://github.com/inclusionAI/TwinFlow): Realizing One-step Generation on Large Models with Self-adversarial Flows,you can use it in comfyUI

# Tips
* clip ：only unload comfy clip ，none：don't unload any ，all，unload all  comfyUI模型卸载模式，clip是保留vae，all是全卸载，none是不卸载（需要大显存）
* support z-image，fix bug ，fix qwen-image gguf （need replace）/ qwen-image的gguf做了重新量化，请更新避免dtype错误；
* z-image Q8 ，12G 1024*768 2-3s/pic , /12G Q8 不开启卸载，1027*768 2-3秒1张图
* qwen-image infer 1027*768 in 12G Vram just need 15s / 12G (50 block)显存 开GPU卸载 单图12G vram只需要15秒 (50 block)
* if VRAM>16, set block number to 0 to use high VRAM / 大显存设置lock number为0 以达到最快推理速度；

# 1.Installation  

* In the ' ./ComfyUI/custom_nodes ' directory, run the following:   

```
git clone https://github.com/smthemex/ComfyUI_TwinFlow

```

# 2.requirements  

```
pip install -r requirements.txt
```

# 3.checkpoints 

* 3.1 [qwen-image gguf](https://huggingface.co/smthem/TwinFlow-Qwen-Image-v1.0-diffusers-gguf/tree/main)  only support gguf now / 有一个 的其他开发者打包safetensor 
* 3.2 [ z-image gguf and safetensors](https://huggingface.co/smthem/TwinFlow-Z-Image-Turbo-diffuser-gguf) / z-image的模型地址
* 3.3 qwen-image [（vae，clip）](https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/tree/main/split_files)  /常规clip和vae
* 3.4 z-image [（vae，clip）](https://huggingface.co/Comfy-Org/z_image_turbo/tree/main/split_files)  /常规z-image clip和vae

```
├── ComfyUI/models/gguf
|     ├── TwinFlow-Qwen-Image-diffusers-Q6_K.gguf  # or Q8_0、 BF16
|     ├── TwinFlow-Z-Image-Turbo-diffuser-Q8_0.gguf # or 
├── ComfyUI/models/vae
|     ├──qwen_image_vae.safetensors
|     ├──ae.safetensors #z-image use flux
├── ComfyUI/models/clip 
|     ├──qwen_2.5_vl_7b_fp8_scaled.safetensors # or bf16
|     ├──qwen_3_4b.safetensors # z image
```

# 4. Example
* qwen-image
![](https://github.com/smthemex/ComfyUI_TwinFlow/blob/main/example_workflows/example.png)
* z-image
![](https://github.com/smthemex/ComfyUI_TwinFlow/blob/main/example_workflows/example-z.png)

# 5. Citation
```
@article{cheng2025twinflow,
  title={TwinFlow: Realizing One-step Generation on Large Models with Self-adversarial Flows},
  author={Cheng, Zhenglin and Sun, Peng and Li, Jianguo and Lin, Tao},
  journal={arXiv preprint arXiv:2512.05150},
  year={2025}
}

```
