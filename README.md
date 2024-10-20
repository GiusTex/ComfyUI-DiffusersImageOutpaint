ComfyUI nodes for outpainting images with diffusers, based on [diffusers-image-outpaint](https://huggingface.co/spaces/fffiloni/diffusers-image-outpaint/tree/main) by fffiloni.

![image](https://github.com/user-attachments/assets/8f7665a1-dd8c-44d6-a067-fcc3f48b1865)


#### Updates:
- 20/10/2024: No more need to download tokenizers nor text encoders! Now comfyui clip loader works, and you can use your clip models. You can also use the Checkpoint Loader Simple node, to skip the clip selection part.
- 10/2024: You don't need any more the diffusers vae, and can use the extension in low vram mode using `sequential_cpu_offload` (also thanks to [zmwv823](https://github.com/GiusTex/ComfyUI-DiffusersImageOutpaint/pull/4)) that pushes the vram usage from *8,3 gb* down to **_6 gb_**.
- 10/2024: If your `text_encoder` and `text_encoder_2` names contain `.fp16.` or other things before `safetensors`, you need to remove it (see the table below).

#### To do list to [change model used](https://github.com/GiusTex/ComfyUI-DiffusersImageOutpaint/pull/14):
- [x] ComfyUI Clip Loader Node
- [ ] ComfyUI Load Diffusion Model Node
- [ ] ComfyUI Load Conotrolnet Model Node

## Installation
- Download this extension or `git clone` it in comfyui/custom_nodes, then (if comfyui-manager didn't already install the requirements or you have missing modules), from comfyui virtual env write `cd your/path/to/this/extension` and `pip install -r requirements.txt`.
- Download models in the **`comfyui/models/diffusion_models`** folder, following the grid below (you can use the links to download the suggested models; you can also change the main model, but you need the specified controlnet since the extension is hardcoded to use it, for now):
  | 	**main model**	 | 	**controlnet model**	 |
  | 	:-----:	 | 	:-----:	 |
  | 	**[Diffuser Model folder](https://huggingface.co/SG161222/RealVisXL_V5.0_Lightning/tree/main)** (you can change this model)	|	**[Diffuser Controlnet folder](https://huggingface.co/xinsir/controlnet-union-sdxl-1.0/tree/main)** (you need this model)	 |
  | 	model_index-json	|	config_promax.json, diffusion_pytorch_model_promax.safetensors	 |
  | 	**Unet folder**	| |
  | 	config.json, diffusion_pytorch_model.fp16.safetensors	| |
  | 	**Scheduler folder**	| |
  | 	scheduler_config.json	| |
  
## Overview
- **Minimum VRAM**: 6 gb with 1280x720 image, rtx 3060, RealVisXL_V5.0_Lightning, sdxl-vae-fp16-fix, controlnet-union-sdxl-promax using `sequential_cpu_offload`, otherwise 8,3 gb;
- As seen in [this issue](https://github.com/GiusTex/ComfyUI-DiffusersImageOutpaint/issues/7#issuecomment-2410852908), images with **square corners** are required.

The extension gives 4 nodes:
- **Load Diffusion Outpaint Models**: a simple node to load diffusion `models`. You can download them from Huggingface (the extension doesn't download them automatically);
- **Paid Image for Diffusers Outpaint**: this node creates an empty image of the `desired size`, fits the original image in the new one based on the chosen `alignment`, then mask the rest;
- **Encode Diffusers Outpaint Prompt**: self explanatory. Works as `clip text encode (prompt)`, and specifies what to add to the image;
- **Diffusers Image Outpaint**: This is the main node, that outpaints the image. Currently the generation process is based on fffiloni's one, so you can't reproduce a specific a specific outpaint, and the `seed` option you see is only used to change the UI and generate a new image. You can specify the amount of `steps` to generate the image.

- You can also pass image and mask to `vae encode (for inpainting)` node, then pass the latent to a `sampler`, but controlnets and ip-adapters are harder to use compared to diffusers outpaint.

## Credits
diffusers-image-outpaint by [fffiloni](https://huggingface.co/spaces/fffiloni/diffusers-image-outpaint/tree/main)
