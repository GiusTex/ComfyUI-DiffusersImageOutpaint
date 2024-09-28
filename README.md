ComfyUI nodes for outpainting images with diffusers, based on [diffusers-image-outpaint](https://huggingface.co/spaces/fffiloni/diffusers-image-outpaint/tree/main) by fffiloni.

![DiffusersImageOutpaint-Nodes-Screen](https://github.com/user-attachments/assets/df6ee871-08ab-4e34-b47e-410673a026ed)

## Installation
- Download this extension or `git clone` it in comfyui/custom_nodes, then (if comfyui-manager didn't already install the requirements or you have missing modules), from comfyui virtual env write `cd your/path/to/this/extension` and `pip install -r requirements.txt`.
- Download models in the **`comfyui/models/diffusion_models`** folder, following the grid below (you can use the links to download the suggested models; you can also change the main model, but you need the specified vae and controlnet since the extension is hardcoded to use them. You can always change the code to use different models):
  | 	**main model**	 | 	**controlnet model**	 | 	**vae model**	 |
  | 	:-----:	 | 	:-----:	 | 	:-----:	 |
  | 	**[Diffuser Model folder](https://huggingface.co/SG161222/RealVisXL_V5.0_Lightning/tree/main)** (you can change this model)	| 	**[Diffuser Vae folder](https://huggingface.co/madebyollin/sdxl-vae-fp16-fix/tree/main)** (you need this model)	| 	**[Diffuser Controlnet folder](https://huggingface.co/xinsir/controlnet-union-sdxl-1.0/tree/main)** (you need this model)	 |
  | 	model_index-json	| 	config.json, sdxl_vae.safetensors	| 	config_promax.json, diffusion_pytorch_model_promax.safetensors	 |
  | 	**Unet folder**	| | |
  | 	config.json, diffusion_pytorch_model.fp16.safetensors	| | |
  | 	**Scheduler folder**	| | |
  | 	scheduler_config.json	| | |
  | 	**Text encoder folder**	| | |
  | 	config.json, model.fp16.safetensors	| | |
  | 	**Text encoder 2 folder**	| | |
  | 	config.json, model.fp16.safetensors	| | |
  | 	**Tokenizer folder**	| | |
  | 	merges.txt, special_tokens_map.json, tokenizer_config.json, vocab.json	| | |
  | 	**Tokenizer 2 folder**	| | |
  | 	merges.txt, special_tokens_map.json, tokenizer_config.json, vocab.json	| | |
  
## Overview
**Minimum VRAM**: 8,4 gb with `model_cpu_offload`, `vae_slicing` and 1280x720 image, on rtx 3060, but [zmwv823](https://github.com/GiusTex/ComfyUI-DiffusersImageOutpaint/issues/3#issue-2554112238) used even less, 5,6 gb, so I'd say vram usage is between those values.

The extension gives 3 nodes:
- **Load Diffusion Outpaint Models**: a simple node to load diffusion `models`. You can download them from Huggingface (the extension doesn't download them automatically);
- **Paid Image for Diffusers Outpaint**: this node creates an empty image of the `desired size`, fits the original image in the new one based on the chosen `alignment`, then mask the rest;
- **Diffusers Image Outpaint**: This is the main node, that outpaints the image. Currently the generation process is based on fffiloni's one, so you can't reproduce a specific a specific outpaint, and the `seed` option you see is only used to change the UI and generate a new image. Anyway, you can specify the amount of `steps` to generate the image and the `prompt` to specify what to add to the image.

- You can also pass image and mask to `vae encode (for inpainting)` node, then pass the latent to a `sampler`, but controlnets and ip-adapters are harder to use compared to diffusers outpaint.

## Credits
diffusers-image-outpaint by [fffiloni](https://huggingface.co/spaces/fffiloni/diffusers-image-outpaint/tree/main)
