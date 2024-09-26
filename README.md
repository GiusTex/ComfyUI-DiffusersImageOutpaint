## Info
ComfyUI nodes for outpainting images with diffusers, based on [diffusers-image-outpaint](https://huggingface.co/spaces/fffiloni/diffusers-image-outpaint/tree/main) by fffiloni.

![DiffusersImageOutpaint-Nodes-Screen](https://github.com/user-attachments/assets/df6ee871-08ab-4e34-b47e-410673a026ed)

The extension gives 3 nodes:
- #### Load Diffusion Outpaint Models
  
  a simple node to load diffusion models; you can download them from Huggingface (the extension doesn't download them automatically), then you have to follow this order (the model links are an example):
  | 	ComfyUI/**Unet folder**	 | 	ComfyUI/**Vae folder**	 | 	ComfyUI/**Controlnet folder**	 |
  | 	:-----:	 | 	:-----:	 | 	:-----:	 |
  | 	**[Diffuser Model folder](https://huggingface.co/SG161222/RealVisXL_V5.0_Lightning/tree/main)**	| 	**[Diffuser Vae folder](https://huggingface.co/madebyollin/sdxl-vae-fp16-fix/tree/main)**	| 	**[Diffuser Controlnet folder](https://huggingface.co/xinsir/controlnet-union-sdxl-1.0/tree/main)**	 |
  | 	**Unet folder**	| 	config.json, sdxl_vae.safetensors	| 	config_promax.json, diffusion_pytorch_model_promax.safetensors	 |
  | 	config.json, diffusion_pytorch_model.fp16.safetensors	| | |
  | 	**Scheduler**	| | |
  | 	File 1, File 2, File 3	| | |
  | 	**Text encoder folder**	| | |
  | 	File 1, File 2, File 3	| | |
  | 	**Text encoder 2 folder**	| | |
  | 	File 1, File 2, File 3	| | |
  | 	**Tokenizer folder**	| | |
  | 	File 1, File 2, File 3	| | |
  | 	**Tokenizer 2 folder**	| | |
  | 	File 1, File 2, File 3	| | |

- ****Paid Image for Diffusers Outpaint****: this node creates an empty image of the desired size, fits the original image in the new one based on the chosen alignment, then mask the rest;
- ****Diffusers Image Outpaint****: This is the main node, that outpaints the image. Currently the generation process is based on fffiloni's one, so you can't reproduce a specific a specific outpaint, and the seed option you see is only used to change the UI and generate a new image.
