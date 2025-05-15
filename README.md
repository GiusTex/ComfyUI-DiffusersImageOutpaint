ComfyUI nodes for outpainting images with diffusers, based on [diffusers-image-outpaint](https://huggingface.co/spaces/fffiloni/diffusers-image-outpaint/tree/main) by fffiloni.

![image](https://github.com/user-attachments/assets/1a02c2d1-f24e-4ad2-acdc-a2cbb15a1f14)

#### Updates:
- 15/05/2025: Fixed `missing 'loaded_keys'` error. More details below.
- 17/11/2024:
  - Added more options to Pad Image node (resize image, custom resize image percentage, mask overlap percentage, overlap left/right/top/bottom).
  - Side notes:
    - Now images with round angles work, since the new editable mask covers them, like in the original huggingface space.
    - You can use "mask" and "diffusers outpaint cnet image" outputs to preview mask and image.
    - You can find in the same [workflow file](https://github.com/GiusTex/ComfyUI-DiffusersImageOutpaint/blob/New-Pad-Node-Options/Diffusers-Outpaint-DoubleWorkflow.json) the workflow with the checkpoint-loader-simple node and another one with clip + vae loader nodes.
- 22/10/2024:
  - Unet and Controlnet Models Loader using ComfYUI nodes canceled, since I can't find a way to load them properly; more info at the end.
  - Guide to change model used.
- 20/10/2024: No more need to download tokenizers nor text encoders! Now comfyui clip loader works, and you can use your clip models. You can also use the Checkpoint Loader Simple node, to skip the clip selection part.
- 10/2024: You don't need any more the diffusers vae, and can use the extension in low vram mode using `sequential_cpu_offload` (also thanks to [zmwv823](https://github.com/GiusTex/ComfyUI-DiffusersImageOutpaint/pull/4)) that pushes the vram usage from *8,3 gb* down to **_6 gb_**.

## Installation
- Download this extension or `git clone` it in comfyui/custom_nodes, then (if comfyui-manager didn't already install the requirements or you have missing modules), from comfyui virtual env write `cd your/path/to/this/extension` and `pip install -r requirements.txt`.
- Download a sdxl model ([example](https://huggingface.co/SG161222/RealVisXL_V5.0_Lightning/resolve/main/unet/diffusion_pytorch_model.fp16.safetensors)) in comfyui/models/diffusion_models;
- Download a sdxl controlnet model ([example](https://huggingface.co/xinsir/controlnet-union-sdxl-1.0/blob/main/diffusion_pytorch_model_promax.safetensors)) in comfyui/models/controlnet.

- (Dual) Clip Loader node: if you use the Clip Loader instead of Checkpoint Loader Simple, and want to use an `sdxl type` model like RealVisXL_V5.0_Lightning, you can download `clip_I` and `clip_g` from [here](https://huggingface.co/Comfy-Org/stable-diffusion-3.5-fp8/tree/main/text_encoders). You can use [this workflow](https://github.com/GiusTex/ComfyUI-DiffusersImageOutpaint/blob/New-Pad-Node-Options/Diffusers-Outpaint-DoubleWorkflow.json) (change model.fp16 with `clip_g`).

## Overview
- **Minimum VRAM**: 6 gb with 1280x720 image, rtx 3060, RealVisXL_V5.0_Lightning, sdxl-vae-fp16-fix, controlnet-union-sdxl-promax using `sequential_cpu_offload`, otherwise 8,3 gb;
- ~As seen in [this issue](https://github.com/GiusTex/ComfyUI-DiffusersImageOutpaint/issues/7#issuecomment-2410852908), images with **square corners** are required~.

The extension gives 5 nodes:
- **Load Diffuser Model**: a simple node to load diffusion `models`. You can download them from Huggingface (the extension doesn't download them automatically). Put them inside the `diffusion_models` folder;
- **Load Diffuser Controlnet**: a simple node to load diffusion `models`. You can download them from Huggingface (the extension doesn't download them automatically).  Put them inside the `controlnet` folder;
- **Paid Image for Diffusers Outpaint**: this node resizes the image based on the specified `width` and `height`, then resizes it again based on the `resize_image` percentage, and if possible it will put the mask based on the `alignment` specified, otherwise it will revert back to the default "middle" `alignment`;
- **Encode Diffusers Outpaint Prompt**: self explanatory. Works as `clip text encode (prompt)`, and specifies what to add to the image;
- **Diffusers Image Outpaint**: This is the main node, that outpaints the image. Currently the generation process is based on fffiloni's one, so you can't reproduce a specific a specific outpaint, and the `seed` option you see is only used to update the UI and generate a new image. You can specify the amount of `steps` to generate the image.

You _can_ also pass image and mask to `vae encode (for inpainting)` node, then pass the latent to a `sampler`, but controlnets and ip-adapters won't always give good results like with diffusers outpaint, and they require a different workflow, not covered by this extension.

Since for now only sdxl models work, the config are chosen automatically. If in the future other types that would require different config will work, I could add more selection options.

### Change model used
- **Main model**: On huggingface, choose a model from [text2image models](https://huggingface.co/models?pipeline_tag=text-to-image&sort=trending) (**sdxl and maybe sd1.5 model types should work, while flux doesn't**), then create a new folder named after it in `comfyui/models/diffusion_models`, then download in it the subfolders `unet` (if not available use `transformer`) and `scheduler`.
  - Hint: sometimes in the `unet` or `transformer` folder there are more model files and not all are required. If you have `model.fp16` and `model`, I suggest you to use the fp16 variant; if you have `model-001-of-002`, `model-002-of-002`, `model`, choose model (instead of the fragmented version).
- **Controlnet model**: download `config.json` and the safetensors `model`.

## Missing 'loaded_keys' error
Recent versions of `transformers` and `diffusers` broke somethings, you need to revert back, command with some working versions (found [here](https://huggingface.co/spaces/fffiloni/diffusers-image-outpaint/blob/main/requirements.txt)) (do it inside your comfyui env): `pip install transformers==4.45.0 --upgrade diffusers==0.32.2 --upgrade`.

## Credits
diffusers-image-outpaint by [fffiloni](https://huggingface.co/spaces/fffiloni/diffusers-image-outpaint/tree/main)
