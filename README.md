ComfyUI nodes for outpainting images with diffusers, based on [diffusers-image-outpaint](https://huggingface.co/spaces/fffiloni/diffusers-image-outpaint/tree/main) by fffiloni.

![Extension-Overview](https://github.com/user-attachments/assets/b801698e-e666-4179-98bd-42dfb1f033ba)

#### Updates:
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

#### To do list to [change model used](https://github.com/GiusTex/ComfyUI-DiffusersImageOutpaint/pull/14):
- - [x] ComfyUI Clip Loader Node
- ~[ ] ComfyUI Load Diffusion Model Node~ (more info [below](https://github.com/GiusTex/ComfyUI-DiffusersImageOutpaint#unet-and-controlnet-models-loader-using-comfyui-nodes-canceled))
- ~[ ] ComfyUI Load Conotrolnet Model Node~ (more info [below](https://github.com/GiusTex/ComfyUI-DiffusersImageOutpaint#unet-and-controlnet-models-loader-using-comfyui-nodes-canceled))

## Installation
- Download this extension or `git clone` it in comfyui/custom_nodes, then (if comfyui-manager didn't already install the requirements or you have missing modules), from comfyui virtual env write `cd your/path/to/this/extension` and `pip install -r requirements.txt`.
- Download models in comfyui/models/diffusion_models:
   - model_name:
      - unet:
         - `diffusion_pytorch_model.fp16.safetensors` ([example](https://huggingface.co/SG161222/RealVisXL_V5.0_Lightning/blob/main/unet/diffusion_pytorch_model.fp16.safetensors))
         - `config.json` ([example](https://huggingface.co/SG161222/RealVisXL_V5.0_Lightning/blob/main/unet/config.json))
     - scheduler:
       - `scheduler_config.json` ([example](https://huggingface.co/SG161222/RealVisXL_V5.0_Lightning/blob/main/scheduler/scheduler_config.json))
     - `model_index.json` ([example](https://huggingface.co/SG161222/RealVisXL_V5.0_Lightning/blob/main/model_index.json))
   - controlnet_name:
     - `config_promax.json` ([example](https://huggingface.co/xinsir/controlnet-union-sdxl-1.0/blob/main/config_promax.json)), `diffusion_pytorch_model_promax.safetensors` ([example](https://huggingface.co/xinsir/controlnet-union-sdxl-1.0/blob/main/diffusion_pytorch_model_promax.safetensors))
- (Dual) Clip Loader node: if you use the Clip Loader instead of Checkpoint Loader Simple, and want to use RealVisXL_V5.0_Lightning, it works with [`clip_I`](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/text_encoder/model.fp16.safetensors) and [`model.fp16`](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/text_encoder_2/model.fp16.safetensors) (from sdxl-base), and `sdxl type`; you can use [this workflow](https://github.com/GiusTex/ComfyUI-DiffusersImageOutpaint/blob/New-Pad-Node-Options/Diffusers-Outpaint-DoubleWorkflow.json).

## Overview
- **Minimum VRAM**: 6 gb with 1280x720 image, rtx 3060, RealVisXL_V5.0_Lightning, sdxl-vae-fp16-fix, controlnet-union-sdxl-promax using `sequential_cpu_offload`, otherwise 8,3 gb;
- ~As seen in [this issue](https://github.com/GiusTex/ComfyUI-DiffusersImageOutpaint/issues/7#issuecomment-2410852908), images with **square corners** are required~.

The extension gives 4 nodes:
- **Load Diffusion Outpaint Models**: a simple node to load diffusion `models`. You can download them from Huggingface (the extension doesn't download them automatically);
- **Paid Image for Diffusers Outpaint**: this node resizes the image based on the specified `width` and `height`, then resizes it again based on the `resize_image` percentage, and if possible it will put the mask based on the `alignment` specified, otherwise it will revert back to the default "middle" `alignment`;
- **Encode Diffusers Outpaint Prompt**: self explanatory. Works as `clip text encode (prompt)`, and specifies what to add to the image;
- **Diffusers Image Outpaint**: This is the main node, that outpaints the image. Currently the generation process is based on fffiloni's one, so you can't reproduce a specific a specific outpaint, and the `seed` option you see is only used to update the UI and generate a new image. You can specify the amount of `steps` to generate the image.

- You can also pass image and mask to `vae encode (for inpainting)` node, then pass the latent to a `sampler`, but controlnets and ip-adapters are harder to use compared to diffusers outpaint.

### Change model used
- **Main model**: On huggingface, choose a model from [text2image models](https://huggingface.co/models?pipeline_tag=text-to-image&sort=trending), then create a new folder named after it in `comfyui/models/diffusion_models`, then download in it the subfolders `unet` (if not available use `transformer`) and `scheduler`.
  - Hint: sometimes in the `unet` or `transformer` folder there are more model files and not all are required. If you have `model.fp16` and `model`, I suggest you to use the fp16 variant; if you have `model-001-of-002`, `model-002-of-002`, `model`, choose model (instead of the fragmented version).
- **Controlnet model**: download `config.json` and the safetensors `model`.

#### Unet and Controlnet Models Loader using ComfYUI nodes canceled
I can load them but then they don't work in the inference code, since comfyui load diffusers models in a different format ([reddit post](https://www.reddit.com/r/comfyui/comments/17fvb49/comment/k6cz9yv/?utm_source=share&utm_medium=web3x&utm_name=web3xcss&utm_term=1&utm_content=share_button)).

## Credits
diffusers-image-outpaint by [fffiloni](https://huggingface.co/spaces/fffiloni/diffusers-image-outpaint/tree/main)
