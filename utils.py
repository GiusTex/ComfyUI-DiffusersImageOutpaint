import torch
import gc
import os
import numpy as np
import json
import comfy.model_management as mm
from PIL import Image

from folder_paths import map_legacy, folder_names_and_paths
from .controlnet_union import ControlNetModel_Union
from .pipeline_fill_sd_xl import encode_prompt, StableDiffusionXLFillPipeline
from diffusers import AutoencoderKL, TCDScheduler
from diffusers.models.model_loading_utils import load_state_dict
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer


def get_first_folder_list(folder_name: str) -> tuple[list[str], dict[str, float], float]:
    folder_name = map_legacy(folder_name)
    global folder_names_and_paths
    folders = folder_names_and_paths[folder_name]
    if folder_name == "unet":
        root_folder = folders[0][0]
    elif folder_name == "diffusion_models":
        root_folder = folders[0][1]
    visible_folders = [name for name in os.listdir(root_folder) if os.path.isdir(os.path.join(root_folder, name))]
    return visible_folders


# Tensor to PIL (grabbed from WAS Suite)
def tensor2pil(image: torch.Tensor) -> Image.Image:
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

# Convert PIL to Tensor (grabbed from WAS Suite)
def pil2tensor(image: Image.Image) -> torch.Tensor:
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)


def get_device_by_name(device):
    if device == 'auto':
        device = mm.get_torch_device()
    return device


def get_dtype_by_name(dtype):
    if dtype == 'auto':
        if mm.should_use_fp16():
            dtype = torch.float16
        elif mm.should_use_bf16():
            dtype = torch.bfloat16
        else:
            dtype = torch.float32
    elif dtype== "fp16":
        dtype = torch.float16
    elif dtype == "bf16":
        dtype = torch.bfloat16
    elif dtype == "fp32":
        dtype = torch.float32
    elif dtype == "fp8_e4m3fn":
        dtype = torch.float8_e4m3fn
    elif dtype == "fp8_e4m3fnuz":
        dtype = torch.float8_e4m3fnuz
    elif dtype == "fp8_e5m2":
        dtype = torch.float8_e5m2
    elif dtype == "fp8_e5m2fnuz":
        dtype = torch.float8_e5m2fnuz

    return dtype


def loadDiffModels1(model_path, dtype, device):
    tokenizer = CLIPTokenizer.from_pretrained(model_path, subfolder="tokenizer", use_fast=False)
    tokenizer_2 = CLIPTokenizer.from_pretrained(model_path, subfolder="tokenizer_2", use_fast=False)
    text_encoder = CLIPTextModel.from_pretrained(model_path, subfolder="text_encoder", torch_dtype=dtype).requires_grad_(False).to(device)
    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(model_path, subfolder="text_encoder_2", torch_dtype=dtype).requires_grad_(False).to(device)
    
    return tokenizer, tokenizer_2, text_encoder, text_encoder_2


def clearVram(device):
    gc.collect()

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    elif device.type == "mps":
        torch.mps.empty_cache()
        torch.mps.ipc_collect()
    elif device.type == "xla":
        torch.xla.empty_cache()
        torch.xla.ipc_collect()
    elif device.type == "xpu":
        torch.xpu.empty_cache()
        torch.xpu.ipc_collect()
    elif device.type == "meta":
        torch.meta.empty_cache()
        torch.meta.ipc_collect()
    else: # for CPU
        torch.ipc_collect()


def encodeDiffOutpaintPrompt(model_path, dtype, final_prompt, device):    
    tokenizer, tokenizer_2, text_encoder, text_encoder_2 = loadDiffModels1(model_path, dtype, device)

    (prompt_embeds,
     negative_prompt_embeds,
     pooled_prompt_embeds,
     negative_pooled_prompt_embeds,
    ) = encode_prompt(final_prompt, tokenizer, tokenizer_2, text_encoder, text_encoder_2, device, True)
    
    del tokenizer, tokenizer_2, text_encoder, text_encoder_2
    
    clearVram(device)

    return prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds


def loadControlnetModel(device, dtype, controlnet_path):
    config_file = f"{controlnet_path}/config_promax.json"
    config = ControlNetModel_Union.load_config(config_file)
    controlnet_model = ControlNetModel_Union.from_config(config)
    
    model_file = f"{controlnet_path}/diffusion_pytorch_model_promax.safetensors"
    state_dict = load_state_dict(model_file)

    model, _, _, _, _ = ControlNetModel_Union._load_pretrained_model(
        controlnet_model, state_dict, model_file, f"{controlnet_path}"
    )
    controlnet_model.to(device, dtype)

    del model, state_dict, model_file
    
    clearVram(device)

    return controlnet_model


def loadVaeModel(vae_path, device, dtype, enable_vae_slicing, enable_vae_tiling):
    vae = AutoencoderKL.from_pretrained(f"{vae_path}").to(device, dtype)
    if enable_vae_slicing:
        vae.enable_slicing()
    else:
        vae.disable_slicing()
    
    if enable_vae_tiling:
        vae.enable_tiling()
    else:
        vae.disable_tiling()
    return vae


def diffuserOutpaintSamples(model_path, controlnet_model, diffuser_outpaint_cnet_image, dtype, controlnet_path, 
                            prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds, 
                            device, steps, controlnet_strength, guidance_scale, 
                            keep_model_device):
    
    controlnet_model = loadControlnetModel(device, dtype, controlnet_path)
    
    with open(f"{model_path}/scheduler/scheduler_config.json", "r") as f:
        scheduler_config = json.load(f)
    scheduler = TCDScheduler.from_config(scheduler_config)
    
    pipe = StableDiffusionXLFillPipeline.from_pretrained(
        model_path,
        torch_dtype=dtype,
        variant="fp16",
        scheduler=scheduler,
    )
    if not keep_model_device:
        pipe.to(device)

    cnet_image = diffuser_outpaint_cnet_image
    cnet_image=tensor2pil(cnet_image)
    cnet_image=cnet_image.convert('RGB')
    
    rgb_latents = list(pipe(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            image=cnet_image,
            num_inference_steps=steps,
            controlnet_model=controlnet_model,
            controlnet_conditioning_scale=controlnet_strength,
            guidance_scale=guidance_scale,
            device=device,
            keep_model_device=keep_model_device,
        ))
    
    last_rgb_latent = rgb_latents[-1] # Access the last image
    
    del pipe, controlnet_model, scheduler, prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds
    
    clearVram(device)

    return last_rgb_latent
