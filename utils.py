import torch
import gc
import os
import numpy as np
import json
import comfy.model_management as mm
from PIL import Image

from folder_paths import map_legacy, folder_names_and_paths
from .pipeline_fill_sd_xl import StableDiffusionXLFillPipeline


def get_first_folder_list(folder_name: str) -> tuple[list[str], dict[str, float], float]:
    folder_name = map_legacy(folder_name)
    global folder_names_and_paths
    folders = folder_names_and_paths[folder_name]
    if folder_name == "unet":
        root_folder = folders[0][0]
    elif folder_name == "diffusion_models":
        root_folder = folders[0][1]
    elif folder_name == "controlnet":
        root_folder = folders[0][0]
    visible_folders = [name for name in os.listdir(root_folder) if os.path.isdir(os.path.join(root_folder, name))]
    return visible_folders

def get_config_folder_list(folder_name: str) -> tuple[list[str], dict[str, float], float]:
    my_dir = os.path.dirname(os.path.abspath(__file__))
    configs_dir = f"{my_dir}/{folder_name}"
    
    folders = [f for f in os.listdir(configs_dir) if os.path.isdir(os.path.join(configs_dir, f))]
    return folders

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

def clearVram(device):
    gc.collect()

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    elif device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "xla":
        torch.xla.empty_cache()
    elif device.type == "xpu":
        torch.xpu.empty_cache()
    elif device.type == "meta":
        torch.meta.empty_cache()


class TCDScheduler_Custom:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def scale_model_input(self, input, t):
        scale_factor = getattr(self, 'scale_factor', 1)
        return input * scale_factor
    
    def __repr__(self):
        attrs = {key: value for key, value in self.__dict__.items()}
        return f"TCDScheduler({attrs})"
    

def test_scheduler_scale_model_input(comfy_dir, model_type):
    scheduler_config_path = f"{comfy_dir}/custom_nodes/ComfyUI-DiffusersImageOutpaint/configs/{model_type}/scheduler/scheduler_config.json"

    with open(scheduler_config_path, 'r') as f:
        config = json.load(f)
    
    scheduler = TCDScheduler_Custom(**config)
    scale_model_input_method = scheduler.scale_model_input

    return scale_model_input_method


def diffuserOutpaintSamples(device, dtype, keep_model_device, scheduler, scale_model_input_method, model, control_net, positive, negative, 
                                                      cnet_image, controlnet_strength, guidance_scale, steps):
    
    prompt_embeds = positive["prompt_embeds"]
    pooled_prompt_embeds = positive["pooled_prompt_embeds"]
    negative_prompt_embeds = negative["prompt_embeds"]
    negative_pooled_prompt_embeds = negative["pooled_prompt_embeds"]
    controlnet_model = control_net
    
    device = get_device_by_name(device)
    dtype = get_dtype_by_name(dtype)
    
    timesteps = None
    unet = model
    
    pipe = StableDiffusionXLFillPipeline()

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
            dtype=dtype,
            unet=unet,
            timesteps=timesteps,
            scale_model_input_method=scale_model_input_method,
            keep_model_device=keep_model_device,
            scheduler=scheduler,
    ))
    
    last_rgb_latent = rgb_latents[-1] # Access the last image
    
    del pipe, unet, controlnet_model, scheduler, prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds
        
    clearVram(device)

    return last_rgb_latent
