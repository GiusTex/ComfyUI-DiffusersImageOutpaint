import torch
import os
from PIL import Image, ImageDraw
from .utils import get_config_folder_list, tensor2pil, pil2tensor, diffuserOutpaintSamples, get_device_by_name, get_dtype_by_name, clearVram, test_scheduler_scale_model_input
import folder_paths
from .unet_2d_condition import UNet2DConditionModel
from diffusers import TCDScheduler
from .controlnet_union import ControlNetModel_Union
from diffusers.models.model_loading_utils import load_state_dict
from safetensors.torch import load_file

import logging


# Get the absolute path of various directories
my_dir = os.path.dirname(os.path.abspath(__file__))

def update_folder_names_and_paths(key, targets=[]):
    # check for existing key
    base = folder_paths.folder_names_and_paths.get(key, ([], {}))
    base = base[0] if isinstance(base[0], (list, set, tuple)) else []
    # find base key & add w/ fallback, sanity check + warning
    target = next((x for x in targets if x in folder_paths.folder_names_and_paths), targets[0])
    orig, _ = folder_paths.folder_names_and_paths.get(target, ([], {}))
    folder_paths.folder_names_and_paths[key] = (orig or base, {".gguf"})
    if base and base != orig:
        logging.warning(f"Unknown file list already present on key {key}: {base}")

def can_expand(source_width, source_height, target_width, target_height, alignment):
    """Checks if the image can be expanded based on the alignment."""
    if alignment in ("Left", "Right") and source_width >= target_width:
        return False
    if alignment in ("Top", "Bottom") and source_height >= target_height:
        return False
    return True


class PadImageForDiffusersOutpaint:
    _alignment_options = ["Middle", "Left", "Right", "Top", "Bottom"]
    _resize_option = ["Full", "50%", "33%", "25%", "Custom"]
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "width": ("INT", {"default": 720, "tooltip": "The width used for the image."}),
                "height": ("INT", {"default": 1280, "tooltip": "The height used for the image."}),
                "alignment": (s._alignment_options, {"tooltip": "Where the original image should be in the outpainted one"}),
                "resize_image": (s._resize_option, {"tooltip": "Resize input image"}),
                "custom_resize_image_percentage": ("INT", {"min": 1, "default": 50, "max": 100, "step": 1, "tooltip": "Custom resize (%)"}),
                "mask_overlap_percentage": ("INT", {"min": 1, "default": 10, "max": 50, "step": 1, "tooltip": "Mask overlap (%)"}),
                "overlap_left": ("BOOLEAN", {"default": True}),
                "overlap_right": ("BOOLEAN", {"default": True}),
                "overlap_top": ("BOOLEAN", {"default": True}),
                "overlap_bottom": ("BOOLEAN", {"default": True}),
            },
        }
    
    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE")
    RETURN_NAMES = ("IMAGE", "MASK", "diffuser_outpaint_cnet_image")
    FUNCTION = "prepare_image_and_mask"
    CATEGORY = "DiffusersOutpaint"
    
    def prepare_image_and_mask(self, image, width, height, mask_overlap_percentage, resize_image, custom_resize_image_percentage, overlap_left, overlap_right, overlap_top, overlap_bottom, alignment="Middle"):
        im=tensor2pil(image)
        source=im.convert('RGB')

        target_size = (width, height)
        
        # Calculate the scaling factor to fit the image within the target size
        scale_factor = min(target_size[0] / source.width, target_size[1] / source.height)
        new_width = int(source.width * scale_factor)
        new_height = int(source.height * scale_factor)

        # Resize the source image to fit within target size
        source = source.resize((new_width, new_height), Image.LANCZOS)

        # Initialize new_width and new_height
        new_width, new_height = source.width, source.height

        # Apply resize option using percentages
        if resize_image == "Full":
            resize_percentage = 100
        elif resize_image == "50%":
            resize_percentage = 50
        elif resize_image == "33%":
            resize_percentage = 33
        elif resize_image == "25%":
            resize_percentage = 25
        else:  # Custom
            resize_percentage = custom_resize_image_percentage
        
        # Calculate new dimensions based on percentage
        resize_factor = resize_percentage / 100
        new_width = int(source.width * resize_factor)
        new_height = int(source.height * resize_factor)
        
        # Ensure minimum size of 64 pixels
        new_width = max(new_width, 64)
        new_height = max(new_height, 64)

        # Resize the image
        source = source.resize((new_width, new_height), Image.LANCZOS)

        # Calculate the overlap in pixels based on the percentage
        overlap_x = int(new_width * (mask_overlap_percentage / 100))
        overlap_y = int(new_height * (mask_overlap_percentage / 100))
        
        # Ensure minimum overlap of 1 pixel
        overlap_x = max(overlap_x, 1)
        overlap_y = max(overlap_y, 1)
        
        # Calculate margins based on alignment
        if alignment == "Middle":
            margin_x = (target_size[0] - source.width) // 2
            margin_y = (target_size[1] - source.height) // 2
        elif alignment == "Left":
            margin_x = 0
            margin_y = (target_size[1] - source.height) // 2
        elif alignment == "Right":
            margin_x = target_size[0] - source.width
            margin_y = (target_size[1] - source.height) // 2
        elif alignment == "Top":
            margin_x = (target_size[0] - source.width) // 2
            margin_y = 0
        elif alignment == "Bottom":
            margin_x = (target_size[0] - source.width) // 2
            margin_y = target_size[1] - source.height

        # Adjust margins to eliminate gaps
        margin_x = max(0, min(margin_x, target_size[0] - new_width))
        margin_y = max(0, min(margin_y, target_size[1] - new_height))

        # Create a new background image and paste the resized source image
        background = Image.new('RGB', target_size, (255, 255, 255))
        background.paste(source, (margin_x, margin_y))

        image=pil2tensor(background)
        #----------------------------------------------------
        # Create the mask
        d1, d2, d3, d4 = image.size()
        left, top, bottom, right = 0, 0, 0, 0
        # Image
        new_image = torch.ones(
            (d1, d2 + top + bottom, d3 + left + right, d4),
            dtype=torch.float32,
        ) * 0.5
        new_image[:, top:top + d2, left:left + d3, :] = image

        im=tensor2pil(new_image)
        pil_new_image=im.convert('RGB')
        #----------------------------------------------------

        # Create the mask
        mask = Image.new('L', target_size, 255)
        mask_draw = ImageDraw.Draw(mask)
        #----------------------------------------------------
        # Calculate overlap areas
        white_gaps_patch = 2

        left_overlap = margin_x + overlap_x if overlap_left else margin_x + white_gaps_patch
        right_overlap = margin_x + new_width - overlap_x if overlap_right else margin_x + new_width - white_gaps_patch
        top_overlap = margin_y + overlap_y if overlap_top else margin_y + white_gaps_patch
        bottom_overlap = margin_y + new_height - overlap_y if overlap_bottom else margin_y + new_height - white_gaps_patch
        #----------------------------------------------------
        # Mask coordinates
        if alignment == "Left":
            left_overlap = margin_x + overlap_x if overlap_left else margin_x
        elif alignment == "Right":
            right_overlap = margin_x + new_width - overlap_x if overlap_right else margin_x + new_width
        elif alignment == "Top":
            top_overlap = margin_y + overlap_y if overlap_top else margin_y
        elif alignment == "Bottom":
            bottom_overlap = margin_y + new_height - overlap_y if overlap_bottom else margin_y + new_height
    
        # Draw the mask
        mask_draw.rectangle([
            (left_overlap, top_overlap),
            (right_overlap, bottom_overlap)
        ], fill=0)

        tensor_mask=pil2tensor(mask)
        #----------------------------------------------------
        if not can_expand(background.width, background.height, width, height, alignment):
            alignment = "Middle"
        
        cnet_image = pil_new_image.copy() # copy background as cnet_image
        cnet_image.paste(0, (0, 0), mask) # paste mask over cnet_image, cropping it a bit
        
        tensor_cnet_image=pil2tensor(cnet_image)

        return (new_image, tensor_mask, tensor_cnet_image,)


class LoadDiffuserModel:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "unet_name": (folder_paths.get_filename_list("diffusion_models"), {"tooltip": "The name of the unet (model) to load."}),
                "device": (["auto", "cuda", "cpu", "mps", "xpu", "meta"],{"default": "auto", "tooltip": "Device for inference, default is auto checked by comfyui"}), 
                "dtype": (["auto","fp16","bf16","fp32", "fp8_e4m3fn", "fp8_e4m3fnuz", "fp8_e5m2", "fp8_e5m2fnuz"],{"default":"auto", "tooltip": "Model precision for inference, default is auto checked by comfyui"}),
                "model_type": (get_config_folder_list("configs"), {"default": "sdxl", "tooltip": "The json configs used for the unet. (Put unet config in \"configs/your model type/unet\", and scheduler config in \"configs/your model type/scheduler\")."}),
            },
        }

    RETURN_TYPES = ("MODEL", "SCHEDULER")
    RETURN_NAMES = ("model", "scheduler configs")
    FUNCTION = "load"
    CATEGORY = "DiffusersOutpaint"
   
    def load(self, unet_name, device, dtype, model_type):

        # Go 2 folders back
        comfy_dir = os.path.dirname(os.path.dirname(my_dir))
        unet_path = f"D:/models/diffusion_models/{unet_name}"

        device = get_device_by_name(device)
        dtype = get_dtype_by_name(dtype)
        
        if model_type == "sdxl":
            print("Loading sdxl unet...")

            unet = UNet2DConditionModel.from_config(f"{comfy_dir}/custom_nodes/ComfyUI-DiffusersImageOutpaint/configs", subfolder=f"{model_type}/unet").to(device, dtype)
            unet.load_state_dict(load_file(unet_path))
        
            scheduler = TCDScheduler.from_config(f"{comfy_dir}/custom_nodes/ComfyUI-DiffusersImageOutpaint/configs", subfolder=f"{model_type}/scheduler")
        
        scale_model_input_method = test_scheduler_scale_model_input(comfy_dir, model_type)
        
        scheduler_configs = {
            "scheduler": scheduler,
            "scale_model_input_method": scale_model_input_method,
        }

        return (unet, scheduler_configs,)


class LoadDiffuserControlnet:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "controlnet_model": (folder_paths.get_filename_list("controlnet"), {"tooltip": "The controlnet model used for denoising the input latent."}),
                "device": (["auto", "cuda", "cpu", "mps", "xpu", "meta"],{"default": "auto", "tooltip": "Device for inference, default is auto checked by comfyui"}), 
                "dtype": (["auto","fp16","bf16","fp32", "fp8_e4m3fn", "fp8_e4m3fnuz", "fp8_e5m2", "fp8_e5m2fnuz"],{"default":"auto", "tooltip": "Model precision for inference, default is auto checked by comfyui"}),
                "controlnet_type": (get_config_folder_list("configs"), {"default": "controlnet-sdxl-promax", "tooltip": "The json configs used for controlnet. (Put config(s) in \"configs/your controlnet type\")."}),
            },
        }

    RETURN_TYPES = ("CONTROL_NET",)
    FUNCTION = "load"
    CATEGORY = "DiffusersOutpaint"
   
    def load(self, controlnet_model, device, dtype, controlnet_type):

        # Go 2 folders back
        comfy_dir = os.path.dirname(os.path.dirname(my_dir))
        #controlnet_path = f"D:/models/controlnet/{controlnet_model}"  <-----------FIX

        device = get_device_by_name(device)
        dtype = get_dtype_by_name(dtype)
        
        if controlnet_type == "controlnet-sdxl-promax":
            print("Loading controlnet-sdxl-promax...")
            controlnet_model = ControlNetModel_Union.from_config(f"{comfy_dir}/custom_nodes/ComfyUI-DiffusersImageOutpaint/configs/{controlnet_type}/config_promax.json")
        
            state_dict = load_state_dict(load_file(controlnet_path))

            model, _, _, _, _ = ControlNetModel_Union._load_pretrained_model(
                controlnet_model, state_dict, controlnet_path, controlnet_path
            )

            controlnet_model.to(device, dtype)
    
            del model, state_dict, controlnet_path
        
            clearVram(device)
            
        return (controlnet_model,)


class EncodeDiffusersOutpaintPrompt:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "device": (["auto", "cuda", "cpu", "mps", "xpu", "meta"],{"default": "auto", "tooltip": "Device for inference, default is auto checked by comfyui"}), 
                "dtype": (["auto","fp16","bf16","fp32", "fp8_e4m3fn", "fp8_e4m3fnuz", "fp8_e5m2", "fp8_e5m2fnuz"],{"default":"auto", "tooltip": "Model precision for inference, default is auto checked by comfyui"}),
                "text": ("STRING", {"multiline": True, "dynamicPrompts": True, "tooltip": "The text to be encoded."}), 
                "clip": ("CLIP", {"tooltip": "The CLIP model used for encoding the text."})
            }
        }
    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("diffusers_conditioning",)
    OUTPUT_TOOLTIPS = ("A conditioning containing the embedded text used to guide the diffusion model.",)
    FUNCTION = "encode"
    CATEGORY = "DiffusersOutpaint"
    DESCRIPTION = "Encodes a text prompt using a CLIP model into an embedding that can be used to guide the diffusion model towards generating specific images."

    def encode(self, device, dtype, text, clip):
        device = get_device_by_name(device)
        dtype = get_dtype_by_name(dtype)
        
        text = f"{text}, high quality, 4k"
        tokens = clip.tokenize(text)
        output = clip.encode_from_tokens(tokens, return_pooled=True, return_dict=True)
        prompt_embeds = output.pop("cond")

        prompt_embeds = prompt_embeds.to(device, dtype=dtype)
        pooled_prompt_embeds = output["pooled_output"].to(device, dtype=dtype)
    
        bs_embed, seq_len, _ = prompt_embeds.shape

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, 1, 1)
        prompt_embeds = prompt_embeds.view(bs_embed * 1, seq_len, -1)

        pooled_prompt_embeds = pooled_prompt_embeds.repeat(1, 1).view(bs_embed * 1, -1)
        
        diffusers_conditioning = {
            "prompt_embeds": prompt_embeds,
            "pooled_prompt_embeds": pooled_prompt_embeds,
        }

        return (diffusers_conditioning,)
    

class DiffusersImageOutpaint:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "The model used for denoising the input latent."}),
                "scheduler_configs": ("SCHEDULER",),
                "control_net": ("CONTROL_NET",),
                "positive": ("CONDITIONING", {"tooltip": "The prompt describing what you want."}),
                "negative": ("CONDITIONING", {"tooltip": "The prompt describing what you don't want."}),
                "diffuser_outpaint_cnet_image": ("IMAGE", {"tooltip": "The image to outpaint."}),
                "guidance_scale": ("FLOAT", {"default": 1.50, "min": 1.01, "max": 10, "step": 0.01, "tooltip": "The Classifier-Free Guidance scale balances creativity and adherence to the prompt. Higher values result in images more closely matching the prompt, however too high values will negatively impact quality."}),
                "controlnet_strength": ("FLOAT", {"default": 1.00, "min": 0.00, "max": 10, "step": 0.01}),
                "steps": ("INT", {"default": 8, "min": 4, "max": 20, "tooltip": "The number of steps used in the denoising process."}),
                "device": (["auto", "cuda", "cpu", "mps", "xpu", "meta"],{"default": "auto", "tooltip": "Device for inference, default is auto checked by comfyui"}), 
                "dtype": (["auto","fp16","bf16","fp32", "fp8_e4m3fn", "fp8_e4m3fnuz", "fp8_e5m2", "fp8_e5m2fnuz"],{"default":"auto", "tooltip": "Model precision for inference, default is auto checked by comfyui"}),
                "sequential_cpu_offload": ("BOOLEAN", {"default": False, "tooltip": "Inference by default needs around 8gb vram, if this option is on it will move controlnet and unet back and forth between cpu and vram, to have only one model loaded at a time (around 6 gb vram used), useful for gpus under 8gb but will impact inference speed."}),
            },
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"
    CATEGORY = "DiffusersOutpaint"
    
    def sample(self, device, dtype, sequential_cpu_offload, scheduler_configs, model, control_net, positive, negative, diffuser_outpaint_cnet_image, guidance_scale, controlnet_strength, steps):
        cnet_image = diffuser_outpaint_cnet_image
        cnet_image=tensor2pil(cnet_image)
        cnet_image=cnet_image.convert('RGB')
        
        keep_model_device = sequential_cpu_offload

        scheduler = scheduler_configs["scheduler"]
        scale_model_input_method = scheduler_configs["scale_model_input_method"]
        
        last_rgb_latent = diffuserOutpaintSamples(device, dtype, keep_model_device, scheduler, scale_model_input_method, model, control_net, positive, negative, 
                                                      cnet_image, controlnet_strength, guidance_scale, steps)
        
        return ({"samples":last_rgb_latent},)
