import torch
import gc
import os
import numpy as np
from PIL import Image

from folder_paths import map_legacy, folder_names_and_paths
from .controlnet_union import ControlNetModel_Union
from .pipeline_fill_sd_xl import StableDiffusionXLFillPipeline
from diffusers import AutoencoderKL, TCDScheduler
from diffusers.models.model_loading_utils import load_state_dict


# Get the absolute path of various directories
my_dir = os.path.dirname(os.path.abspath(__file__))

class PadImageForDiffusersOutpaint:
    _alignment_options = ["Middle", "Left", "Right", "Top", "Bottom"]
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "width": ("INT", {"default": 720, "min": 320, "max": 1536, "tooltip": "The width used for the image."}),
                "height": ("INT", {"default": 1280, "min": 320, "max": 1536, "tooltip": "The height used for the image."}),
                "alignment": (s._alignment_options, {"tooltip": "Where the original image should be in the outpainted one"}),
            },
        }
    
    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE")
    RETURN_NAMES = ("IMAGE", "MASK", "diffuser_outpaint_cnet_image")
    FUNCTION = "expand_image"
    CATEGORY = "DiffusersOutpaint"
    
    def expand_image(self, image, width, height, alignment="Middle"):
        
        # Resize Image
        def can_expand(source_width, source_height, target_width, target_height, alignment):
            """Checks if the image can be expanded based on the alignment."""
            if alignment in ("Left", "Right") and source_width >= target_width:
                return False
            if alignment in ("Top", "Bottom") and source_height >= target_height:
                return False
            return True
        
        im=tensor2pil(image)
        source=im.convert('RGB')
        target_size = (width, height)

        # Upscale if source is smaller than target in both dimensions
        if source.width < target_size[0] and source.height < target_size[1]:
            scale_factor = min(target_size[0] / source.width, target_size[1] / source.height)
            new_width = int(source.width * scale_factor)
            new_height = int(source.height * scale_factor)
            source = source.resize((new_width, new_height), Image.LANCZOS)

        if source.width > target_size[0] or source.height > target_size[1]:
            scale_factor = min(target_size[0] / source.width, target_size[1] / source.height)
            new_width = int(source.width * scale_factor)
            new_height = int(source.height * scale_factor)
            source = source.resize((new_width, new_height), Image.LANCZOS)

        if source.width == target_size[0] and source.height != target_size[1]:
            scale_factor = min(target_size[0] / source.width, target_size[1] / source.height)
            new_width = target_size[0]
            new_height = int(source.height * scale_factor)
            source = source.resize((new_width, new_height), Image.LANCZOS)

        if not can_expand(source.width, source.height, target_size[0], target_size[1], alignment):
            alignment = "Middle"
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

        background = Image.new('RGB', target_size, (255, 255, 255))
        background.paste(source, (margin_x, margin_y))

        image=pil2tensor(background)
        #----------------------------------------------------
        d1, d2, d3, d4 = image.size()
        left, top, bottom, right = 0, 0, 0, 0
        # Image
        new_image = torch.ones(
            (d1, d2 + top + bottom, d3 + left + right, d4),
            dtype=torch.float32,
        ) * 0.5
        new_image[:, top:top + d2, left:left + d3, :] = image
        #----------------------------------------------------
        # Mask coordinates
        if alignment == "Middle":
            margin_x = (width - new_width) // 2
            margin_y = (height - new_height) // 2
        elif alignment == "Left":
            margin_x = 0
            margin_y = (height - new_height) // 2
        elif alignment == "Right":
            margin_x = width - new_width
            margin_y = (height - new_height) // 2
        elif alignment == "Top":
            margin_x = (width - new_width) // 2
            margin_y = 0
        elif alignment == "Bottom":
            margin_x = (width - new_width) // 2
            margin_y = height - new_height
        
        # Create mask as big as new img
        mask = torch.ones(
            (height, width),
            dtype=torch.float32,
        )
        # Create hole in mask
        t = torch.zeros(
            (new_height, new_width),
            dtype=torch.float32
        )
        # Create holed mask
        mask[margin_y:margin_y + new_height, 
             margin_x:margin_x + new_width
        ] = t
        #----------------------------------------------------
        # Prepare "cn_image" for diffusers outpaint
        im=tensor2pil(new_image)
        pil_new_image=im.convert('RGB')
        
        pil_mask=tensor2pil(mask)
        
        cnet_image = pil_new_image.copy() # copy background as cnet_image
        cnet_image.paste(0, (0, 0), pil_mask) # paste mask over cnet_image, cropping it a bit
        
        tensor_cnet_image=pil2tensor(cnet_image)

        return (new_image, mask, tensor_cnet_image,)


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


class LoadDiffusersOutpaintModels:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": (get_first_folder_list("diffusion_models"), {"default": "RealVisXL_V5.0_Lightning", "tooltip": "The diffuser model used for denoising the input latent. (Put model files in a folder, in diffusion_models folder)."}),
                "vae": (get_first_folder_list("diffusion_models"), {"default": "sdxl-vae-fp16-fix", "tooltip": "The vae model used for denoising the input latent. (Put model files in a folder, in diffusion_models folder)."}),
                "controlnet_model": (get_first_folder_list("diffusion_models"), {"default": "controlnet-union-sdxl-1.0", "tooltip": "The controlnet model used for denoising the input latent. (Put model files in a folder, in diffusion_models folder)."}),
            },
            "optional": {
                "enable_vae_slicing": ("BOOLEAN", {"default": True, "tooltip": "VAE will split the input tensor in slices to compute decoding in several steps. This is useful to save some memory and allow larger batch sizes."}),
                "enable_vae_tiling": ("BOOLEAN", {"default": False, "tooltip": "Drastically reduces memory use but may introduce seams"}),
            },
        }

    RETURN_TYPES = ("PIPE",)
    RETURN_NAMES = ("diffusers_outpaint_pipe",)
    FUNCTION = "load"
    CATEGORY = "DiffusersOutpaint"
   
    def load(self, model, vae, controlnet_model, enable_vae_slicing, enable_vae_tiling):
        # Go 2 folders back
        comfy_dir = os.path.dirname(os.path.dirname(my_dir))
        
        model_path = f"{comfy_dir}/models/diffusion_models/{model}"
        vae_path = f"{comfy_dir}/models/diffusion_models/{vae}"
        controlnet_path = f"{comfy_dir}/models/diffusion_models/{controlnet_model}"
        #-----------------------------------------------------------------------

        # Set up Controlnet-Union-Promax-SDXL model
        config_file = f"{controlnet_path}/config_promax.json"
        config = ControlNetModel_Union.load_config(config_file)
        controlnet_model = ControlNetModel_Union.from_config(config)

        model_file = f"{controlnet_path}/diffusion_pytorch_model_promax.safetensors"
        state_dict = load_state_dict(model_file)

        model, _, _, _, _ = ControlNetModel_Union._load_pretrained_model(
            controlnet_model, state_dict, model_file, f"{controlnet_path}"
        )
        model.to(device="cuda", dtype=torch.float16)
        #-----------------------------------------------------------------------

        # Set up VAE
        vae = AutoencoderKL.from_pretrained(f"{vae_path}", torch_dtype=torch.float16).to("cuda")
        
        if enable_vae_slicing:
            vae.enable_slicing()
        else:
            vae.disable_slicing()
        
        if enable_vae_tiling:
            vae.enable_tiling()
        else:
            vae.disable_tiling()
        #-----------------------------------------------------------------------

        # Load Controlnet + Vae into RealVisXL model
        pipe = StableDiffusionXLFillPipeline.from_pretrained(
            f"{model_path}",
            torch_dtype=torch.float16,
            vae=vae,
            controlnet=controlnet_model,
            variant="fp16"
        )
        
        pipe.scheduler = TCDScheduler.from_config(pipe.scheduler.config)
        pipe.enable_model_cpu_offload()

        del model, state_dict, model_file
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

        diffusers_outpaint_pipe = {
            "pipe": pipe,
            "vae": vae,
            "controlnet_model": controlnet_model
        }

        return (diffusers_outpaint_pipe,)


# Tensor to PIL (grabbed from WAS Suite)
def tensor2pil(image: torch.Tensor) -> Image.Image:
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

# Convert PIL to Tensor (grabbed from WAS Suite)
def pil2tensor(image: Image.Image) -> torch.Tensor:
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

class EncodeDiffusersOutpaintPrompt:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "diffusers_outpaint_pipe": ("PIPE", {"tooltip": "Load the diffusers outpaint models."}),
                "extra_prompt": ("STRING", {"default": "", "tooltip": "The extra prompt to append, describing attributes etc. you want to include in the image. Default: \"(extra_prompt), high quality, 4k\""}),
            }
        }

    RETURN_TYPES = ("PIPE","CONDITIONING",)
    RETURN_NAMES = ("diffusers_outpaint_pipe","diffusers_outpaint_conditioning",)
    FUNCTION = "sample"
    CATEGORY = "DiffusersOutpaint"

    def sample(self, diffusers_outpaint_pipe, extra_prompt=None):
        pipe = diffusers_outpaint_pipe["pipe"]
        
        final_prompt = f"{extra_prompt}, high quality, 4k"
        
        (prompt_embeds,
         negative_prompt_embeds,
         pooled_prompt_embeds,
         negative_pooled_prompt_embeds,
        ) = pipe.encode_prompt(final_prompt)
        
        diffusers_outpaint_conditioning = {
            "prompt_embeds": prompt_embeds,
            "negative_prompt_embeds": negative_prompt_embeds,
            "pooled_prompt_embeds": pooled_prompt_embeds,
            "negative_pooled_prompt_embeds": negative_pooled_prompt_embeds
        }

        return (diffusers_outpaint_pipe,diffusers_outpaint_conditioning,)


class DiffusersImageOutpaint:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "diffusers_outpaint_pipe": ("PIPE", {"tooltip": "Load the diffusers outpaint models."}),
                "diffusers_outpaint_conditioning": ("CONDITIONING", {"tooltip": "The prompt describing what you want."}),
                "diffuser_outpaint_cnet_image": ("IMAGE", {"tooltip": "The image to outpaint."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "tooltip": "Fake seed, workaround used to keep generating different outpaints. Set to -1 to generate different images, or a fixed number to stop that."}),
                "steps": ("INT", {"default": 8, "min": 4, "max": 20, "tooltip": "The number of steps used in the denoising process."}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "sample"
    CATEGORY = "DiffusersOutpaint"

    def sample(self, diffusers_outpaint_pipe, diffusers_outpaint_conditioning, diffuser_outpaint_cnet_image, seed, steps):
        # I have to save them here to cache them. The node doesn't load them again
        pipe = diffusers_outpaint_pipe["pipe"]
        vae = diffusers_outpaint_pipe["vae"]
        controlnet_model = diffusers_outpaint_pipe["controlnet_model"]
        prompt_embeds = diffusers_outpaint_conditioning["prompt_embeds"]
        negative_prompt_embeds = diffusers_outpaint_conditioning["negative_prompt_embeds"]
        pooled_prompt_embeds = diffusers_outpaint_conditioning["pooled_prompt_embeds"]
        negative_pooled_prompt_embeds = diffusers_outpaint_conditioning["negative_pooled_prompt_embeds"]

        cnet_image = diffuser_outpaint_cnet_image
        cnet_image=tensor2pil(cnet_image)
        cnet_image=cnet_image.convert('RGB')
        
        generated_images = list(pipe(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            image=cnet_image,
            num_inference_steps=steps
        ))
        
        del pipe, vae, controlnet_model, prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

        last_image = generated_images[-1] # Access the last image
        image = last_image.convert("RGB")
        output=pil2tensor(image)
        
        return (output,)
