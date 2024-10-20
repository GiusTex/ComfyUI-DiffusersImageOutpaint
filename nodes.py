import torch
import os
from PIL import Image
from .utils import get_first_folder_list, tensor2pil, pil2tensor, diffuserOutpaintSamples, get_device_by_name, get_dtype_by_name, clearVram


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
        
        # Raise an error.
        if source.width == width and source.height == height:
            raise ValueError(f'Input image size is the same as target size, resize input image or change target size.')
        
        # Initialize new_width and new_height
        new_width, new_height = source.width, source.height

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


class LoadDiffusersOutpaintModels:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": (get_first_folder_list("diffusion_models"), {"default": "RealVisXL_V5.0_Lightning", "tooltip": "The diffuser model used for denoising the input latent. (Put model files in a folder, in diffusion_models folder)."}),
                "controlnet_model": (get_first_folder_list("diffusion_models"), {"default": "controlnet-union-sdxl-1.0", "tooltip": "The controlnet model used for denoising the input latent. (Put model files in a folder, in diffusion_models folder)."}),
                "device": (["auto", "cuda", "cpu", "mps", "xpu", "meta"],{"default": "auto", "tooltip": "Device for inference, default is auto checked by comfyui"}), 
                "dtype": (["auto","fp16","bf16","fp32", "fp8_e4m3fn", "fp8_e4m3fnuz", "fp8_e5m2", "fp8_e5m2fnuz"],{"default":"auto", "tooltip": "Model precision for inference, default is auto checked by comfyui"}),
                "sequential_cpu_offload": ("BOOLEAN", {"default": False, "tooltip": "Inference by default needs around 8gb vram, if this option is on it will move controlnet and unet back and forth between cpu and vram, to have only one model loaded at a time (around 6 gb vram used), useful for gpus under 8gb but will impact inference speed."}),
            },
        }

    RETURN_TYPES = ("PIPE",)
    RETURN_NAMES = ("diffusers_outpaint_pipe",)
    FUNCTION = "load"
    CATEGORY = "DiffusersOutpaint"
   
    def load(self, model, controlnet_model, device, dtype, sequential_cpu_offload):
        # Go 2 folders back
        comfy_dir = os.path.dirname(os.path.dirname(my_dir))
        
        model_path = f"{comfy_dir}/models/diffusion_models/{model}"
        controlnet_path = f"{comfy_dir}/models/diffusion_models/{controlnet_model}"
        
        device = get_device_by_name(device)
        dtype = get_dtype_by_name(dtype)
        
        diffusers_outpaint_pipe = {
            "model_path": model_path,
            "controlnet_model": controlnet_model,
            "controlnet_path": controlnet_path,
            "device": device,
            "dtype": dtype,
            "keep_model_device": sequential_cpu_offload,
        }
        
        return (diffusers_outpaint_pipe,)


class EncodeDiffusersOutpaintPrompt:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "diffusers_outpaint_pipe": ("PIPE", {"tooltip": "Load the diffusers outpaint models."}),
                "text": ("STRING", {"multiline": True, "dynamicPrompts": True, "tooltip": "The text to be encoded."}), 
                "clip": ("CLIP", {"tooltip": "The CLIP model used for encoding the text."})
            }
        }
    RETURN_TYPES = ("PIPE","CONDITIONING",)
    RETURN_NAMES = ("diffusers_outpaint_pipe","diffusers_conditioning",)
    OUTPUT_TOOLTIPS = ("A conditioning containing the embedded text used to guide the diffusion model.",)
    FUNCTION = "encode"
    CATEGORY = "DiffusersOutpaint"
    DESCRIPTION = "Encodes a text prompt using a CLIP model into an embedding that can be used to guide the diffusion model towards generating specific images."

    def encode(self, diffusers_outpaint_pipe, text, clip):
        dtype = diffusers_outpaint_pipe["dtype"]
        device = diffusers_outpaint_pipe["device"]
        
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

        return (diffusers_outpaint_pipe,diffusers_conditioning,)
    

class DiffusersImageOutpaint:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "diffusers_outpaint_pipe": ("PIPE", {"tooltip": "Load the diffusers outpaint models."}),
                "positive": ("CONDITIONING", {"tooltip": "The prompt describing what you want."}),
                "negative": ("CONDITIONING", {"tooltip": "The prompt describing what you don't want."}),
                "diffuser_outpaint_cnet_image": ("IMAGE", {"tooltip": "The image to outpaint."}),
                "guidance_scale": ("FLOAT", {"default": 1.50, "min": 1.01, "max": 10, "step": 0.01, "tooltip": "The Classifier-Free Guidance scale balances creativity and adherence to the prompt. Higher values result in images more closely matching the prompt, however too high values will negatively impact quality."}),
                "controlnet_strength": ("FLOAT", {"default": 1.00, "min": 0.00, "max": 10, "step": 0.01}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "tooltip": "Fake seed, workaround used to keep generating different outpaints. Set to -1 to generate different images, or a fixed number to stop that."}),
                "steps": ("INT", {"default": 8, "min": 4, "max": 20, "tooltip": "The number of steps used in the denoising process."}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"
    CATEGORY = "DiffusersOutpaint"

    def sample(self, diffusers_outpaint_pipe, positive, negative, diffuser_outpaint_cnet_image, guidance_scale, controlnet_strength, seed, steps):
        
        cnet_image = diffuser_outpaint_cnet_image
        cnet_image=tensor2pil(cnet_image)
        cnet_image=cnet_image.convert('RGB')
        
        model_path = diffusers_outpaint_pipe["model_path"]
        controlnet_model = diffusers_outpaint_pipe["controlnet_model"]
        controlnet_path = diffusers_outpaint_pipe["controlnet_path"]
        dtype = diffusers_outpaint_pipe["dtype"]
        device = diffusers_outpaint_pipe["device"]
        keep_model_device = diffusers_outpaint_pipe["keep_model_device"]
                
        prompt_embeds = positive["prompt_embeds"]
        pooled_prompt_embeds = positive["pooled_prompt_embeds"]
        negative_prompt_embeds = negative["prompt_embeds"]
        negative_pooled_prompt_embeds = negative["pooled_prompt_embeds"]

        last_rgb_latent = diffuserOutpaintSamples(model_path, controlnet_model, diffuser_outpaint_cnet_image, dtype, controlnet_path, 
                                                  prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds, 
                                                  device, steps, controlnet_strength, guidance_scale, 
                                                  keep_model_device)
        
        del prompt_embeds, pooled_prompt_embeds, negative_prompt_embeds, negative_pooled_prompt_embeds
        clearVram(device)

        return ({"samples":last_rgb_latent},)
