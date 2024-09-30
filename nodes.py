import torch
import gc
import os
import numpy as np
from PIL import Image
from comfy import model_management
from folder_paths import map_legacy, folder_names_and_paths, models_dir, get_filename_list, get_full_path_or_raise
# Get the absolute path of various directories
my_dir = os.path.dirname(os.path.abspath(__file__))

class PadImageForDiffusersOutpaint:
    _alignment_options = ["Middle", "Left", "Right", "Top", "Bottom"]
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "width": ("INT", {"default": 1024, "min": 320, "max": 1536, "tooltip": "The width used for the image."}),
                "height": ("INT", {"default": 576, "min": 320, "max": 1536, "tooltip": "The height used for the image."}),
                "alignment": (s._alignment_options, {"tooltip": "Where the original image should be in the outpainted one"}),
            },
        }
    
    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE")
    RETURN_NAMES = ("IMAGE", "MASK", "outpaint_cnet_image")
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


def get_first_folder_list(folder_name: str) -> tuple[list[str], dict[str, float], float]:
    folder_name = map_legacy(folder_name)
    global folder_names_and_paths
    folders = folder_names_and_paths[folder_name]
    root_folder = folders[0][0]
    visible_folders = [name for name in os.listdir(root_folder) if os.path.isdir(os.path.join(root_folder, name))]
    return visible_folders

# Tensor to PIL (grabbed from WAS Suite)
def tensor2pil(image: torch.Tensor) -> Image.Image:
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

# Convert PIL to Tensor (grabbed from WAS Suite)
def pil2tensor(image: Image.Image) -> torch.Tensor:
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

class DiffusersImageOutpaint:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "outpaint_cnet_image": ("IMAGE", {"tooltip": "The image to outpaint."}),
                "base_model": (get_first_folder_list("unet"), {"tooltip": "The diffuser model used for denoising the input latent. (Put model files in the unet folder)."}),
                "controlnet": (get_filename_list("controlnet"), {"tooltip": "The controlnet model used for denoising the input latent.(Put model files in the controlnet folder)."}),
                "extra_prompt": ("STRING", {"default": "", "multiline": True, "tooltip": "The extra prompt to append, describing attributes etc. you want to include in the image. Default: \"(extra_prompt), high quality, 4k\""}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 4294967295, "tooltip": "Seed used to generate different images. Fixed it to stop that or for replicate results."}),
                "steps": ("INT", {"default": 8, "min": 1, "max": 99, "tooltip": "The number of steps used in the denoising process."}),
                "guidance_scale": ("FLOAT", {"default": 1.50, "min": 1.01, "max": 10, "step": 0.01, "tooltip": "The Classifier-Free Guidance scale balances creativity and adherence to the prompt. Higher values result in images more closely matching the prompt however too high values will negatively impact quality."}),
                "controlnet_strength": ("FLOAT", {"default": 1.00, "min": 0.00, "max": 10, "step": 0.01}),
                "device": (["auto", "cuda", "cpu", "mps", "xpu", "meta"],{"default": "auto", "tooltip": "Device for inference, default is auto checked by comfyui"}), 
                "dtype": (["auto","fp16","bf16","fp32", "fp8_e4m3fn", "fp8_e4m3fnuz", "fp8_e5m2", "fp8_e5m2fnuz"],{"default":"auto", "tooltip": "Model precision for inference, default is auto checked by comfyui"}),
                "keep_model_loaded": ("BOOLEAN", {"default": False, "label_on": "yes", "label_off": "no", "tooltip": "Set to false to unload all diffusion models."}),
                "keep_model_device": ("BOOLEAN", {"default": True, "label_on": "cpu", "label_off": "device", "tooltip": "If set to device such as cuda, need at least 8gb vram to hold unet + controlnet + vae. If false all models will move to ram, only 1 model will in vram at the same time, useful for under 8gb."}),
                # "debug": ("BOOLEAN", {"default": False, "label_on": "yes", "label_off": "no", "tooltip": "Show debug info."}),
            }
        }

    # RETURN_TYPES = ("IMAGE",)
    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent", )
    FUNCTION = "sample"
    CATEGORY = "DiffusersOutpaint"
    
    def __init__(self) -> None:
        self.prompt_embeds = None
        self.negative_prompt_embeds = None
        self.pooled_prompt_embeds = None
        self.negative_pooled_prompt_embeds = None
        self.cnet_img = None
        self.pipe = None
        self.state_dict = None
        self.model = None
        self.tokenizer = None
        self.tokenizer_2 = None
        self.text_encoder = None
        self.text_encoder_2 = None
        self.loaded_prompt = None
        self.controlnet_model = None
        self.keep_model_loaded = None
        self.fuse_unet = None
        self.loaded_controlnet_name = None

    def sample(self, outpaint_cnet_image, seed, steps, keep_model_loaded, keep_model_device, device, dtype, base_model, controlnet, extra_prompt=None, debug=False, guidance_scale=1.5, controlnet_strength=1.0):
        cnet_image=tensor2pil(outpaint_cnet_image)
        cnet_image=cnet_image.convert('RGB')
        
        final_prompt = extra_prompt + ", high quality, 4k"
        
        base_model_path = os.path.join(models_dir, 'unet', base_model)
        # base_model_path = r'C:\Users\pc\Desktop\RealVisXL_V5.0_Lightning_fp16'
        # debug = True
        
        device = get_device_by_name(device, debug)
        dtype = get_dtype_by_name(dtype, debug)
        
        if self.loaded_prompt == None or self.loaded_prompt != final_prompt:
            if self.tokenizer == None or self.keep_model_loaded == False:
                from transformers import AutoTokenizer
                from transformers import CLIPTextModel
                from transformers import CLIPTextModelWithProjection
                from .DiffusersImageOutpaint_Scripts.pipeline_fill_sd_xl import encode_prompt
                
                if debug:
                    print('\033[93m', 'Loading text_encoders.', '\033[0m')
                self.tokenizer = AutoTokenizer.from_pretrained(base_model_path, subfolder='tokenizer', use_fast=False)
                self.tokenizer_2 = AutoTokenizer.from_pretrained(base_model_path, subfolder='tokenizer_2', use_fast=False)
                self.text_encoder = CLIPTextModel.from_pretrained(base_model_path, subfolder='text_encoder', torch_dtype=dtype).requires_grad_(False).to(device)
                self.text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(base_model_path, subfolder='text_encoder_2', torch_dtype=dtype).requires_grad_(False).to(device)
                if debug:
                    print('\033[93m', 'Text_encoders loading completed.', '\033[0m')
            else:
                self.text_encoder.to(device)
                self.text_encoder_2.to(device)
            
            (self.prompt_embeds,
            self.negative_prompt_embeds,
            self.pooled_prompt_embeds,
            self.negative_pooled_prompt_embeds,
            ) = encode_prompt(self.tokenizer, self.tokenizer_2, self.text_encoder, self.text_encoder_2, final_prompt, device, True)
            if debug:
                print('\033[93m', 'Prompt encoded.', '\033[0m')
        self.loaded_prompt = final_prompt
                
        if not keep_model_loaded:
            del self.text_encoder, self.text_encoder_2, self.tokenizer, self.tokenizer_2
            self.text_encoder = None
            self.text_encoder_2 = None
            self.tokenizer = None
            self.tokenizer_2 = None
            if debug:
                print('\033[93m', 'Delete text_encoders to release vram.', '\033[0m')
            gc.collect()
            torch.cuda.empty_cache()
        else:
            if self.tokenizer != None:
                self.text_encoder.to('cpu')
                self.text_encoder_2.to('cpu')
        self.keep_model_loaded = keep_model_loaded
            
        # Set up Controlnet-Union-Promax-SDXL model
        if self.pipe == None or self.loaded_controlnet_name != controlnet:
            # for speed up startup comfyui, import modules only when this node excuted.
            from .DiffusersImageOutpaint_Scripts.controlnet_union import ControlNetModel_Union
            from diffusers.models.model_loading_utils import load_state_dict
            if debug:
                print('\033[93m', 'Loading ControlNetModel_Union.', '\033[0m')
            config_file = os.path.join(my_dir, 'Config_Files', 'xinsir--controlnet-union-sdxl-1.0_config_promax.json')
            config = ControlNetModel_Union.load_config(config_file)
            self.controlnet_model = ControlNetModel_Union.from_config(config)
            
            controlnet_model_file = get_full_path_or_raise('controlnet', controlnet)
            self.state_dict = load_state_dict(controlnet_model_file)

            self.model, _, _, _, _ = ControlNetModel_Union._load_pretrained_model(
                self.controlnet_model, self.state_dict, controlnet_model_file, 'Any String'
            )
            self.controlnet_model.to(device=device, dtype=dtype)
            if debug:
                print('\033[93m', 'ControlNetModel_Union loading completed.', '\033[0m')
            self.loaded_controlnet_name = controlnet
        
        if self.pipe == None:
            # for speed up startup comfyui, import modules only when this node excuted.
            from .DiffusersImageOutpaint_Scripts.pipeline_fill_sd_xl import StableDiffusionXLFillPipeline
            from diffusers import TCDScheduler

            # Load RealVisXL Unet pipe.
            if debug:
                print('\033[93m', 'Loading StableDiffusionXLFillPipeline.', '\033[0m')
            self.pipe = StableDiffusionXLFillPipeline.from_pretrained(
                base_model_path,
                torch_dtype=dtype,
                variant="fp16",
            )
            if not keep_model_device:
                self.pipe.to(device)
            
            # set up scheduler.
            self.pipe.scheduler = TCDScheduler.from_config(self.pipe.scheduler.config)
            if debug:
                print('\033[93m', 'StableDiffusionXLFillPipeline loading completed.', '\033[0m')
        
        from pytorch_lightning import seed_everything
        seed_everything(seed)
        
        self.cnet_img = cnet_image

        generated_images = list(self.pipe(
            prompt_embeds=self.prompt_embeds,
            negative_prompt_embeds=self.negative_prompt_embeds,
            pooled_prompt_embeds=self.pooled_prompt_embeds,
            negative_pooled_prompt_embeds=self.negative_pooled_prompt_embeds,
            image=cnet_image,
            num_inference_steps=steps,
            keep_model_loaded=keep_model_loaded,
            controlnet_model=self.controlnet_model,
            debug=debug,
            controlnet_conditioning_scale=controlnet_strength,
            guidance_scale=guidance_scale,
            device=device,
            keep_model_device=keep_model_device,
        ))
        
        if keep_model_device:
            self.controlnet_model.to(device)
        
        if not keep_model_loaded:
            del self.state_dict
            del self.model
            del self.pipe
            del self.controlnet_model
            self.state_dict = None
            self.model = None
            self.pipe = None
            self.controlnet_model = None
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        else:
            if keep_model_device:
                self.controlnet_model.to('cpu')

        # last_image = generated_images[-1] # Access the last image
        # image = last_image.convert("RGBA")
        # output=pil2tensor(image)
        output = generated_images[-1]
        
        return (output,)
    
def get_device_by_name(device, debug: bool=False):
    """
    Args:
        "device": (["auto", "cuda", "cpu", "mps", "xpu", "meta", "directml"],{"default": "auto"}), 
    """
    if device == 'auto':
        try:
            device = model_management.get_torch_device()
        except:
                raise AttributeError("What's your device(到底用什么设备跑的)？")
    if debug:
        print("\033[93mUse Device(使用设备):", device, "\033[0m")
    return device

def get_dtype_by_name(dtype, debug: bool=False):
    """
    "dtype": (["auto","fp16","bf16","fp32", "fp8_e4m3fn", "fp8_e4m3fnuz", "fp8_e5m2", "fp8_e5m2fnuz"],{"default":"auto"}),返回模型精度选择。
    """
    if dtype == 'auto':
        try:
            if model_management.should_use_fp16():
                dtype = torch.float16
            elif model_management.should_use_bf16():
                dtype = torch.bfloat16
            else:
                dtype = torch.float32
        except:
                raise AttributeError("ComfyUI version too old, can't autodetect properly. Set your dtypes manually.")
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
    if debug:
        print("\033[93mModel Precision(模型精度):", dtype, "\033[0m")
    return dtype