from .nodes import (PadImageForDiffusersOutpaint, LoadDiffuserModel, LoadDiffuserControlnet, EncodeDiffusersOutpaintPrompt, DiffusersImageOutpaint)

NODE_CLASS_MAPPINGS = {
    "PadImageForDiffusersOutpaint": PadImageForDiffusersOutpaint,
    "LoadDiffuserModel": LoadDiffuserModel,
    "LoadDiffuserControlnet": LoadDiffuserControlnet,
    "EncodeDiffusersOutpaintPrompt": EncodeDiffusersOutpaintPrompt,
    "DiffusersImageOutpaint": DiffusersImageOutpaint
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PadImageForDiffusersOutpaint": "Pad Image For Diffusers Outpaint",
    "LoadDiffuserModel": "Load Diffuser Model",
    "LoadDiffuserControlnet": "Load Diffuser Controlnet",
    "EncodeDiffusersOutpaintPrompt": "Encode Diffusers Outpaint Prompt",
    "DiffusersImageOutpaint": "Diffusers Image Outpaint"
}
