from .nodes import (PadImageForDiffusersOutpaint, LoadDiffusersOutpaintModels, DiffusersImageOutpaint)

NODE_CLASS_MAPPINGS = {
    "PadImageForDiffusersOutpaint": PadImageForDiffusersOutpaint,
    "LoadDiffusersOutpaintModels": LoadDiffusersOutpaintModels,
    "DiffusersImageOutpaint": DiffusersImageOutpaint
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PadImageForDiffusersOutpaint": "Pad Image For Diffusers Outpaint",
    "LoadDiffusersOutpaintModels": "Load Diffusers Outpaint Models",
    "DiffusersImageOutpaint": "Diffusers Image Outpaint"
}