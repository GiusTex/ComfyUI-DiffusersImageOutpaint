from .nodes import (PadImageForDiffusersOutpaint, DiffusersImageOutpaint)

NODE_CLASS_MAPPINGS = {
    "PadImageForDiffusersOutpaint": PadImageForDiffusersOutpaint,
    "DiffusersImageOutpaint": DiffusersImageOutpaint
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PadImageForDiffusersOutpaint": "Pad Image For Outpaint",
    "DiffusersImageOutpaint": "Diffusers Image Outpaint"
}