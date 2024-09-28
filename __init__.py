from .nodes import (PadImageForDiffusersOutpaint, LoadDiffusersOutpaintModels, EncodeDiffusersOutpaintPrompt, DiffusersImageOutpaint)

NODE_CLASS_MAPPINGS = {
    "PadImageForDiffusersOutpaint": PadImageForDiffusersOutpaint,
    "LoadDiffusersOutpaintModels": LoadDiffusersOutpaintModels,
    "EncodeDiffusersOutpaintPrompt": EncodeDiffusersOutpaintPrompt,
    "DiffusersImageOutpaint": DiffusersImageOutpaint
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PadImageForDiffusersOutpaint": "Pad Image For Diffusers Outpaint",
    "LoadDiffusersOutpaintModels": "Load Diffusers Outpaint Models",
    "EncodeDiffusersOutpaintPrompt": "Encode Diffusers Outpaint Prompt",
    "DiffusersImageOutpaint": "Diffusers Image Outpaint"
}
