# (c) TenserTensor || Apache-2.0 (apache.org/licenses/LICENSE-2.0)

from nodes import MAX_RESOLUTION
from .encoder_helpers import encode_prompts_sdxl


class TT_ClipTextEncodeSdxl():
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "clip_l_positive": ("STRING", {"multiline": True, "placeholder": "CLIP_L Positive", "dynamicPrompts": True}),
                "clip_g_positive": ("STRING", {"multiline": True, "placeholder": "CLIP_G Positive", "dynamicPrompts": True}),
                "clip_l_negative": ("STRING", {"multiline": True, "placeholder": "CLIP_L Negative", "dynamicPrompts": True}),
                "clip_g_negative": ("STRING", {"multiline": True, "placeholder": "CLIP_G Negative", "dynamicPrompts": True}),
                "ascore_positive": ("FLOAT", {"default": 9.0, "min": 0.0, "max": 1000.0, "step": 0.1}),
                "ascore_negative": ("FLOAT", {"default": 6.0, "min": 0.0, "max": 1000.0, "step": 0.1}),
                "width": ("INT", {"default": 512, "min": 0.0, "max": MAX_RESOLUTION}),
                "height": ("INT", {"default": 512, "min": 0.0, "max": MAX_RESOLUTION}),
                "target_width": ("INT", {"default": 512, "min": 0.0, "max": MAX_RESOLUTION}),
                "target_height": ("INT", {"default": 512, "min": 0.0, "max": MAX_RESOLUTION}),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING",)
    RETURN_NAMES = ("POSITIVE", "NEGATIVE",)
    FUNCTION = "execute"
    CATEGORY = "TenserTensor/Deprecated/Text Encoders/SDXL"

    def execute(self, **kwargs):
        return encode_prompts_sdxl(**kwargs)
