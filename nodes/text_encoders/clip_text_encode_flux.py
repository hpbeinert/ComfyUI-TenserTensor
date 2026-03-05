# (c) TenserTensor || Apache-2.0 (apache.org/licenses/LICENSE-2.0)

from .encoder_helpers import encode_prompts_flux


class TT_ClipTextEncodeFlux():
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "clip_l_positive": ("STRING", {"multiline": True, "placeholder": "CLIP_L Positive", "dynamicPrompts": True}),
                "t5xxl_positive": ("STRING", {"multiline": True, "placeholder": "T5XXL Positive", "dynamicPrompts": True}),
                "clip_l_negative": ("STRING", {"multiline": True, "placeholder": "CLIP_L Negative", "dynamicPrompts": True}),
                "t5xxl_negative": ("STRING", {"multiline": True, "placeholder": "CLIP_L Negative", "dynamicPrompts": True}),
                "guidance": ("FLOAT", {"default": 3.5, "min": 1.0, "max": 10.0, "step": 0.1})
            }
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING",)
    RETURN_NAMES = ("POSITIVE", "NEGATIVE",)
    FUNCTION = "execute"
    CATEGORY = "TenserTensor/Deprecated/Text Encoders/FLUX"

    def execute(self, **kwargs):
        return encode_prompts_flux(**kwargs)
