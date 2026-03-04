# (c) TenserTensor || Apache-2.0 (apache.org/licenses/LICENSE-2.0)

from .encoder_helpers import encode_prompts_flux2


class TT_ClipTextEncodeFlux2():
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "prompt": ("STRING", {"multiline": True, "placeholder": "Prompt", "dynamicPrompts": True}),
                "lora_triggers": ("STRING", {"multiline": True, "placeholder": "LoRA Triggers", "dynamicPrompts": True}),
                "guidance": ("FLOAT", {"default": 3.5, "min": 1.0, "max": 10.0, "step": 0.1})
            }
        }

    RETURN_TYPES = ("GUIDER",)
    RETURN_NAMES = ("GUIDER",)
    FUNCTION = "execute"
    CATEGORY = "TenserTensor/Deprecated/Text Encoders/FLUX2"

    def execute(self, model, clip, prompt, lora_triggers, guidance):
        return encode_prompts_flux2(model, clip, prompt, lora_triggers, guidance)
