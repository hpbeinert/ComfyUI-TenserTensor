# (c) TenserTensor || Apache-2.0 (apache.org/licenses/LICENSE-2.0)

from .encoder_helpers import encode_prompts_flux2


class TT_ClipTextEncodeFlux2Context():
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "context": ("TT_CONTEXT",)
            }
        }

    RETURN_TYPES = ("TT_CONTEXT", "GUIDER",)
    RETURN_NAMES = ("CONTEXT", "GUIDER",)
    FUNCTION = "execute"
    CATEGORY = "TenserTensor/Deprecated/Text Encoders/FLUX2"

    def execute(self, context):
        model = context.get("model", None)
        clip = context.get("clip", None)
        config = context.get("workflow_config", None)

        if model is None:
            raise ValueError("Model is required for text encoder")
        if clip is None:
            raise ValueError("CLIP is required for text encoder")
        if config is None:
            raise ValueError("Workflow Config is required for text encoder")

        guider = encode_prompts_flux2(
            model,
            clip,
            config.get("prompt", None),
            config.get("lora_triggers", None),
            config.get("guidance", None),
        )
        context["guider"] = guider

        return (context, guider,)
