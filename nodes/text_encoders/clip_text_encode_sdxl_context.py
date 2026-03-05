# (c) TenserTensor || Apache-2.0 (apache.org/licenses/LICENSE-2.0)

from .encoder_helpers import encode_prompts_sdxl


class TT_ClipTextEncodeSdxlContext():
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "context": ("TT_CONTEXT",)
            }
        }

    RETURN_TYPES = ("TT_CONTEXT", "CONDITIONING", "CONDITIONING",)
    RETURN_NAMES = ("CONTEXT", "POSITIVE", "NEGATIVE",)
    FUNCTION = "execute"
    CATEGORY = "TenserTensor/Deprecated/Text Encoders/SDXL"

    def execute(self, context):
        clip = context["clip"]
        config = context["workflow_config"]

        positive, negative = encode_prompts_sdxl(
            clip,
            config["clip_l_positive"],
            config["clip_g_positive"],
            config["clip_l_negative"],
            config["clip_g_negative"],
            config["ascore_positive"],
            config["ascore_negative"],
            config["width"],
            config["height"],
            config["target_width"],
            config["target_height"],
        )

        context["positive"] = positive
        context["negative"] = negative

        return (context, positive, negative,)
