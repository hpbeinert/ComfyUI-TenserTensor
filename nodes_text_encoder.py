# (c) TenserTensor <tenser.tensor@proton.me> || Apache-2.0 (apache.org/licenses/LICENSE-2.0)

from typing import override

from comfy_api.latest import io, ComfyExtension
from nodes import MAX_RESOLUTION
from .nodes_context import Context
from .nodes_image import SingleCondCFGGuider

CATEGORY = "TenserTensor/Text Encoder"


def encode_prompts_sdxl(**kwargs):
    clip = kwargs.get("clip")

    if clip is None:
        raise ValueError("ERROR: CLIP is required for text encoder")

    positive_tokens = clip.tokenize(kwargs.get("clip_g_positive"))
    positive_tokens["l"] = clip.tokenize(kwargs.get("clip_l_positive"))["l"]
    negative_tokens = clip.tokenize(kwargs.get("clip_g_negative"))
    negative_tokens["l"] = clip.tokenize(kwargs.get("clip_l_negative"))["l"]

    cond_dict = {
        "aesthetic_score": kwargs.get("ascore_positive"),
        "width": kwargs.get("width"),
        "height": kwargs.get("height"),
        "target_width": kwargs.get("target_width"),
        "target_height": kwargs.get("target_height"),
    }

    positive = clip.encode_from_tokens_scheduled(positive_tokens, add_dict=cond_dict)
    cond_dict["aesthetic_score"] = kwargs.get("ascore_negative")
    negative = clip.encode_from_tokens_scheduled(negative_tokens, add_dict=cond_dict)

    return positive, negative


class TT_SdxlClipTextEncoderNode(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="TT_SdxlClipTextEncoderNode",
            display_name="TT SDXL CLIP Text Encoder",
            category=CATEGORY,
            description="",
            inputs=[
                io.Clip.Input("clip"),
                io.String.Input("clip_l_positive", multiline=True, placeholder="CLIP_L Positive", dynamic_prompts=True),
                io.String.Input("clip_g_positive", multiline=True, placeholder="CLIP_G Positive", dynamic_prompts=True),
                io.String.Input("clip_l_negative", multiline=True, placeholder="CLIP_L Negative", dynamic_prompts=True),
                io.String.Input("clip_g_negative", multiline=True, placeholder="CLIP_G Negative", dynamic_prompts=True),
                io.Float.Input("ascore_positive", default=9.0, min=0.0, max=100.0, step=0.1),
                io.Float.Input("ascore_negative", default=6.0, min=0.0, max=100.0, step=0.1),
                io.Int.Input("width", default=512, min=64, max=MAX_RESOLUTION, step=8),
                io.Int.Input("height", default=512, min=64, max=MAX_RESOLUTION, step=8),
                io.Int.Input("target_width", default=512, min=64, max=MAX_RESOLUTION, step=8),
                io.Int.Input("target_height", default=512, min=64, max=MAX_RESOLUTION, step=8),
            ],
            outputs=[
                io.Conditioning.Output("POSITIVE"),
                io.Conditioning.Output("NEGATIVE"),
            ]
        )

    @classmethod
    def execute(cls, **kwargs) -> io.NodeOutput:
        positive, negative = encode_prompts_sdxl(**kwargs)

        return io.NodeOutput(positive, negative)


class TT_SdxlClipTextEncoderContextNode(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="TT_SdxlClipTextEncoderContextNode",
            display_name="TT SDXL CLIP Text Encoder (Context)",
            category=CATEGORY,
            description="",
            inputs=[
                Context.Input("context")
            ],
            outputs=[
                Context.Output("CONTEXT"),
                io.Conditioning.Output("POSITIVE"),
                io.Conditioning.Output("NEGATIVE"),
            ]
        )

    @classmethod
    def execute(cls, **kwargs) -> io.NodeOutput:
        context = kwargs.get("context")

        args = {}
        for key in (
                "clip", "clip_l_positive", "clip_g_positive", "clip_l_negative", "clip_g_negative",
                "ascore_positive", "ascore_negative", "width", "height", "target_width", "target_height",
        ):
            args[key] = context.get_attr(key)

        positive, negative = encode_prompts_sdxl(**args)
        context.set_attr("positive", positive)
        context.set_attr("negative", negative)

        return io.NodeOutput(context, positive, negative)


def encode_prompts_flux(**kwargs):
    clip = kwargs.get("clip")

    if clip is None:
        raise ValueError("ERROR: CLIP is required for text encoder")

    positive_tokens = clip.tokenize(kwargs.get("clip_l_positive"))
    positive_tokens["t5xxl"] = clip.tokenize(kwargs.get("t5xxl_positive"))["t5xxl"]
    negative_tokens = clip.tokenize(kwargs.get("clip_l_negative"))
    negative_tokens["t5xxl"] = clip.tokenize(kwargs.get("t5xxl_negative"))["t5xxl"]

    guidance = kwargs.get("guidance")
    positive = clip.encode_from_tokens_scheduled(positive_tokens, add_dict={"guidance": guidance, })
    negative = clip.encode_from_tokens_scheduled(negative_tokens, add_dict={"guidance": guidance, })

    return positive, negative


class TT_Flux1ClipTextEncoderNode(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="TT_Flux1ClipTextEncoderNode",
            display_name="TT FLUX1 CLIP Text Encoder",
            category=CATEGORY,
            description="",
            inputs=[
                io.Clip.Input("clip"),
                io.String.Input("clip_l_positive", multiline=True, placeholder="CLIP_L Positive", dynamic_prompts=True),
                io.String.Input("t5xxl_positive", multiline=True, placeholder="T5XXL Positive", dynamic_prompts=True),
                io.String.Input("clip_l_negative", multiline=True, placeholder="CLIP_L Negative", dynamic_prompts=True),
                io.String.Input("t5xxl_negative", multiline=True, placeholder="T5XXL Negative", dynamic_prompts=True),
                io.Float.Input("guidance", default=9.0, min=0.0, max=100.0, step=0.1),
            ],
            outputs=[
                io.Conditioning.Output("POSITIVE"),
                io.Conditioning.Output("NEGATIVE"),
            ]
        )

    @classmethod
    def execute(cls, **kwargs) -> io.NodeOutput:
        positive, negative = encode_prompts_flux(**kwargs)

        return io.NodeOutput(positive, negative)


class TT_Flux1ClipTextEncoderContextNode(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="TT_Flux1ClipTextEncoderContextNode",
            display_name="TT FLUX1 CLIP Text Encoder (Context)",
            category=CATEGORY,
            description="",
            inputs=[
                Context.Input("context")
            ],
            outputs=[
                Context.Output("CONTEXT"),
                io.Conditioning.Output("POSITIVE"),
                io.Conditioning.Output("NEGATIVE"),
            ]
        )

    @classmethod
    def execute(cls, **kwargs) -> io.NodeOutput:
        context = kwargs.get("context")

        args = {}
        for key in (
                "clip", "clip_l_positive", "t5xxl_positive", "clip_l_negative", "t5xxl_negative",
                "guidance",
        ):
            args[key] = context.get_attr(key)

        positive, negative = encode_prompts_flux(**args)
        context.set_attr("positive", positive)
        context.set_attr("negative", negative)

        return io.NodeOutput(context, positive, negative)


def encode_prompts_flux2(**kwargs):
    clip = kwargs.get("clip")
    model = kwargs.get("model")

    if model is None:
        raise ValueError("ERROR: MODEL is required for text encoder")

    if clip is None:
        raise ValueError("ERROR: CLIP is required for text encoder")

    prompt, lora_triggers, guidance = (
        kwargs.get("prompt"),
        kwargs.get("lora_triggers"),
        kwargs.get("guidance"),
    )

    full_prompt = f"{lora_triggers}, {prompt}" if lora_triggers.strip() else prompt
    tokens = clip.tokenize(full_prompt)
    conditioning = clip.encode_from_tokens_scheduled(tokens, add_dict={"guidance": guidance, })
    guider = SingleCondCFGGuider(model)
    guider.set_conds(conditioning)

    return guider


class TT_Flux2TextEncoderNode(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="TT_Flux2TextEncoderNode",
            display_name="TT FLUX2 Text Encoder",
            category=CATEGORY,
            description="",
            inputs=[
                io.Model.Input("model"),
                io.Clip.Input("clip"),
                io.String.Input("prompt", multiline=True, placeholder="Prompt", dynamic_prompts=True),
                io.String.Input("lora_triggers", multiline=True, placeholder="LoRA Triggers", dynamic_prompts=True),
                io.Float.Input("guidance", default=3.5, min=1.0, max=10.0, step=0.1)
            ],
            outputs=[
                io.Guider.Output("GUIDER"),
            ]
        )

    @classmethod
    def execute(cls, **kwargs) -> io.NodeOutput:
        guider = encode_prompts_flux2(**kwargs)

        return io.NodeOutput(guider)


class TT_Flux2TextEncoderContextNode(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="TT_Flux2TextEncoderContextNode",
            display_name="TT FLUX2 Text Encoder (Context)",
            category=CATEGORY,
            description="",
            inputs=[
                Context.Input("context")
            ],
            outputs=[
                Context.Output("CONTEXT"),
                io.Guider.Output("GUIDER"),
            ]
        )

    @classmethod
    def execute(cls, **kwargs) -> io.NodeOutput:
        context = kwargs.get("context")

        args = {
            "model": context.get_attr("model"),
            "clip": context.get_attr("clip"),
            "prompt": context.get_attr("prompt"),
            "lora_triggers": context.get_attr("lora_triggers"),
            "guidance": context.get_attr("guidance"),
        }
        guider = encode_prompts_flux2(**args)
        context.set_attr("guider", guider)

        return io.NodeOutput(context, guider)


# ==============================================================================
# V3 entrypoint — registers context nodes with ComfyUI
# ==============================================================================

class TextEncodeNodesExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            TT_SdxlClipTextEncoderNode,
            TT_SdxlClipTextEncoderContextNode,
            TT_Flux1ClipTextEncoderNode,
            TT_Flux1ClipTextEncoderContextNode,
            TT_Flux2TextEncoderNode,
            TT_Flux2TextEncoderContextNode,
        ]


async def comfy_entrypoint() -> TextEncodeNodesExtension:
    return TextEncodeNodesExtension()


# ==============================================================================
# Re-exports for backward compatibility
# ==============================================================================

__all__ = [
    "TT_SdxlClipTextEncoderNode",
    "TT_SdxlClipTextEncoderContextNode",
    "TT_Flux1ClipTextEncoderNode",
    "TT_Flux1ClipTextEncoderContextNode",
    "TT_Flux2TextEncoderNode",
    "TT_Flux2TextEncoderContextNode",
]
