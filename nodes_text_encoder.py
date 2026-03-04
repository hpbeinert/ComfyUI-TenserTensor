# (c) TenserTensor <tenser.tensor@proton.me> || Apache-2.0 (apache.org/licenses/LICENSE-2.0)

from typing import override

from comfy_api.latest import io, ComfyExtension
from .nodes_context import Context
from .nodes_image import SingleCondCFGGuider

CATEGORY = "TenserTensor/Text Encoder"


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
            TT_Flux2TextEncoderNode,
            TT_Flux2TextEncoderContextNode,
        ]


async def comfy_entrypoint() -> TextEncodeNodesExtension:
    return TextEncodeNodesExtension()


# ==============================================================================
# Re-exports for backward compatibility
# ==============================================================================

__all__ = [
    "TT_Flux2TextEncoderNode",
    "TT_Flux2TextEncoderContextNode",
]
