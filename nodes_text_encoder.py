# (c) TenserTensor <tenser.tensor@proton.me> || Apache-2.0 (apache.org/licenses/LICENSE-2.0)

from typing import override

from comfy_api.latest import io, ComfyExtension

CATEGORY = "TenserTensor/Text Encoder"


class TT_Flux2TextEncoderNode(io.ComfyNode):
    pass


class TT_Flux2TextEncoderContextNode(io.ComfyNode):
    pass


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
