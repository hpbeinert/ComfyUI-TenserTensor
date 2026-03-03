# (c) TenserTensor <tenser.tensor@proton.me> || Apache-2.0 (apache.org/licenses/LICENSE-2.0)

from typing import override

from comfy_api.latest import ComfyExtension, IO


# ==============================================================================
# V3 entrypoint — registers context nodes with ComfyUI
# ==============================================================================

class VaeNodesExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[IO.ComfyNode]]:
        return [
            # TT_Node,
        ]


async def comfy_entrypoint() -> VaeNodesExtension:
    return VaeNodesExtension()


# ==============================================================================
# Re-exports for backward compatibility
# ==============================================================================

__all__ = [
    # "TT_Node",
]
