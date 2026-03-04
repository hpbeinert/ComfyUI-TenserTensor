# (c) TenserTensor <tenser.tensor@proton.me> || Apache-2.0 (apache.org/licenses/LICENSE-2.0)


from typing import override

from kornia.filters import canny

from comfy import model_management
from comfy_api.latest import io, ui, ComfyExtension
from .nodes_image import get_image_files, load_image

CATEGORY = "TenserTensor/Detector"


def detect_edge(image, low_threshold, high_threshold):
    torch_device = model_management.get_torch_device()
    intermediate_device = model_management.intermediate_device()
    timage = image.to(torch_device).movedim(-1, 1)
    o_images = canny(timage, low_threshold, high_threshold)

    return o_images[1].to(intermediate_device).expand(-1, 3, -1, -1).movedim(1, -1)


class TT_CannyEdgeDetectorNode(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="TT_CannyEdgeDetectorNode",
            display_name="TT Canny Edge Detector",
            category=CATEGORY,
            description="",
            inputs=[
                io.Float.Input("low_threshold", default=0.4, min=0.01, max=0.99, step=0.01),
                io.Float.Input("high_threshold", default=0.8, min=0.01, max=0.99, step=0.01),
                io.Combo.Input("reference_image", options=get_image_files(), upload=io.UploadType.image)
            ],
            outputs=[
                io.Image.Output("IMAGE")
            ]
        )

    @classmethod
    def execute(cls, **kwargs) -> io.NodeOutput:
        timage, _ = load_image(kwargs.get("reference_image"))
        cn_image = detect_edge(timage, kwargs.get("low_threshold"), kwargs.get("high_threshold"))

        return io.NodeOutput(cn_image, ui=ui.PreviewImage(timage, cls=cls))


# ==============================================================================
# V3 entrypoint — registers context nodes with ComfyUI
# ==============================================================================

class DetectorNodesExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            TT_CannyEdgeDetectorNode,
        ]


async def comfy_entrypoint() -> DetectorNodesExtension:
    return DetectorNodesExtension()


# ==============================================================================
# Re-exports for backward compatibility
# ==============================================================================

__all__ = [
    "TT_CannyEdgeDetectorNode",
]
