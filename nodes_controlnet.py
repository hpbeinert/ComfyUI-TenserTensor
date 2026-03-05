# (c) TenserTensor <tenser.tensor@proton.me> || Apache-2.0 (apache.org/licenses/LICENSE-2.0)

from typing import override

import folder_paths
from comfy import controlnet
from comfy_api.latest import io, ui, ComfyExtension
from .nodes_image import SingleCondCFGGuider, get_image_files, load_image

CATEGORY = "TenserTensor/ControlNet"


def get_control_net_files():
    return folder_paths.get_filename_list("controlnet")


def load_control_net(filename):
    cnet_path = folder_paths.get_full_path_or_raise("controlnet", filename)

    print(f"CNET path: {cnet_path}")

    cnet = controlnet.load_controlnet(cnet_path)

    if cnet is None:
        raise RuntimeError("ERROR: ControlNet file is invalid or corrupted.")

    return cnet


def apply_control_net(**kwargs):
    guider, image, loaded_cnet = (
        kwargs.get("guider"),
        kwargs.get("cond_hint"),
        kwargs.get("loaded_cnet"),
    )
    conditioning = guider.get_conds()

    cnet_cache = {}
    cond_hint = image.movedim(-1, 1)
    for tensor in conditioning:
        cond_copy = tensor[1].copy()
        prev_cnet = cond_copy.get("control", None)
        if prev_cnet in cnet_cache:
            applied_cnet = cnet_cache[prev_cnet]
        else:
            applied_cnet = loaded_cnet.copy()
            applied_cnet.set_cond_hint(
                cond_hint,
                kwargs.get("strength"),
                (kwargs.get("start_percent") / 100, kwargs.get("end_percent") / 100),
                vae=kwargs.get("vae_opt"),
                extra_concat=kwargs.get("extra_concat", None)
            )
            applied_cnet.set_previous_controlnet(prev_cnet)
            cnet_cache[prev_cnet] = applied_cnet

        cond_copy["control"] = applied_cnet
        cond_copy['control_apply_to_uncond'] = False
        updated_cond = [tensor[0], cond_copy]

    guider.set_conds(updated_cond)

    return guider


class TT_Flux2ApplyControlNetNode(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="TT_Flux2ApplyControlNetNode",
            display_name="TT FLUX2 Apply ControlNet",
            category=CATEGORY,
            description="",
            inputs=[
                io.Guider.Input("guider"),
                io.Image.Input("reference_image"),
                io.ControlNet.Input("control_net_opt", optional=True),
                io.Vae.Input("vae_opt", optional=True),
                io.Combo.Input("control_net", options=get_control_net_files()),
                io.Float.Input("strength", default=1.0, min=0.0, max=10.0, step=0.1),
                io.Float.Input("start_percent", default=0.0, min=0.0, max=100.0, step=0.1, advanced=True),
                io.Float.Input("end_percent", default=100.0, min=0.0, max=100.0, step=0.1, advanced=True),
            ],
            outputs=[
                io.Guider.Output("GUIDER"),
            ]
        )

    @classmethod
    def execute(cls, **kwargs) -> io.NodeOutput:
        guider, strength, control_net = (
            kwargs.get("guider"),
            kwargs.get("strength"),
            kwargs.get("control_net"))

        if not isinstance(guider, SingleCondCFGGuider):
            guider = SingleCondCFGGuider.from_cfg_guider(guider)

        if strength == 0.0:
            return io.NodeOutput(guider)

        loaded_cnet = kwargs.get("control_net_opt")
        kwargs["loaded_cnet"] = (
            loaded_cnet
            if loaded_cnet is not None
            else load_control_net(kwargs.get("control_net"))
        )
        kwargs["cond_hint"] = kwargs.get("reference_image")
        guider = apply_control_net(**kwargs)

        return io.NodeOutput(guider)


class TT_Flux2ApplyControlNetAdvancedNode(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="TT_Flux2ApplyControlNetAdvancedNode",
            display_name="TT FLUX2 Apply ControlNet (Advanced)",
            category=CATEGORY,
            description="",
            inputs=[
                io.Guider.Input("guider"),
                io.ControlNet.Input("control_net_opt", optional=True),
                io.Image.Input("reference_image_opt", optional=True),
                io.Vae.Input("vae_opt", optional=True),
                io.Combo.Input("control_net", options=get_control_net_files()),
                io.Float.Input("strength", default=1.0, min=0.0, max=10.0, step=0.1),
                io.Float.Input("start_percent", default=0.0, min=0.0, max=100.0, step=0.1, advanced=True),
                io.Float.Input("end_percent", default=100.0, min=0.0, max=100.0, step=0.1, advanced=True),
                io.Combo.Input("reference_image", options=get_image_files(), upload=io.UploadType.image)
            ],
            outputs=[
                io.Guider.Output("GUIDER"),
            ]
        )

    @classmethod
    def execute(cls, **kwargs) -> io.NodeOutput:
        guider, strength, control_net = (
            kwargs.get("guider"),
            kwargs.get("strength"),
            kwargs.get("control_net"))

        if not isinstance(guider, SingleCondCFGGuider):
            guider = SingleCondCFGGuider.from_cfg_guider(guider)

        if strength == 0.0:
            return io.NodeOutput(guider)

        reference_image = kwargs.get("reference_image_opt")
        kwargs["cond_hint"] = (
            reference_image
            if reference_image is not None
            else load_image(kwargs.get("reference_image"))[0]
        )

        loaded_cnet = kwargs.get("control_net_opt")
        kwargs["loaded_cnet"] = (
            loaded_cnet
            if loaded_cnet is not None
            else load_control_net(kwargs.get("control_net"))
        )
        guider = apply_control_net(**kwargs)

        return io.NodeOutput(guider, ui=ui.PreviewImage(kwargs["cond_hint"], cls=cls))


# ==============================================================================
# V3 entrypoint — registers context nodes with ComfyUI
# ==============================================================================

class ControlNetNodesExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            TT_Flux2ApplyControlNetNode,
            TT_Flux2ApplyControlNetAdvancedNode,
        ]


async def comfy_entrypoint() -> ControlNetNodesExtension:
    return ControlNetNodesExtension()


# ==============================================================================
# Re-exports for backward compatibility
# ==============================================================================

__all__ = [
    "TT_Flux2ApplyControlNetNode",
    "TT_Flux2ApplyControlNetAdvancedNode",
]
