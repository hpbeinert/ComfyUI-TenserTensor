# (c) TenserTensor <tenser.tensor@proton.me> || Apache-2.0 (apache.org/licenses/LICENSE-2.0)

from typing import override

import latent_preview
from comfy import sample, utils, model_management
from comfy_api.latest import ComfyExtension, IO

CATEGORY = "TenserTensor/Sampling"
SCALE_METHODS = ["nearest-exact", "bilinear", "area", "bicubic", "bislerp"]
MEMORY_POLICIES = ["default", "balanced", "aggressive"]
SCALE_FACTORS = ["0.25x", "0.5x", "1x", "2x", "4x", "8x"]


def execute_guided_sampling(**kwargs):
    guider, latent, sigmas, random_noise, sampler = (
        kwargs.get("guider"),
        kwargs.get("latent"),
        kwargs.get("sigmas"),
        kwargs.get("random_noise"),
        kwargs.get("sampler"),
    )

    model_patcher = guider.model_patcher
    latent_dict = latent.copy()
    latent_tensor = sample.fix_empty_latent_channels(
        model_patcher,
        latent_dict["samples"],
        latent.get("downscale_ratio_spacial", None)
    )
    latent_dict["samples"] = latent_tensor
    noise_mask = latent.get("noise_mask", None)
    interim_buffer = {}
    callback = latent_preview.prepare_callback(model_patcher, sigmas.shape[-1] - 1, interim_buffer)
    disable_progress_bar = not utils.PROGRESS_BAR_ENABLED

    samples = guider.sample(
        random_noise.generate_noise(latent),
        latent_tensor,
        sampler,
        sigmas,
        denoise_mask=noise_mask,
        callback=callback,
        disable_pbar=disable_progress_bar,
        seed=random_noise.seed
    ).to(model_management.intermediate_device())

    sampled_latent = latent_dict.copy()
    sampled_latent.pop("downscale_ratio_spacial", None)
    sampled_latent["samples"] = samples

    return sampled_latent


class TT_GuidedKSamplerNode(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="TT_GuidedKSamplerNode",
            display_name="TT Guided KSampler",
            category=CATEGORY,
            description="",
            inputs=[
                IO.Latent.Input("latent"),
                IO.Guider.Input("guider"),
                IO.Sigmas.Input("sigmas"),
                IO.Sampler.Input("sampler"),
                IO.Noise.Input("random_noise"),
            ],
            outputs=[
                IO.Latent.Output("SAMPLES")
            ]
        )

    @classmethod
    def execute(cls, **kwargs) -> IO.NodeOutput:
        samples = execute_guided_sampling(**kwargs)

        return IO.NodeOutput(samples)


def scale_latent(latent, **kwargs):
    samples = latent.copy()["samples"]
    orig_height, orig_width = samples.shape[2:]
    scale_factor = float(kwargs.get("scale_factor").replace("x", ""))
    final_width, final_height = round(orig_width * scale_factor), round(orig_height * scale_factor)
    scaled = utils.common_upscale(
        samples,
        final_width,
        final_height, kwargs.get("scale_method"),
        "disabled"
    )

    return {"samples": scaled}


class TT_GuidedUpscaleKSamplerNode(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="TT_GuidedUpscaleKSamplerNode",
            display_name="TT Guided Upscale KSampler",
            category=CATEGORY,
            description="",
            inputs=[
                IO.Latent.Input("latent"),
                IO.Guider.Input("guider"),
                IO.Sigmas.Input("sigmas"),
                IO.Sampler.Input("sampler"),
                IO.Noise.Input("random_noise"),
                IO.Boolean.Input("scale_latent", default=True, label_on="Scale latent", label_off="Keep size"),
                IO.Combo.Input("scale_factor", options=SCALE_FACTORS, default="1x", advanced=True),
                IO.Combo.Input("scale_method", options=SCALE_METHODS, default="nearest-exact", advanced=True),
            ],
            outputs=[
                IO.Latent.Output("SAMPLES")
            ]
        )

    @classmethod
    def execute(cls, **kwargs) -> IO.NodeOutput:
        samples = execute_guided_sampling(**kwargs)

        if kwargs.get("scale_latent"):
            samples = scale_latent(samples, **kwargs)

        return IO.NodeOutput(samples)

class TT_LatentMultiTransformNode(IO.ComfyNode):
    pass

# ==============================================================================
# V3 entrypoint — registers context nodes with ComfyUI
# ==============================================================================

class LatentNodesExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[IO.ComfyNode]]:
        return [
            TT_GuidedKSamplerNode,
            TT_GuidedUpscaleKSamplerNode,
        ]


async def comfy_entrypoint() -> LatentNodesExtension:
    return LatentNodesExtension()


# ==============================================================================
# Re-exports for backward compatibility
# ==============================================================================

__all__ = [
    "TT_GuidedKSamplerNode",
    "TT_GuidedUpscaleKSamplerNode",
]
