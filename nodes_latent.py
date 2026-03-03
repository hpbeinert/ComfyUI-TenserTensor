# (c) TenserTensor <tenser.tensor@proton.me> || Apache-2.0 (apache.org/licenses/LICENSE-2.0)

import math
from typing import Any, override

import torch

from comfy import sample
from comfy_api.latest import IO, ComfyExtension

CATEGORY = "TenserTensor/Latent"

ASPECT_RATIOS = ["1:1", "4:3", "3:2", "16:9", "21:9"]
MEGAPIXELS = ["0.25 MP", "0.5 MP", "1 MP", "2 MP", "4 MP", "8 MP"]
ORIENTATIONS = ["landscape", "portrait"]
MODEL_TYPES = ["FLUX1.D", "FLUX2.D", "SDXL"]
CLIP_MULTIPLIERS = ["1x", "2x", "4x"]


class RandomNoise:
    def __init__(self, seed):
        self.seed = seed

    def generate_noise(self, input_latent):
        latent_image = input_latent["samples"]
        batch_idx = input_latent["batch_index"] if "batch_index" in input_latent else None

        return sample.prepare_noise(latent_image, self.seed, batch_idx)


class TT_LatentFactoryNode(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="TT_LatentFactoryNode",
            display_name="TT Latent Factory",
            category=CATEGORY,
            description="",
            inputs=[
                IO.Int.Input("seed", default=0, min=0, max=0xffffffffffffffff),
                IO.Int.Input("noise_seed", default=0, min=0, max=0xffffffffffffffff, control_after_generate=True),
                IO.Combo.Input("aspect_ratio", options=ASPECT_RATIOS),
                IO.Combo.Input("megapixels", options=MEGAPIXELS),
                IO.Combo.Input("orientation", options=ORIENTATIONS),
                IO.Combo.Input("model_type", options=MODEL_TYPES),
                IO.Int.Input("batch_size", default=1, min=1, max=64, advanced=True),
                IO.Combo.Input("clip_multiplier", options=CLIP_MULTIPLIERS, advanced=True),
            ],
            outputs=[
                IO.Latent.Output(display_name="LATENT"),
                IO.Noise.Output(display_name="RND_NOISE"),
                IO.Int.Output(display_name="SEED"),
                IO.Int.Output(display_name="NOISE_SEED"),
                IO.String.Output(display_name="MEGAPIXELS"),
                IO.Int.Output(display_name="WIDTH"),
                IO.Int.Output(display_name="HEIGHT"),
                IO.Int.Output(display_name="TARGET_WIDTH"),
                IO.Int.Output(display_name="TARGET_HEIGHT"),
            ]
        )

    LATENT_DIMENSION_STEP = 64

    @classmethod
    def calculate_dimensions(cls, total_pixels, ratio_w, ratio_h) -> tuple[int, int]:
        height = int(math.sqrt(total_pixels * ratio_h / ratio_w))
        width = int(height * ratio_w / ratio_h)
        width = round(width / cls.LATENT_DIMENSION_STEP) * cls.LATENT_DIMENSION_STEP
        height = round(height / cls.LATENT_DIMENSION_STEP) * cls.LATENT_DIMENSION_STEP

        return width, height,

    @classmethod
    def build_latent(cls, seed, batch_size, channels, latent_width, latent_height) -> torch.Tensor:
        generator = torch.Generator().manual_seed(seed)
        latent = torch.randn(
            batch_size,
            channels,
            latent_height,
            latent_width,
            generator=generator
        )

        return latent

    @classmethod
    def create_empty_latent(cls, **kwargs) -> tuple[dict[str, Any], int, int]:
        total_pixels = int(float(kwargs.get("megapixels").split()[0]) * 1_000_000)
        ratio_parts = kwargs.get("aspect_ratio").split(':')
        ratio_w, ratio_h = (
            (int(ratio_parts[0]), int(ratio_parts[1]))
            if kwargs.get("orientation") == "landscape"
            else (int(ratio_parts[1]), int(ratio_parts[0]))
        )
        width, height = cls.calculate_dimensions(total_pixels, ratio_w, ratio_h)

        scale_factor, channels = None, None
        match kwargs.get("model_type"):
            case "FLUX1.D":
                scale_factor, channels = 8, 16
            case "FLUX2.D":
                scale_factor, channels = 16, 128
            case "SDXL":
                scale_factor, channels = 8, 4

        latent_width, latent_height = (width // scale_factor, height // scale_factor,)
        latent = cls.build_latent(kwargs.get("seed"), kwargs.get("batch_size"), channels, latent_width, latent_height)

        return {"samples": latent}, width, height,

    @classmethod
    def execute(cls, **kwargs) -> IO.NodeOutput:
        samples, width, height = cls.create_empty_latent(**kwargs)
        seed, noise_seed = kwargs.get("seed"), kwargs.get("noise_seed")
        noise = RandomNoise(noise_seed)
        multiplier = int(kwargs.get("clip_multiplier").replace('x', ''))
        clip_width, clip_height = (width * multiplier, height * multiplier,)

        args = {
            "latent": samples,
            "noise": noise,
            "seed": seed,
            "noise_seed": noise_seed,
            "megapixels": kwargs.get("megapixels"),
            "width": width,
            "height": height,
            "target_width": clip_width,
            "target_height": clip_height,
        }

        return IO.NodeOutput(*args.values())


# ==============================================================================
# V3 entrypoint — registers context nodes with ComfyUI
# ==============================================================================

class LatentNodesExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[IO.ComfyNode]]:
        return [
            TT_LatentFactoryNode,
        ]


async def comfy_entrypoint() -> LatentNodesExtension:
    return LatentNodesExtension()


# ==============================================================================
# Re-exports for backward compatibility
# ==============================================================================

__all__ = [
    "TT_LatentFactoryNode",
]
