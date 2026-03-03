# (c) TenserTensor <tenser.tensor@proton.me> || Apache-2.0 (apache.org/licenses/LICENSE-2.0)

import dataclasses
import math
from dataclasses import dataclass
from typing import override

import torch

from comfy.samplers import KSampler, sampler_object, Sampler
from comfy_api.latest import IO, ComfyExtension
from nodes import MAX_RESOLUTION


@dataclass
class TTWorkflowSettings():
    ascore_negative: float | None = None
    ascore_positive: float | None = None
    cfg: int | None = None
    clip_g_negative: str | None = None
    clip_l_negative: str | None = None
    clip_g_positive: str | None = None
    clip_l_positive: str | None = None
    guidance: float | None = None
    height: int | None = None
    lora_triggers: str | None = None
    prompt: str | None = None
    sampler: Sampler | None = None
    sampler_name: str | None = None
    scheduler: str | None = None
    seed: int | None = None
    sigmas: torch.Tensor | None = None
    steps: int | None = None
    target_height: int | None = None
    target_width: int | None = None
    t5xxl_negative: str | None = None
    t5xxl_positive: str | None = None
    width: int | None = None

    @classmethod
    def create(cls, **kwargs) -> TTWorkflowSettings:
        wf = cls()
        for field in dataclasses.fields(wf):
            wf.set_attr(field.name, kwargs.get(field.name, None))

        return wf

    def update_attrs(self, **kwargs) -> TTWorkflowSettings:
        for key in kwargs.keys():
            if hasattr(self, key):
                val = kwargs.get(key)
                if val is not None:
                    self.set_attr(key, val)

        return self

    def set_attr(self, key, attr) -> None:
        if hasattr(self, key):
            setattr(self, key, attr)

    def get_attr(self, key, default=None) -> None:
        if hasattr(self, key):
            val = getattr(self, key)
            return val if val is not None else default


@IO.comfytype(io_type="TT_WORKFLOW_CONFIG")
class WorkflowSettings:
    Type = TTWorkflowSettings

    class Input(IO.Input):
        def __init__(self, id: str, **kwargs):
            super().__init__(id, **kwargs)

    class Output(IO.Output):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)


def compute_empirical_mu(image_seq_len, num_steps):
    slope_high_res, intercept_high_res = 8.73809524e-05, 1.89833333
    slope_low_res, intercept_low_res = 0.00016927, 0.45666666
    HIGH_RES_THRESHOLD = 4300
    MIN_STEPS, MAX_STEPS = 10, 200
    mu_at_max_steps = slope_low_res * image_seq_len + intercept_low_res

    if image_seq_len > HIGH_RES_THRESHOLD:
        return float(mu_at_max_steps)

    mu_at_min_steps = slope_high_res * image_seq_len + intercept_high_res
    slope = (mu_at_max_steps - mu_at_min_steps) / (MAX_STEPS - MIN_STEPS)
    intercept = mu_at_max_steps - MAX_STEPS * slope
    mu = slope * num_steps + intercept

    return float(mu)


def generalized_time_snr_shift(t, mu, sigma):
    return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)


def get_schedule(steps, width, height):
    image_seq_len = (width * height / (16 * 16))
    mu = compute_empirical_mu(image_seq_len, steps)
    timesteps = torch.linspace(1, 0, steps + 1)
    timesteps = generalized_time_snr_shift(timesteps, mu, 1.0)

    return timesteps


def build_sampler_sigmas(sampler_name, steps, width, height):
    sampler = sampler_object(sampler_name)
    sigmas = get_schedule(steps, width, height)

    return sampler, sigmas


class TT_Flux2WorkflowSettingsNode(IO.ComfyNode):
    """
    Use this node as the single source of truth for FLUX2 workflow parameters. The sampler and sigma schedule
    are computed automatically from the selected sampler, step count, and resolution. Connect `WORKFLOW_CONFIG`
    to context nodes, or use individual outputs to connect to standard ComfyUI nodes.
    """

    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="TT_Flux2WorkflowSettingsNode",
            display_name="TT FLUX2 Workflow Settings",
            category="TenserTensor/Workflow",
            description="",
            inputs=[
                IO.Int.Input("seed", default=0, min=0, max=0xffffffffffffffff),
                IO.Int.Input("steps", default=30, min=1, max=10_000),
                IO.Float.Input("cfg", default=3.0, min=0.0, max=100.0, step=0.1),
                IO.Float.Input("guidance", default=3.0, min=1.0, max=10.0, step=0.1),
                IO.Int.Input("width", default=1024, min=16, max=MAX_RESOLUTION, step=8),
                IO.Int.Input("height", default=1024, min=16, max=MAX_RESOLUTION, step=8),
                IO.Combo.Input("sampler_name", options=KSampler.SAMPLERS)
            ],
            outputs=[
                WorkflowSettings.Output(display_name="WORKFLOW_CONFIG"),
                IO.Sampler.Output(display_name="SAMPLER"),
                IO.Sigmas.Output(display_name="SIGMAS"),
                IO.Int.Output(display_name="SEED"),
                IO.Int.Output(display_name="STEPS"),
                IO.Float.Output(display_name="CFG"),
                IO.Float.Output(display_name="GUIDANCE"),
                IO.Int.Output(display_name="WIDTH"),
                IO.Int.Output(display_name="HEIGHT"),
            ]
        )

    @classmethod
    def execute(cls, **kwargs) -> IO.NodeOutput:
        kwargs["sampler"], kwargs["sigmas"] = build_sampler_sigmas(
            kwargs.get("sampler_name"),
            kwargs.get("steps"),
            kwargs.get("width"),
            kwargs.get("height"),
        )

        workflow_config = TTWorkflowSettings.create(**kwargs)
        args = {
            "workflow_config": workflow_config,
            "sampler": kwargs["sampler"],
            "sigmas": kwargs["sigmas"],
            "seed": kwargs.get("seed"),
            "steps": kwargs.get("steps"),
            "cfg": kwargs.get("cfg"),
            "guidance": kwargs.get("guidance"),
            "width": kwargs.get("width"),
            "height": kwargs.get("height"),
        }

        return IO.NodeOutput(*args.values())


class TT_Flux2WorkflowSettingsAdvancedNode(IO.ComfyNode):
    """
    Advanced version of TT FLUX2 Workflow Settings with `prompt` and `lora_triggers` fields included directly in the
    node. Both are stored in the `WORKFLOW_CONFIG` output alongside all other sampling parameters, making this node
    sufficient as the sole configuration source for full context-driven FLUX2 pipelines. Individual outputs remain
    available for connecting to standard ComfyUI nodes.
    """

    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="TT_Flux2WorkflowSettingsAdvancedNode",
            display_name="TT FLUX2 Workflow Settings (Advanced)",
            category="TenserTensor/Workflow",
            description="",
            inputs=[
                IO.Int.Input("seed", default=0, min=0, max=0xffffffffffffffff),
                IO.Int.Input("steps", default=30, min=1, max=10_000),
                IO.Float.Input("cfg", default=3.0, min=0.0, max=100.0, step=0.1),
                IO.Combo.Input("sampler_name", options=KSampler.SAMPLERS),
                IO.Int.Input("width", default=1024, min=16, max=MAX_RESOLUTION, step=8),
                IO.Int.Input("height", default=1024, min=16, max=MAX_RESOLUTION, step=8),
                IO.String.Input("prompt", multiline=True, dynamic_prompts=True),
                IO.String.Input("lora_triggers", multiline=True, dynamic_prompts=True),
                IO.Float.Input("guidance", default=3.0, min=1.0, max=10.0, step=0.1),
            ],
            outputs=[
                WorkflowSettings.Output(display_name="WORKFLOW_CONFIG"),
                IO.Sampler.Output(display_name="SAMPLER"),
                IO.Sigmas.Output(display_name="SIGMAS"),
                IO.Int.Output(display_name="SEED"),
                IO.Int.Output(display_name="STEPS"),
                IO.Float.Output(display_name="CFG"),
                IO.Int.Output(display_name="WIDTH"),
                IO.Int.Output(display_name="HEIGHT"),
                IO.String.Output(display_name="PROMPT"),
                IO.String.Output(display_name="LORA_TRIGGERS"),
                IO.Float.Output(display_name="GUIDANCE"),
            ]
        )

    @classmethod
    def execute(cls, **kwargs) -> IO.NodeOutput:
        kwargs["sampler"], kwargs["sigmas"] = build_sampler_sigmas(
            kwargs.get("sampler_name"), kwargs.get("steps"), kwargs.get("width"), kwargs.get("height"),
        )

        workflow_config = TTWorkflowSettings.create(**kwargs)
        args = {
            "workflow_config": workflow_config,
            "sampler": kwargs["sampler"],
            "sigmas": kwargs["sigmas"],
            "seed": kwargs.get("seed"),
            "steps": kwargs.get("steps"),
            "cfg": kwargs.get("cfg"),
            "width": kwargs.get("width"),
            "height": kwargs.get("height"),
            "prompt": kwargs.get("prompt"),
            "lora_triggers": kwargs.get("lora_triggers"),
            "guidance": kwargs.get("guidance"),
        }

        return IO.NodeOutput(*args.values())


# ==============================================================================
# V3 entrypoint — registers context nodes with ComfyUI
# ==============================================================================

class WorkflowNodesExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[IO.ComfyNode]]:
        return [
            TT_Flux2WorkflowSettingsNode,
            TT_Flux2WorkflowSettingsAdvancedNode,
        ]


async def comfy_entrypoint() -> WorkflowNodesExtension:
    return WorkflowNodesExtension()


# ==============================================================================
# Re-exports for backward compatibility
# ==============================================================================

__all__ = [
    "TT_Flux2WorkflowSettingsNode",
    "TT_Flux2WorkflowSettingsAdvancedNode",
]
