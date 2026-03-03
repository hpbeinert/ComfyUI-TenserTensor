# (c) TenserTensor <tenser.tensor@proton.me> || Apache-2.0 (apache.org/licenses/LICENSE-2.0)

import dataclasses
import re
from dataclasses import dataclass
from typing import override

import torch

from comfy.clip_vision import Output
from comfy_api.latest import IO, ComfyExtension, Input
from .nodes_workflow import TTWorkflowSettings, WorkflowSettings

CATEGORY = "TenserTensor/Context"


@dataclass
class TTContext():
    """
    Shared context object passed between TenserTensor context nodes.

    Holds all pipeline state as optional fields.
    All fields default to None and are populated incrementally as the workflow progresses.

    Use create() to instantiate from kwargs, and set_attr/get_attr for safe field access.
    Workflow config fields are accessible via set_workflow_config_attr/get_workflow_config_attr.
    """
    cfg: IO.Float | None = None
    clip: IO.Clip | None = None
    control_net: IO.ControlNet | None = None
    conditioning: IO.Conditioning | None = None
    guidance: IO.Float | None = None
    guider: IO.Guider | None = None
    image: IO.Image | None = None
    latent: IO.Latent | None = None
    lora_triggers: IO.String | None = None
    mask: IO.Mask | None = None
    model: IO.Model | None = None
    negative: IO.Conditioning | None = None
    positive: IO.Conditioning | None = None
    prompt: IO.String | None = None
    rnd_noise: IO.Noise | None = None
    sampler: IO.Sampler | None = None
    seed: IO.Int | None = None
    sigmas: IO.Sigmas | None = None
    steps: IO.Int | None = None
    vae: IO.Vae | None = None
    workflow_config: WorkflowSettings | None = None
    sampler_name: IO.String | None = None
    scheduler: IO.String | None = None

    @classmethod
    def create(cls, **kwargs):
        ctx = cls()
        for field in dataclasses.fields(ctx):
            ctx.set_attr(field.name, kwargs.get(field.name, None))

        return ctx

    def set_attr(self, key, attr):
        if attr is not None:
            if hasattr(self.workflow_config, key):
                self.set_workflow_config_attr(key, attr)
            else:
                setattr(self, key, attr)

    def get_attr(self, key, default=None):
        if hasattr(self.workflow_config, key):
            val = self.get_workflow_config_attr(key, default)
        else:
            val = getattr(self, key)

        return val if val is not None else default

    def update_attrs(self, **kwargs):
        for key in kwargs.keys():
            val = kwargs.get(key, None)
            if hasattr(self.workflow_config, key):
                self.set_workflow_config_attr(key, val)
            else:
                self.set_attr(key, val)

    def set_workflow_config_attr(self, key, attr):
        if self.workflow_config:
            self.workflow_config.set_attr(key, attr)
        else:
            raise AttributeError(f"ERROR: Workflow Config is not set in context")

    def get_workflow_config_attr(self, key, default=None):
        if self.workflow_config:
            return self.workflow_config.get_attr(key, default)
        else:
            raise AttributeError(f"ERROR: Workflow Config is not set in context")

    def update_workflow_config_attrs(self, **kwargs):
        if self.workflow_config:
            for key in kwargs.keys():
                if hasattr(self.workflow_config, key):
                    val = kwargs.get(key)
                    if val is not None:
                        self.workflow_config.set_attr(key, val)
        else:
            raise AttributeError(f"ERROR: Workflow Config is not set in context")


@IO.comfytype(io_type="TT_CONTEXT")
class Context:
    Type = TTContext

    class Input(IO.Input):
        def __init__(self, id: str, **kwargs):
            super().__init__(id, **kwargs)

    class Output(IO.Output):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)


class ContextNode(IO.ComfyNode):
    """
    Abstract base class for all context nodes.
    Handles context creation, delegation to handle(), and building return tuple.
    Subclasses should set INPUT_TYPES_DICT, RETURN_TYPES, RETURN_NAMES from a Schema,
    and override handle() to implement node-specific logic.
    """
    REQUIRED = ()
    OPTIONAL = ()
    RETURNS = ()

    F_TYPES = {}

    @classmethod
    def fill_field_types(cls):
        cls.F_TYPES = {f.name: f.type.__args__[0] for f in dataclasses.fields(TTContext)}

    @classmethod
    def inputs(cls) -> list[Input]:
        def get_input(input_name, optional):
            match input_name:
                case "context":
                    return Context.Input(input_name, optional=optional)
                case _:
                    f_type = cls.F_TYPES[input_name]
                    if f_type in (IO.Boolean, IO.Int, IO.Float, IO.String,):
                        return f_type.Input(input_name, optional=optional, force_input=True)
                    else:
                        return f_type.Input(input_name, optional=optional)

        input_schema = []

        for f_name in cls.REQUIRED:
            input_schema.append(get_input(f_name, optional=False))

        for f_name in cls.OPTIONAL:
            input_schema.append(get_input(f_name, optional=True))

        return input_schema

    @classmethod
    def outputs(cls) -> list[Output]:
        output_schema = []

        for f_name in cls.RETURNS if len(cls.RETURNS) > 0 else cls.REQUIRED + cls.OPTIONAL:
            output_schema.append(
                Context.Output(display_name="CONTEXT")
                if f_name == "context"
                else cls.F_TYPES[f_name].Output(display_name=f_name.upper())
            )

        return output_schema

    @classmethod
    def node_output(cls, context) -> IO.NodeOutput:
        args = [context]
        for f_name in cls.RETURNS if len(cls.RETURNS) > 0 else cls.REQUIRED + cls.OPTIONAL:
            if f_name != "context":
                args.append(context.get_attr(f_name))

        return IO.NodeOutput(*args)

    @classmethod
    def display_name(cls):
        return re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', cls.__name__.replace("TT_", "TT ").replace("Node", ""))

    @classmethod
    def define_schema(cls) -> IO.Schema:
        if len(cls.F_TYPES) == 0:
            cls.fill_field_types()

        return IO.Schema(
            node_id=cls.__name__,
            display_name=cls.display_name(),
            category=CATEGORY,
            inputs=cls.inputs(),
            outputs=cls.outputs(),
        )

    @classmethod
    def handle(cls, context, **kwargs):
        workflow_config = context.get_attr("workflow_config")
        kwargs["workflow_config"] = (
            TTWorkflowSettings.create(**kwargs)
            if not workflow_config
            else workflow_config.update_attrs(**kwargs)
        )
        context.update_attrs(**kwargs)

        return (context, kwargs,)

    @classmethod
    def execute(cls, context=None, **kwargs) -> IO.NodeOutput:
        if not context:
            context = TTContext.create(**kwargs)
            cls.handle(context, **kwargs)
        else:
            context, kwargs = cls.handle(context, **kwargs)

        return cls.node_output(context)


class TT_BaseContextNode(ContextNode):
    """Base context node. Accepts model, clip, vae, latent and optional workflow_config."""
    REQUIRED = ("model", "clip", "vae", "latent")
    OPTIONAL = ("workflow_config",)
    RETURNS = ("context",)


class TT_BaseContextFlux2Node(ContextNode):
    """
    Base context node for FLUX2 workflows. Accepts model, clip, vae, latent, sampler, noise, and sigmas.
    Optionally accepts workflow_config. Returns context.
    """
    REQUIRED = ("model", "clip", "vae", "latent", "rnd_noise",)
    OPTIONAL = ("workflow_config", "sampler", "sigmas",)
    RETURNS = ("context",)


class TT_BaseContextPassthroughNode(ContextNode):
    """Passes context through without modification."""
    REQUIRED = ("context",)
    OPTIONAL = ()
    RETURNS = ("context",)


class TT_ContextNode(ContextNode):
    """General-purpose context node for FLUX and SDXL workflows. All fields are optional."""
    REQUIRED = ()
    OPTIONAL = (
        "context", "workflow_config", "model", "clip", "vae", "latent", "positive", "negative", "image",
        "seed", "steps", "cfg",
    )
    RETURNS = ()


class TT_ContextFlux2Node(ContextNode):
    """Context node for FLUX2 workflows. All fields are optional."""
    REQUIRED = ()
    OPTIONAL = (
        "context", "workflow_config", "model", "clip", "vae", "latent", "conditioning", "guider", "rnd_noise",
        "sampler", "sigmas", "image", "mask", "seed", "steps", "cfg",
    )
    RETURNS = ()


class TT_ContextSetGuiderNode(ContextNode):
    """Sets the guider field in the context object."""
    REQUIRED = ("context", "guider",)
    OPTIONAL = ()
    RETURNS = ("context",)


class TT_ContextSetImageNode(ContextNode):
    """Sets the image field in the context object."""
    REQUIRED = ("context", "image",)
    OPTIONAL = ()
    RETURNS = ("context",)


class TT_ContextSetLatentNode(ContextNode):
    """Sets the latent field in the context object."""
    REQUIRED = ("context", "latent",)
    OPTIONAL = ()
    RETURNS = ("context",)


class TT_ContextExtractEncoderFlux2Node(ContextNode):
    """Extracts model, clip, and encoding parameters from context for use with a FLUX2 text encoder."""
    REQUIRED = ("context",)
    OPTIONAL = ()
    RETURNS = ("context", "model", "clip", "prompt", "lora_triggers", "guidance",)


class TT_ContextExtractGuidedSamplerFlux2Node(ContextNode):
    """Extracts latent, guider, sigmas, sampler, and noise from context for use with a FLUX2 guided sampler."""
    REQUIRED = ("context",)
    OPTIONAL = ()
    RETURNS = ("context", "latent", "guider", "sigmas", "sampler", "rnd_noise",)


class TT_ContextExtractVaeNode(ContextNode):
    """Extracts VAE from context for use with a VAE decoder or encoder node."""
    REQUIRED = ("context",)
    OPTIONAL = ()
    RETURNS = ("context", "vae",)


class TT_ContextExtractImageNode(ContextNode):
    """Extracts Image from context for use with e.g. preview or postproduction node."""
    REQUIRED = ("context",)
    OPTIONAL = ()
    RETURNS = ("context", "image",)


# ==============================================================================
# V3 entrypoint — registers context nodes with ComfyUI
# ==============================================================================

class ContextNodesExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[IO.ComfyNode]]:
        return [
            TT_BaseContextNode,
            TT_BaseContextFlux2Node,
            TT_BaseContextPassthroughNode,
            TT_ContextNode,
            TT_ContextFlux2Node,
            TT_ContextSetGuiderNode,
            TT_ContextSetImageNode,
            TT_ContextSetLatentNode,
            TT_ContextExtractEncoderFlux2Node,
            TT_ContextExtractGuidedSamplerFlux2Node,
            TT_ContextExtractVaeNode,
            TT_ContextExtractImageNode,
        ]


async def comfy_entrypoint() -> ContextNodesExtension:
    return ContextNodesExtension()


# ==============================================================================
# Re-exports for backward compatibility
# ==============================================================================

__all__ = [
    "TT_BaseContextNode",
    "TT_BaseContextFlux2Node",
    "TT_BaseContextPassthroughNode",
    "TT_ContextNode",
    "TT_ContextFlux2Node",
    "TT_ContextSetGuiderNode",
    "TT_ContextSetImageNode",
    "TT_ContextSetLatentNode",
    "TT_ContextExtractEncoderFlux2Node",
    "TT_ContextExtractGuidedSamplerFlux2Node",
    "TT_ContextExtractVaeNode",
    "TT_ContextExtractImageNode",
]
