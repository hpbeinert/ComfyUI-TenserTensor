# (c) TenserTensor <tenser.tensor@proton.me> || Apache-2.0 (apache.org/licenses/LICENSE-2.0)

RESET = "\033[0m"
BOLD = "\033[1m"
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
CYAN = "\033[36m"
WHITE = "\033[37m"
BRIGHT_GREEN = "\033[92m"
BRIGHT_CYAN = "\033[96m"

import inspect
import sys

try:
    from .nodes_context import *
    from .nodes.context.base_context import TT_BaseContext
    from .nodes.context.context import TT_Context
    from .nodes.context.context_passthrough import TT_ContextPassthrough
    from .nodes.context.context_set_guider import TT_ContextSetGuider
    from .nodes.context.context_set_image import TT_ContextSetImage
    from .nodes.context.context_set_latent import TT_ContextSetLatent
    from .nodes.context.even_larger_context import TT_EvenLargerContext
    from .nodes.context.large_context_flux import TT_LargeContextFlux
    from .nodes.context.large_context_sdxl import TT_LargeContextSdxl
except ImportError:
    print(f"{YELLOW}TenserTensor: {RED}ERROR: Context nodes unavailable{RESET}")

try:
    from .nodes_detector import *
except ImportError:
    print(f"{YELLOW}TenserTensor: {RED}ERROR: Detector nodes unavailable{RESET}")

try:
    from .nodes.image.image_preview_save import TT_ImagePreviewSave
    from .nodes.image.image_preview_upscale_save import TT_ImagePreviewUpscaleSave
    from .nodes.image.guider_image_reference import TT_GuiderImageReference
    from .nodes_image import *
except ImportError:
    print(f"{YELLOW}TenserTensor: {RED}ERROR: Image nodes unavailable{RESET}")

try:
    from .nodes_controlnet import *
except ImportError:
    print(f"{YELLOW}TenserTensor: {RED}ERROR: ControlNet nodes unavailable{RESET}")

try:
    from .nodes.latent.latent_factory import TT_LatentFactory
    from .nodes_latent import *
except ImportError:
    print(f"{YELLOW}TenserTensor: {RED}ERROR: Latent nodes unavailable{RESET}")

try:
    from .nodes.loaders.flux_models_loader import TT_FluxModelsLoader
    from .nodes.loaders.flux_models_loader_advanced import TT_FluxModelsLoaderAdvanced
    from .nodes.loaders.gguf_models_loader import TT_Flux2GgufModelsLoader
    from .nodes.loaders.gguf_models_loader_advanced import TT_Flux2GgufModelsLoaderAdvanced
    from .nodes.loaders.sdxl_models_loader import TT_SdxlModelsLoader
    from .nodes.loaders.sdxl_models_loader_advanced import TT_SdxlModelsLoaderAdvanced
except ImportError:
    print(f"{YELLOW}TenserTensor: {RED}ERROR: Loader nodes unavailable{RESET}")

try:
    before = set(dir())
    from .nodes.postproduction.add_film_grain import TT_AddFilmGrain
    from .nodes.postproduction.apply_lut import TT_ApplyLut
    from .nodes.postproduction.image_enhancer import TT_ImageEnhancer
    from .nodes.postproduction.postproduction import TT_Postproduction
    from .nodes.postproduction.postproduction_advanced import TT_PostproductionAdvanced
    from .nodes.postproduction.quick_image_upscaler import TT_QuickImageUpscaler
except ImportError:
    print(f"{YELLOW}TenserTensor: {RED}ERROR: Postproduction nodes unavailable{RESET}")

try:
    from .nodes.sampling.ksampler import TT_KSampler
    from .nodes.sampling.ksampler_advanced import TT_KSamplerAdvanced
    from .nodes.sampling.ksampler_context import TT_KSamplerContext
    from .nodes.sampling.ksampler_two_stage import TT_KSamplerTwoStage
    from .nodes.sampling.ksampler_guided import TT_KSamplerGuided
except ImportError:
    print("f{YELLOW}TenserTensor: {RED}ERROR: Sampler nodes unavailable{RESET}")

try:
    from .nodes.text_encoders.clip_text_encode_flux import TT_ClipTextEncodeFlux
    from .nodes.text_encoders.clip_text_encode_flux_context import TT_ClipTextEncodeFluxContext
    from .nodes.text_encoders.clip_text_encode_sdxl import TT_ClipTextEncodeSdxl
    from .nodes.text_encoders.clip_text_encode_sdxl_context import TT_ClipTextEncodeSdxlContext
    from .nodes.text_encoders.clip_text_encode_flux2 import TT_ClipTextEncodeFlux2
    from .nodes.text_encoders.clip_text_encode_flux2_context import TT_ClipTextEncodeFlux2Context
    from .nodes_text_encoder import *
except ImportError:
    print(f"{YELLOW}TenserTensor: {RED}ERROR: Text Encoder nodes unavailable{RESET}")

try:
    from .nodes.vae.vae_decode_context import TT_VaeDecodeContext
    from .nodes.vae.vae_decode_tiled import TT_VaeDecodeTiled
    from .nodes.vae.vae_encode_context import TT_VaeEncodeContext
    from .nodes.vae.vae_encode_tiled import TT_VaeEncodeTiled
except ImportError:
    print(f"{YELLOW}TenserTensor: {RED}ERROR: VAE nodes unavailable{RESET}")

try:
    before = set(dir())
    from .nodes.workflow.flux_workflow_settings import TT_FluxWorkflowSettings
    from .nodes.workflow.flux_workflow_settings_advanced import TT_FluxWorkflowSettingsAdvanced
    from .nodes.workflow.sdxl_workflow_settings import TT_SdxlWorkflowSettings
    from .nodes.workflow.sdxl_workflow_settings_advanced import TT_SdxlWorkflowSettingsAdvanced
    from .nodes.workflow.flux2_workflow_settings import TT_Flux2WorkflowSettings
    from .nodes.workflow.flux2_workflow_settings_advanced import TT_Flux2WorkflowSettingsAdvanced
    from .nodes_workflow import *
except ImportError:
    print(f"{YELLOW}TenserTensor: {RED}ERROR: Workflow nodes unavailable{RESET}")

ADDED = {
    name
    for name, obj
    in inspect.getmembers(sys.modules[__name__], inspect.isclass)
    if name.startswith("TT_")
}
NODES_COUNT = len(ADDED)

NODE_CLASS_MAPPINGS = {
    # Context Deprecated
    "TT_BaseContext": TT_BaseContext,
    "TT_Context": TT_Context,
    "TT_LargeContextFlux": TT_LargeContextFlux,
    "TT_LargeContextSdxl": TT_LargeContextSdxl,
    "TT_EvenLargerContext": TT_EvenLargerContext,
    "TT_ContextSetImage": TT_ContextSetImage,
    "TT_ContextSetLatent": TT_ContextSetLatent,
    "TT_ContextSetGuider": TT_ContextSetGuider,
    "TT_ContextPassthrough": TT_ContextPassthrough,
    # Context V3 Nodes
    "TT_BaseContextNode": TT_BaseContextNode,
    "TT_BaseContextFlux2Node": TT_BaseContextFlux2Node,
    "TT_ContextPassthroughNode": TT_ContextPassthroughNode,
    "TT_ContextNode": TT_ContextNode,
    "TT_ContextFlux2Node": TT_ContextFlux2Node,
    "TT_ContextSetGuiderNode": TT_ContextSetGuiderNode,
    "TT_ContextSetImageNode": TT_ContextSetImageNode,
    "TT_ContextSetLatentNode": TT_ContextSetLatentNode,
    "TT_ContextExtractEncoderFlux2Node": TT_ContextExtractEncoderFlux2Node,
    "TT_ContextExtractGuidedSamplerFlux2Node": TT_ContextExtractGuidedSamplerFlux2Node,
    "TT_ContextExtractVaeNode": TT_ContextExtractVaeNode,
    "TT_ContextExtractImageNode": TT_ContextExtractImageNode,
    # ControlNet V3 Nodes
    "TT_Flux2ApplyControlNetNode": TT_Flux2ApplyControlNetNode,
    "TT_Flux2ApplyControlNetAdvancedNode": TT_Flux2ApplyControlNetAdvancedNode,
    # Detector V3 Nodes
    "TT_CannyEdgeDetectorNode": TT_CannyEdgeDetectorNode,
    # Image
    "TT_ImagePreviewSave": TT_ImagePreviewSave,
    "TT_ImagePreviewUpscaleSave": TT_ImagePreviewUpscaleSave,
    "TT_GuiderImageReference": TT_GuiderImageReference,
    # Image V3 Nodes
    "TT_ImageLoaderResizerNode": TT_ImageLoaderResizerNode,
    "TT_ImagePreviewSaveNode": TT_ImagePreviewSaveNode,
    "TT_ImagePreviewUpscaleSaveNode": TT_ImagePreviewUpscaleSaveNode,
    "TT_GuiderImageReferenceNode": TT_GuiderImageReferenceNode,
    # Latent
    "TT_LatentFactory": TT_LatentFactory,
    # Latent V3 Nodes
    "TT_LatentFactoryNode": TT_LatentFactoryNode,
    "TT_LatentMultiTransformNode": TT_LatentMultiTransformNode,
    "TT_LatentMultiTransformOnPixelSpaceNode": TT_LatentMultiTransformOnPixelSpaceNode,
    # Loaders
    "TT_SdxlModelsLoader": TT_SdxlModelsLoader,
    "TT_SdxlModelsLoaderAdvanced": TT_SdxlModelsLoaderAdvanced,
    "TT_Flux2GgufModelsLoader": TT_Flux2GgufModelsLoader,
    "TT_Flux2GgufModelsLoaderAdvanced": TT_Flux2GgufModelsLoaderAdvanced,
    "TT_FluxModelsLoader": TT_FluxModelsLoader,
    "TT_FluxModelsLoaderAdvanced": TT_FluxModelsLoaderAdvanced,
    # Postproduction
    "TT_ApplyLut": TT_ApplyLut,
    "TT_AddFilmGrain": TT_AddFilmGrain,
    "TT_QuickImageUpscaler": TT_QuickImageUpscaler,
    "TT_ImageEnhancer": TT_ImageEnhancer,
    "TT_Postproduction": TT_Postproduction,
    "TT_PostproductionAdvanced": TT_PostproductionAdvanced,
    # Samplers
    "TT_KSampler": TT_KSampler,
    "TT_KSamplerAdvanced": TT_KSamplerAdvanced,
    "TT_KSamplerContext": TT_KSamplerContext,
    "TT_KSamplerTwoStage": TT_KSamplerTwoStage,
    "TT_KSamplerGuided": TT_KSamplerGuided,
    # Text Encoder
    "TT_ClipTextEncodeFlux": TT_ClipTextEncodeFlux,
    "TT_ClipTextEncodeFluxContext": TT_ClipTextEncodeFluxContext,
    "TT_ClipTextEncodeSdxl": TT_ClipTextEncodeSdxl,
    "TT_ClipTextEncodeSdxlContext": TT_ClipTextEncodeSdxlContext,
    "TT_ClipTextEncodeFlux2": TT_ClipTextEncodeFlux2,
    "TT_ClipTextEncodeFlux2Context": TT_ClipTextEncodeFlux2Context,
    # Text Encoder V3 Nodes
    "TT_Flux2TextEncoderNode": TT_Flux2TextEncoderNode,
    "TT_Flux2TextEncoderContextNode": TT_Flux2TextEncoderContextNode,
    # VAE
    "TT_VaeDecodeContext": TT_VaeDecodeContext,
    "TT_VaeDecodeTiled": TT_VaeDecodeTiled,
    "TT_VaeEncodeContext": TT_VaeEncodeContext,
    "TT_VaeEncodeTiled": TT_VaeEncodeTiled,
    # Workflow
    "TT_FluxWorkflowSettings": TT_FluxWorkflowSettings,
    "TT_FluxWorkflowSettingsAdvanced": TT_FluxWorkflowSettingsAdvanced,
    "TT_SdxlWorkflowSettings": TT_SdxlWorkflowSettings,
    "TT_SdxlWorkflowSettingsAdvanced": TT_SdxlWorkflowSettingsAdvanced,
    "TT_Flux2WorkflowSettings": TT_Flux2WorkflowSettings,
    "TT_Flux2WorkflowSettingsAdvanced": TT_Flux2WorkflowSettingsAdvanced,
    # Workflow V3 Nodes
    "TT_Flux2WorkflowSettingsNode": TT_Flux2WorkflowSettingsNode,
    "TT_Flux2WorkflowSettingsAdvancedNode": TT_Flux2WorkflowSettingsAdvancedNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    # Context Deprecated
    "TT_BaseContext": "TT Base Context (Deprecated)",
    "TT_Context": "TT Context  (Deprecated)",
    "TT_LargeContextFlux": "TT FLUX Large Context (Deprecated)",
    "TT_LargeContextSdxl": "TT SDXL Large Context (Deprecated)",
    "TT_EvenLargerContext": "TT Even Larger Context (Deprecated)",
    "TT_ContextSetImage": "TT Context Set Image (Deprecated)",
    "TT_ContextSetLatent": "TT Context Set Latent (Deprecated)",
    "TT_ContextSetGuider": "TT Context Set Guider (Deprecated)",
    "TT_ContextPassthrough": "TT Context Passthrough (Deprecated)",
    # Context V3 Nodes
    "TT_BaseContextNode": "TT Base Context",
    "TT_BaseContextFlux2Node": "TT FLUX2 Base Context",
    "TT_ContextPassthroughNode": "TT Context Passthrough",
    "TT_ContextNode": "TT Context",
    "TT_ContextFlux2Node": "TT FLUX2 Context",
    "TT_ContextSetGuiderNode": "TT Context Set Guider",
    "TT_ContextSetImageNode": "TT Context Set Image",
    "TT_ContextSetLatentNode": "TT Context Set Latent",
    "TT_ContextExtractEncoderFlux2Node": "TT Context Extract Encoder FLUX2",
    "TT_ContextExtractGuidedSamplerFlux2Node": "TT Context Extract Guided Sampler Flux2",
    "TT_ContextExtractVaeNode": "TT Context Extract VAE Flux2",
    "TT_ContextExtractImageNode": "TT Context Extract Image Flux2",
    # ControlNet V3 Nodes
    "TT_Flux2ApplyControlNetNode": "TT Flux2 Apply ControlNet",
    "TT_Flux2ApplyControlNetAdvancedNode": "TT FLUX2 Apply ControlNet (Advanced)",
    # Detector V3 Nodes
    "TT_CannyEdgeDetectorNode": "TT_Canny Edge Detector",
    # Image
    "TT_ImagePreviewSave": "TT Image Preview / Save  (Deprecated)",
    "TT_ImagePreviewUpscaleSave": "TT Image Preview / Upscale / Save  (Deprecated)",
    "TT_GuiderImageReference": "TT Guider Image Reference  (Deprecated)",
    # Image V3 Nodes
    "TT_ImageLoaderResizerNode": "TT Image Loader / Resizer",
    "TT_ImagePreviewSaveNode": "TT Image Preview / Save",
    "TT_ImagePreviewUpscaleSaveNode": "TT Image Preview / Upscale / Save",
    "TT_GuiderImageReferenceNode": "TT Guider Image Reference",
    # Latent
    "TT_LatentFactory": "TT Latent Factory (Deprecated)",
    # Latent V3 Nodes
    "TT_LatentFactoryNode": "TT Latent Factory",
    "TT_LatentMultiTransformNode": "TT Latent Multi Transform",
    "TT_LatentMultiTransformOnPixelSpaceNode": "TT Latent MultiTransform On Pixel Space",
    # Loaders
    "TT_SdxlModelsLoader": "TT SDXL Models Loader",
    "TT_SdxlModelsLoaderAdvanced": "TT SDXL Models Loader (Advanced)",
    "TT_Flux2GgufModelsLoader": "TT FLUX2 GGUF Models Loader",
    "TT_Flux2GgufModelsLoaderAdvanced": "TT FLUX2 GGUF Models Loader (Advanced)",
    "TT_FluxModelsLoader": "TT FLUX Models Loader",
    "TT_FluxModelsLoaderAdvanced": "TT FLUX Models Loader (Advanced)",
    # Postproduction
    "TT_ApplyLut": "TT Apply LUT",
    "TT_AddFilmGrain": "TT Add Film Grain",
    "TT_QuickImageUpscaler": "TT Quick Image Upscaler",
    "TT_ImageEnhancer": "TT Image Enhancer",
    "TT_Postproduction": "TT Postproduction",
    "TT_PostproductionAdvanced": "TT Postproduction (Advanced)",
    # Samplers
    "TT_KSampler": "TT KSampler",
    "TT_KSamplerAdvanced": "TT KSampler (Advanced)",
    "TT_KSamplerContext": "TT KSampler (Context)",
    "TT_KSamplerTwoStage": "TT KSampler (Two Stages)",
    "TT_KSamplerGuided": "TT KSampler (Guided)",
    # Text Encoder
    "TT_ClipTextEncodeFlux": "TT CLIP Text Encode FLUX",
    "TT_ClipTextEncodeFluxContext": "TT CLIP Text Encode FLUX (Context)",
    "TT_ClipTextEncodeSdxl": "TT CLIP Text Encode SDXL",
    "TT_ClipTextEncodeSdxlContext": "TT CLIP Text Encode SDXL (Context)",
    "TT_ClipTextEncodeFlux2": "TT CLIP Text Encode FLUX2",
    "TT_ClipTextEncodeFlux2Context": "TT CLIP Text Encode FLUX2 (Context)",
    # Text Encoder V3 Nodes
    "TT_Flux2TextEncoderNode":"",
    "TT_Flux2TextEncoderContextNode":"",
    # VAE
    "TT_VaeDecodeContext": "TT VAE Decode (Context)",
    "TT_VaeDecodeTiled": "TT VAE Decode (Tiled)",
    "TT_VaeEncodeContext": "TT Vae Encode (Context)",
    "TT_VaeEncodeTiled": "TT Vae Encode (Tiled)",
    # Workflow
    "TT_FluxWorkflowSettings": "TT FLUX Workflow Settings",
    "TT_FluxWorkflowSettingsAdvanced": "TT FLUX Workflow Settings (Advanced)",
    "TT_SdxlWorkflowSettings": "TT SDXL Workflow Settings",
    "TT_SdxlWorkflowSettingsAdvanced": "TT SDXL Workflow Settings (Advanced)",
    "TT_Flux2WorkflowSettings": "TT FLUX2 Workflow Settings (Deprecated)",
    "TT_Flux2WorkflowSettingsAdvanced": "TT FLUX2 Workflow Settings (Deprecated/Advanced)",
    # Workflow V3 Nodes
    "TT_Flux2WorkflowSettingsNode": "TT FLUX2 Workflow Settings",
    "TT_Flux2WorkflowSettingsAdvancedNode": "TT FLUX2 Workflow Settings (Advanced)",
}
