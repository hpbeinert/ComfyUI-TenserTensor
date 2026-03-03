## Nodes List

### Workflow

#### TT FLUX Workflow Settings

Node for configuring Flux workflow generation parameters.

#### TT FLUX Workflow Settings (Advanced)

Advanced node for configuring Flux workflow generation parameters with separate CLIP-L and T5XXL prompts.

#### TT Flux2 Workflow Settings

Use this node as the single source of truth for FLUX2 workflow parameters. The `WORKFLOW_CONFIG` output is
designed for context-aware nodes that consume a TT_WORKFLOW_CONFIG directly. Individual outputs are available
for connecting to standard ComfyUI nodes. The sigma schedule (`SCHEDULER`) is computed automatically from
the seed and resolution via `get_schedule`.

#### TT Flux2 Workflow Settings (Advanced)

Advanced version of TT Flux2 Workflow Settings with `prompt` and `lora_triggers` fields included directly in the
node. Both are stored in the `WORKFLOW_CONFIG` output alongside all other sampling parameters, making this node
sufficient as the sole configuration source for full context-driven FLUX2 pipelines. Individual outputs remain
available for connecting to standard ComfyUI nodes.

#### TT SDXL Workflow Settings

Node for configuring SDXL workflow generation parameters.

#### TT SDXL Workflow Settings (Advanced)

Advanced SDXL workflow configuration with separate CLIP-L/G prompts and aesthetic scores.

### Latent

#### TT Latent Factory

Select aspect ratio, resolution, and orientation to automatically compute pixel dimensions suited for the chosen
model. FLUX1.D uses 16-channel latents, FLUX2.D uses 128-channel, SDXL uses 4-channel. `clip_multiplier` scales the TARGET
dimensions relative to the image dimensions — useful for multi-resolution CLIP conditioning. Connect `RANDOM_NOISE`
directly to a sampler node for reproducible noise tied to `noise_seed`.

### Loaders

#### TT SDXL Models Loader

All-in-one model loader for SDXL workflows with checkpoint merging support.

#### TT SDXL Models Loader (Advanced)

Extended all-in-one model loader for SDXL with checkpoint merging and LoRA support.

#### TT GGUF Models Loader

All-in-one loader for GGUF-quantized models. Place your `.gguf` diffusion model files in the `diffusion_models` or `unet` directory, CLIP files in `text_encoders` or `clip` directory. VAE can be any standard ComfyUI-compatible VAE file.

#### TT GGUF Models Loader (Advanced)

Extended version of GGUF Models Loader. Supports up to 4 LoRAs loaded simultaneously. Enable `apply_sampling` to apply flux sampling shift patch — useful for improving quality at non-standard resolutions. Advanced parameters are hidden by default and can be revealed via the node's show/hide controls.

#### TT FLUX Models Loader

All-in-one model loader for Flux workflows.

#### TT FLUX Models Loader (Advanced)

Extended all-in-one model loader for Flux with sampling patches, LoRA support, and advanced settings.

### Text Encoder

#### TT CLIP Text Encode FLUX

Flux text encoder with separate CLIP-L and T5-XXL prompt inputs.

#### TT CLIP Text Encode FLUX (Context)

Context-based Flux text encoder that extracts prompts from workflow config.

#### TT CLIP Text Encode SDXL

SDXL text encoder with separate CLIP-L and CLIP-G prompt inputs and conditioning parameters.

#### TT CLIP Text Encode SDXL (Context)

Context-based SDXL text encoder that extracts prompts and parameters from workflow config.

#### TT Clip Text Encode Flux2

Encodes text prompts specifically for FLUX2 models. Enter your main prompt in `prompt` and any LoRA-specific trigger
words separately in `lora_triggers` — the node combines them internally. Adjust `guidance` to control how strongly
the model follows the prompt (3.5 is a good starting point for FLUX2).

#### TT Clip Text Encode Flux2 (Context)

Context-aware version of the FLUX2 text encoder. Instead of individual inputs, it reads `model`, `clip`, and
`workflow_config` (containing `prompt`, `lora_triggers`, and `guidance`) from the passed `context`. The resulting
guider is both returned as an output and stored back into the context under the `guider` key. Raises an error if
any of the required fields are missing from the context.

### Samplers

#### TT KSampler

Standard KSampler for generating images. Accepts all common sampling parameters and returns a denoised latent ready for
VAE decoding.

#### TT KSampler (Advanced)

Advanced sampler for multi-stage workflows and fine control. Use start_step/last_step for partial denoising, add_noise
to control noise injection, and denoise for strength control. Useful for img2img, refinement passes, and complex
sampling workflows.

#### TT KSampler (Context)

Simplified sampler for context-based workflows. Automatically extracts all sampling parameters from the context object:

- Model, positive/negative conditioning, and latent from context
- Seed, steps, cfg, sampler_name, and scheduler from workflow_config

Requires context to contain: model, latent, and workflow_config. Raises error if any required component is missing.

#### TT KSampler (Two Stages)

Automated two-stage sampling workflow. First performs a draft pass with partial denoising, then refines the result with
a second pass. Each stage can use different samplers, schedulers, and step counts. Useful for quick previews followed by
quality refinement, or combining different sampling strategies.

#### TT KSampler (Guided)

Runs guided diffusion sampling on the input latent. Connect a guider from a text encoder node, sigmas from a
scheduler, and a sampler from a workflow settings node. `random_noise` controls stochasticity — use a seeded
noise node for reproducible results.

### VAE

#### TT VAE Decode (Context)

Simplified decoder for context-based workflows. Automatically extracts VAE and latent from context, decodes to image
space, and stores the result back in context. Handles nested tensors and reshapes output to standard format. Raises
error if VAE or latent is missing from context.

#### TT VAE Decode (Tiled)

Decode large latent images by processing them in smaller tiles. Useful for generating high-resolution images that would
otherwise exceed VRAM capacity. The overlap parameter helps reduce visible seams between tiles. Enable circular padding
for seamless texture generation.

#### TT Vae Encode (Context)

Encodes pixel-space images into latent representations for diffusion processing. Extracts VAE and image from context,
performs encoding, and stores the result back in context.

Requires context to contain both `vae` and `image` components. Raises error if either is missing.

#### TT Vae Encode (Tiled)

Processes large images in smaller tiles to reduce memory requirements during VAE encoding. Useful for encoding
high-resolution images that would otherwise cause out-of-memory errors.

Tile size should be set based on available VRAM (smaller tiles = less memory). Overlap prevents visible seams at tile
boundaries - higher overlap produces smoother results but increases processing time. Recommended overlap is 10-15% of
tile size.

### Postproduction

#### TT Apply LUT

Apply professional color grading using LUT files. Place your .cube LUT files in the designated LUT directory. Adjust
strength to blend between original and graded image (1.0 = full LUT, 0.0 = original). Choose colorspace based on your
LUT file's intended working space.

#### TT Add Film Grain

Add realistic film grain texture to images for a cinematic aesthetic. Adjust scale for grain size, strength for
intensity, and saturation for color. The toe parameter controls the distribution curve of grain in shadows/highlights.
Use different seeds for varied grain patterns.

#### TT Quick Image Upscaler

Quick and simple AI upscaling node using tiled processing for memory efficiency. Select an upscale model from your
upscale_models directory. Adjust tile size based on VRAM - smaller tiles use less memory. Increase overlap to reduce
visible seams between tiles.

#### TT Image Enhancer

All-in-one image enhancement node for adjusting brightness, contrast, colors, and applying effects. Use default values (
brightness: 0.0, contrast/gamma/saturation: 1.0, posterize: 8, solarize: 1.0) for no effect. Adjust individual
parameters as needed - effects are applied sequentially.

#### TT Postproduction

Complete post-production pipeline in one node. Toggle each effect on/off with boolean switches. Processing order: LUT →
Film Grain → Upscale. Disable any effect by setting its toggle to False. All effects are optional and can be used
independently or combined.

#### TT Postproduction (Advanced)

Complete post-production pipeline with full enhancement controls. Processing order: Image Enhancement → LUT → Film
Grain → Upscale. Toggle each effect on/off independently. Image enhancement includes brightness, contrast, gamma, hue,
saturation, sharpness, posterize, and solarize adjustments. All effects are optional.

### Image

#### TT Image Preview / Save

Dual-purpose node that can preview images in the UI and/or save them to disk with extensive formatting options.

#### TT Image Preview / Upscale / Save

All-in-one node that can preview, upscale, and save images in a single step. Upscaling is applied before saving — the upscaled result is both displayed in the UI and written to disk. When `upscale_image` is disabled, the original image is used. When `save_image` is disabled, the image is only previewed in the UI without writing to disk.

### Context

#### TT Base Context

Use this node to package your core workflow components into a single context object that can be passed through your
workflow and modified by other context nodes.

#### TT Context

Use this node to create a new context from scratch or modify specific parameters in an existing context. All parameters
are optional - provide only what you need to set or change.

#### TT FLUX Large Context

Use this node for Flux workflows when you need fine-grained control over CLIP-L and T5-XXL prompts separately, along
with Flux-specific guidance settings.

#### TT SDXL Large Context

Use this node for SDXL workflows when you need separate control over CLIP-L and CLIP-G prompts, aesthetic scores, and
target dimensions for conditioning.

#### TT Even Larger Context

Use this node when you need comprehensive control over all workflow parameters including ControlNet and masking.
Supports both SDXL (CLIP-L/G) and Flux (T5-XXL) conditioning simultaneously.

#### TT Context Set Image

Stores pixel-space image in context for use by downstream nodes. Overwrites existing image if present.

#### TT Context Set Latent

Stores latent representation in context for use by downstream nodes. Overwrites existing latent if present.

#### TT Context Set Guider

Sets the `guider` field inside a TT_CONTEXT object. Use this node when you have a guider produced outside the
context pipeline and need to inject it back in for use by subsequent context-aware nodes.

#### TT Context Passthrough

Use this node to pass context through your workflow while optionally modifying specific parameters without changing the
rest of the context.

---

_(c) TenserTensor <tenser.tensor@proton.me> || Apache-2.0 (apache.org/licenses/LICENSE-2.0)_
