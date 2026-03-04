## TT FLUX2 Text Encoder *TT_Flux2TextEncoderNode*

*This description was generated with AI assistance. If you spot any errors, please create an issue on GitHub.*

Encodes text prompts for FLUX2 models and returns a ready-to-use Guider object. Combines prompt and LoRA trigger words into a single conditioning
step.

### Inputs

| Parameter       | Type   | Required | Description                                |
|-----------------|--------|----------|--------------------------------------------|
| `model`         | MODEL  | Yes      | Diffusion model                            |
| `clip`          | CLIP   | Yes      | CLIP model for text encoding               |
| `prompt`        | STRING | Yes      | Main generation prompt                     |
| `lora_triggers` | STRING | No       | LoRA trigger words, appended to the prompt |
| `guidance`      | FLOAT  | Yes      | Guidance scale (CFG), default 3.5          |

### Outputs

| Parameter | Type   | Description                                 |
|-----------|--------|---------------------------------------------|
| `GUIDER`  | GUIDER | Configured Guider with encoded conditioning |

### Usage

Connect to a sampler node that accepts a Guider input. The `lora_triggers` field is a convenience input — enter trigger words for any loaded LoRA
models here instead of adding them manually to the main prompt.

---

[View on GitHub](https://github.com/tenser-tensor/ComfyUI-TenserTensor)
