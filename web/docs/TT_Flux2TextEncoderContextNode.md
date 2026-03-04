## TT FLUX2 Text Encoder (Context) *TT_Flux2TextEncoderContextNode*

*This description was generated with AI assistance. If you spot any errors, please create an issue on GitHub.*

Context-based variant of the FLUX2 text encoder. Reads all required parameters from a Context object and writes the resulting Guider back into it.

### Inputs

| Parameter | Type       | Required | Description                                                                    |
|-----------|------------|----------|--------------------------------------------------------------------------------|
| `context` | TT_CONTEXT | Yes      | Context object carrying model, CLIP, prompt, LoRA triggers, and guidance scale |

### Outputs

| Parameter | Type       | Description                                 |
|-----------|------------|---------------------------------------------|
| `CONTEXT` | TT_CONTEXT | Updated Context with Guider stored inside   |
| `GUIDER`  | GUIDER     | Configured Guider with encoded conditioning |

### Usage

Drop-in replacement for [TT FLUX2 Text Encoder](TT_Flux2TextEncoderNode.md) in Context-based workflows. The Context must have `model`, `clip`,
`prompt`, `lora_triggers`, and `guidance` set before reaching this node.

---

[View on GitHub](https://github.com/tenser-tensor/ComfyUI-TenserTensor)
