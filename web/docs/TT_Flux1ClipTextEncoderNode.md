## TT FLUX1 CLIP Text Encoder *TT_Flux1ClipTextEncoderNode*

*This description was generated with AI assistance. If you spot any errors, please create an issue on GitHub.*

Encodes positive and negative text prompts for FLUX1 models using dual text encoders (CLIP-L and T5-XXL) with guidance scale conditioning.

### Inputs

| Parameter         | Type   | Required | Description                               |
|-------------------|--------|----------|-------------------------------------------|
| `clip`            | CLIP   | Yes      | Dual text encoder model (CLIP-L + T5-XXL) |
| `clip_l_positive` | STRING | Yes      | CLIP-L positive prompt                    |
| `t5xxl_positive`  | STRING | Yes      | T5-XXL positive prompt                    |
| `clip_l_negative` | STRING | Yes      | CLIP-L negative prompt                    |
| `t5xxl_negative`  | STRING | Yes      | T5-XXL negative prompt                    |
| `guidance`        | FLOAT  | Yes      | Guidance scale, default 9.0               |

### Outputs

| Parameter  | Type         | Description           |
|------------|--------------|-----------------------|
| `POSITIVE` | CONDITIONING | Positive conditioning |
| `NEGATIVE` | CONDITIONING | Negative conditioning |

### Usage

Connect `POSITIVE` and `NEGATIVE` outputs to a sampler. CLIP-L handles short, structured descriptions; T5-XXL understands longer, more complex
prompts — you can use the same text in both or tailor each separately.

---

[View on GitHub](https://github.com/tenser-tensor/ComfyUI-TenserTensor)
