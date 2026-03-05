## TT SDXL CLIP Text Encoder (Context) *TT_SdxlClipTextEncoderContextNode*

*This description was generated with AI assistance. If you spot any errors, please create an issue on GitHub.*

Context-based variant of the SDXL CLIP text encoder. Reads all required parameters from a Context object and writes the resulting conditioning back
into it.

### Inputs

| Parameter | Type       | Required | Description                                                                        |
|-----------|------------|----------|------------------------------------------------------------------------------------|
| `context` | TT_CONTEXT | Yes      | Context object carrying CLIP, prompts, aesthetic scores, and resolution parameters |

### Outputs

| Parameter  | Type         | Description                                                           |
|------------|--------------|-----------------------------------------------------------------------|
| `CONTEXT`  | TT_CONTEXT   | Updated Context with positive and negative conditioning stored inside |
| `POSITIVE` | CONDITIONING | Positive conditioning                                                 |
| `NEGATIVE` | CONDITIONING | Negative conditioning                                                 |

### Usage

Drop-in replacement for [TT SDXL CLIP Text Encoder](TT_SdxlClipTextEncoderNode.md) in Context-based workflows. The Context must have `clip`,
`clip_l_positive`, `clip_g_positive`, `clip_l_negative`, `clip_g_negative`, `ascore_positive`, `ascore_negative`, `width`, `height`, `target_width`,
and `target_height` set before reaching this node.

---

[View on GitHub](https://github.com/tenser-tensor/ComfyUI-TenserTensor)
