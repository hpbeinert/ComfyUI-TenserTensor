## TT FLUX1 CLIP Text Encoder (Context) *TT_Flux1ClipTextEncoderContextNode*

*This description was generated with AI assistance. If you spot any errors, please create an issue on GitHub.*

Context-based variant of the FLUX1 CLIP text encoder. Reads all required parameters from a Context object and writes the resulting conditioning back
into it.

### Inputs

| Parameter | Type       | Required | Description                                               |
|-----------|------------|----------|-----------------------------------------------------------|
| `context` | TT_CONTEXT | Yes      | Context object carrying CLIP, prompts, and guidance scale |

### Outputs

| Parameter  | Type         | Description                                                           |
|------------|--------------|-----------------------------------------------------------------------|
| `CONTEXT`  | TT_CONTEXT   | Updated Context with positive and negative conditioning stored inside |
| `POSITIVE` | CONDITIONING | Positive conditioning                                                 |
| `NEGATIVE` | CONDITIONING | Negative conditioning                                                 |

### Usage

Drop-in replacement for [TT FLUX1 CLIP Text Encoder](TT_Flux1ClipTextEncoderNode.md) in Context-based workflows. The Context must have `clip`,
`clip_l_positive`, `t5xxl_positive`, `clip_l_negative`, `t5xxl_negative`, and `guidance` set before reaching this node.

---

[View on GitHub](https://github.com/tenser-tensor/ComfyUI-TenserTensor)
