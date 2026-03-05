## TT SDXL CLIP Text Encoder *TT_SdxlClipTextEncoderNode*

*This description was generated with AI assistance. If you spot any errors, please create an issue on GitHub.*

Encodes positive and negative text prompts for SDXL models using dual CLIP encoders (CLIP-L and CLIP-G) with aesthetic score and resolution
conditioning.

### Inputs

| Parameter         | Type   | Required | Description                                            |
|-------------------|--------|----------|--------------------------------------------------------|
| `clip`            | CLIP   | Yes      | Dual CLIP model (CLIP-L + CLIP-G)                      |
| `clip_l_positive` | STRING | Yes      | CLIP-L positive prompt                                 |
| `clip_g_positive` | STRING | Yes      | CLIP-G positive prompt                                 |
| `clip_l_negative` | STRING | Yes      | CLIP-L negative prompt                                 |
| `clip_g_negative` | STRING | Yes      | CLIP-G negative prompt                                 |
| `ascore_positive` | FLOAT  | Yes      | Aesthetic score for positive conditioning, default 9.0 |
| `ascore_negative` | FLOAT  | Yes      | Aesthetic score for negative conditioning, default 6.0 |
| `width`           | INT    | Yes      | Source image width                                     |
| `height`          | INT    | Yes      | Source image height                                    |
| `target_width`    | INT    | Yes      | Target output width                                    |
| `target_height`   | INT    | Yes      | Target output height                                   |

### Outputs

| Parameter  | Type         | Description           |
|------------|--------------|-----------------------|
| `POSITIVE` | CONDITIONING | Positive conditioning |
| `NEGATIVE` | CONDITIONING | Negative conditioning |

### Usage

Connect `POSITIVE` and `NEGATIVE` outputs to a sampler. Set `width`/`height` to match your source image dimensions and `target_width`/`target_height`
to your desired output resolution. Higher `ascore_positive` (8–10) steers generation toward higher aesthetic quality; lower `ascore_negative` (4–6)
pushes away from low-quality outputs.

---

[View on GitHub](https://github.com/tenser-tensor/ComfyUI-TenserTensor)
