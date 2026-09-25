# Reliability-gated loss proxy

Reliability threshold: `0.200`. The gate does not use MLLM labels.

| Domain | Median split-half | Selected |
|---|---:|---|
| caption | 0.7308 | yes |
| ocr | -0.0165 | no |
| reasoning | 0.2802 | yes |
| vqa | 0.0483 | no |

| Tokenizer | Mean rank |
|---|---:|
| siglip2_b16_256 | 2.0000 |
| mc1_b16_224_2.5b | 2.0000 |
| raev2 | 2.0000 |

Predicted: `siglip2_b16_256 = mc1_b16_224_2.5b = raev2`

Expected: `siglip2_b16_256 > mc1_b16_224_2.5b > raev2`

Exact match: `no`; Spearman: `nan`.
