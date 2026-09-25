# Tokenizer-pair gradient compatibility

Primary scores use real-minus-shuffled LoRA-B gradients. The unweighted mean is
reported as exploratory because two tokenizers are insufficient for fitting weights.

| Tokenizer | Seed | Domain | Delta alignment | 95% CI | Raw alignment | Visual signal | Split-half |
|---|---:|---|---:|---:|---:|---:|---:|
| siglip2_b16_256 | 0 | caption | 0.1508 | [0.1426, 0.1589] | 0.0230 | 1.0763 | 0.8035 |
| siglip2_b16_256 | 0 | vqa | -0.0226 | [-0.0265, -0.0186] | -0.0049 | 0.5408 | 0.1116 |
| siglip2_b16_256 | 0 | ocr | 0.0099 | [0.0061, 0.0136] | -0.0123 | 0.5126 | 0.1308 |
| siglip2_b16_256 | 0 | reasoning | 0.0008 | [-0.0010, 0.0026] | -0.0038 | 0.3320 | 0.2802 |
| mc1_b16_224_2.5b | 0 | caption | 0.1092 | [0.1002, 0.1187] | 0.0324 | 0.6541 | 0.6430 |
| mc1_b16_224_2.5b | 0 | vqa | -0.0010 | [-0.0033, 0.0015] | -0.0008 | 0.5098 | 0.0474 |
| mc1_b16_224_2.5b | 0 | ocr | -0.0069 | [-0.0117, -0.0021] | -0.0030 | 0.3793 | -0.1123 |
| mc1_b16_224_2.5b | 0 | reasoning | 0.0020 | [0.0007, 0.0032] | -0.0049 | 0.2182 | 0.1127 |
| raev2 | 0 | caption | 0.1341 | [0.1261, 0.1413] | 0.0239 | 1.0310 | 0.7308 |
| raev2 | 0 | vqa | 0.0041 | [0.0015, 0.0068] | -0.0006 | 0.6585 | 0.0483 |
| raev2 | 0 | ocr | 0.0045 | [0.0012, 0.0078] | -0.0020 | 0.6494 | -0.0165 |
| raev2 | 0 | reasoning | 0.0048 | [0.0022, 0.0076] | -0.0069 | 0.3830 | 0.2863 |

## Exploratory aggregate

| Tokenizer | Seed | Mean delta alignment | Mean visual signal |
|---|---:|---:|---:|
| siglip2_b16_256 | 0 | 0.0347 | 0.6154 |
| mc1_b16_224_2.5b | 0 | 0.0258 | 0.4403 |
| raev2 | 0 | 0.0369 | 0.6805 |

## MLLM ranking check

Expected Qwen2.5 order: `siglip2_b16_256 > mc1_b16_224_2.5b > raev2`.

| Seed | Domain | Predicted order | Exact match | Spearman |
|---:|---|---|---|---:|
| 0 | caption | siglip2_b16_256 > raev2 > mc1_b16_224_2.5b | no | 0.5000 |
| 0 | vqa | raev2 > mc1_b16_224_2.5b > siglip2_b16_256 | no | -1.0000 |
| 0 | ocr | siglip2_b16_256 > raev2 > mc1_b16_224_2.5b | no | 0.5000 |
| 0 | reasoning | raev2 > mc1_b16_224_2.5b > siglip2_b16_256 | no | -1.0000 |
| 0 | exploratory_mean | raev2 > siglip2_b16_256 > mc1_b16_224_2.5b | no | -0.5000 |
