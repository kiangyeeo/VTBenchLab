# PE pair gradient compatibility

Primary scores use real-minus-shuffled LoRA-B gradients. The unweighted mean is
reported as exploratory because two tokenizers are insufficient for fitting weights.

| Tokenizer | Seed | Domain | Delta alignment | 95% CI | Raw alignment | Visual signal | Split-half |
|---|---:|---|---:|---:|---:|---:|---:|
| pe_lang_g14_448 | 0 | caption | 0.0156 | [-0.0397, 0.0709] | 0.3123 | 0.3601 | nan |
| pe_lang_g14_448 | 0 | vqa | -0.0077 | [-0.0219, 0.0064] | 0.2283 | 0.4362 | nan |
| pe_lang_g14_448 | 0 | ocr | 0.0016 | [-0.0354, 0.0385] | 0.1440 | 0.3174 | nan |
| pe_lang_g14_448 | 0 | reasoning | -0.0049 | [-0.0076, -0.0022] | 0.0263 | 0.5080 | nan |
| pe_core_g14_448 | 0 | caption | -0.0716 | [-0.1992, 0.0560] | 0.3649 | 0.8486 | nan |
| pe_core_g14_448 | 0 | vqa | -0.0147 | [-0.0717, 0.0423] | 0.2407 | 1.2353 | nan |
| pe_core_g14_448 | 0 | ocr | 0.0265 | [-0.0967, 0.1497] | 0.2408 | 1.0245 | nan |
| pe_core_g14_448 | 0 | reasoning | 0.0089 | [-0.0094, 0.0272] | 0.1695 | 1.5713 | nan |

## Exploratory aggregate

| Tokenizer | Seed | Mean delta alignment | Mean visual signal |
|---|---:|---:|---:|
| pe_lang_g14_448 | 0 | 0.0011 | 0.4054 |
| pe_core_g14_448 | 0 | -0.0127 | 1.1699 |

## Pairwise reversal check

The known Qwen2.5 MLLM difference is `pe_lang_g14_448 - pe_core_g14_448 = 6.15`.

| Seed | Domain | Alignment difference | Recovers MLLM direction |
|---:|---|---:|---|
| 0 | caption | 0.0872 | yes |
| 0 | vqa | 0.0069 | yes |
| 0 | ocr | -0.0250 | no |
| 0 | reasoning | -0.0138 | no |
| 0 | exploratory_mean | 0.0139 | yes |
