# PE pair gradient compatibility

Primary scores use real-minus-shuffled LoRA-B gradients. The unweighted mean is
reported as exploratory because two tokenizers are insufficient for fitting weights.

| Tokenizer | Seed | Domain | Delta alignment | 95% CI | Raw alignment | Visual signal | Split-half |
|---|---:|---|---:|---:|---:|---:|---:|
| pe_lang_g14_448 | 0 | caption | 0.1274 | [0.1188, 0.1359] | 0.0306 | 0.8304 | 0.7150 |
| pe_lang_g14_448 | 0 | vqa | -0.0075 | [-0.0099, -0.0051] | 0.0070 | 0.4400 | -0.0165 |
| pe_lang_g14_448 | 0 | ocr | 0.0057 | [0.0023, 0.0089] | -0.0088 | 0.3699 | 0.0308 |
| pe_lang_g14_448 | 0 | reasoning | -0.0056 | [-0.0083, -0.0029] | -0.0053 | 0.1898 | 0.4856 |
| pe_core_g14_448 | 0 | caption | 0.2475 | [0.2300, 0.2641] | 0.1053 | 1.1123 | 0.9134 |
| pe_core_g14_448 | 0 | vqa | 0.0373 | [0.0299, 0.0450] | 0.0176 | 0.6751 | -0.5003 |
| pe_core_g14_448 | 0 | ocr | 0.1137 | [0.1017, 0.1254] | 0.0001 | 0.6378 | 0.0379 |
| pe_core_g14_448 | 0 | reasoning | 0.0984 | [0.0860, 0.1102] | -0.0363 | 0.6367 | 0.6613 |

## Exploratory aggregate

| Tokenizer | Seed | Mean delta alignment | Mean visual signal |
|---|---:|---:|---:|
| pe_lang_g14_448 | 0 | 0.0300 | 0.4575 |
| pe_core_g14_448 | 0 | 0.1242 | 0.7655 |

## Pairwise reversal check

The known Qwen2.5 MLLM difference is `pe_lang_g14_448 - pe_core_g14_448 = 6.15`.

| Seed | Domain | Alignment difference | Recovers MLLM direction |
|---:|---|---:|---|
| 0 | caption | -0.1201 | no |
| 0 | vqa | -0.0447 | no |
| 0 | ocr | -0.1080 | no |
| 0 | reasoning | -0.1040 | no |
| 0 | exploratory_mean | -0.0942 | no |
