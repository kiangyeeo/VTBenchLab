# Reliability-gated loss proxy

Reliability threshold: `0.200`. The gate does not use MLLM labels.

| Domain | Median split-half | Selected |
|---|---:|---|
| caption | 0.8142 | yes |
| ocr | 0.0343 | no |
| reasoning | 0.5735 | yes |
| vqa | -0.2584 | no |

| Tokenizer | Mean rank |
|---|---:|
| pe_lang_g14_448 | 1.0000 |
| pe_core_g14_448 | 2.0000 |

Predicted: `pe_lang_g14_448 > pe_core_g14_448`

Expected: `pe_lang_g14_448 > pe_core_g14_448`

Exact match: `yes`; Spearman: `1.0000`.
