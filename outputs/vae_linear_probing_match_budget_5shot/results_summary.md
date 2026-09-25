# ImageNet-1K 5-shot linear probing results

## Summary

- Models: **70/70 complete**.
- Training protocol: 5 images per class, 5,000 images total, batch size 1,000,
  exactly 5 optimizer updates (one epoch).
- Validation: complete 50,000-image ImageNet-1K validation split.
- Mean Top-1: **23.404%**.
- Mean Top-5: **39.639%**.
- Best Top-1: **65.260%**, `dinov2_giant`.
- Lowest Top-1: **0.350%**, `webssl_mae300m_full2b_224`.
- 65 of 70 models select the highest tested base LR (0.5), so the LR grid is
  boundary-limited for most models under this very short five-update schedule.

## Compute

The frozen encoder surface used by probing was profiled once per model with
PyTorch 2.4 `FlopCounterMode`, using each model's actual input resolution and
transform. Linear-head compute is not included; it is negligible and was also
outside the previously used 303P frozen-encoder budget.

- Mean per-image operator FLOPs (multiply and add counted separately):
  **367.337844 GFLOPs/image**.
- Mean 5-shot training compute in strict operator-FLOPs convention:
  **1.836689 PFLOPs/model**.
- Mean per-image compute in the convention used by the earlier 303P estimate
  (one multiply-accumulate counted as one operation):
  **183.668922 GMACs/image**.
- Mean 5-shot training compute in that same earlier convention:
  **0.918345 PMACs/model**, conventionally reported here as approximately
  **0.92 PFLOPs/model**.
- Total 70-model 5-shot training compute: **64.284123 PMACs**, or
  **128.568245 PFLOPs** when multiply and add are counted separately.
- Full validation is reported separately. Its average encoder compute is
  **9.183446 PMACs/model** (strict operator count: **18.366892 PFLOPs/model**).
- Training plus validation averages **10.101791 PMACs/model** (strict operator
  count: **20.203581 PFLOPs/model**).

As a consistency check, applying the same profiler to the original 42 models
and 1,280,000 training images gives **302.897 PMACs/model**, matching the earlier
approximately 303P value. Five-shot is exactly 5,000 / 1,280,000 = 1/256 of a
full training epoch for the same model.

## Per-tokenizer results

Accuracy values are percentages on the official ImageNet-1K validation set.
The selected LR is the base LR before global-batch scaling; the corresponding
effective LR is also shown. Rows follow the canonical 70-tokenizer manifest.

| # | Tokenizer | Model ID | Head | Top-1 (%) | Top-5 (%) | Best base LR | Effective LR | LR position |
|---:|---|---|---:|---:|---:|---:|---:|---|
| 1 | clip_openai__l14 | `clip_openai__l14` | Linear | 32.152 | 54.158 | 0.5 | 1.95312 | high |
| 2 | dino_vitb16 | `dinov1_vitb16` | Linear | 41.342 | 64.374 | 0.3 | 1.17188 | interior |
| 3 | dino_vitb8 | `dinov1_vitb8` | Linear | 47.988 | 71.566 | 0.3 | 1.17188 | interior |
| 4 | dino_vits16 | `dinov1_vits16` | Linear | 40.568 | 62.872 | 0.2 | 0.78125 | interior |
| 5 | dino_vits8 | `dinov1_vits8` | Linear | 40.894 | 61.078 | 0.2 | 0.78125 | interior |
| 6 | dinov2_base | `dinov2_base` | Linear | 57.194 | 82.290 | 0.5 | 1.95312 | high |
| 7 | dinov2_giant | `dinov2_giant` | Linear | 65.260 | 86.230 | 0.5 | 1.95312 | high |
| 8 | dinov2_large | `dinov2_large` | Linear | 63.518 | 85.388 | 0.5 | 1.95312 | high |
| 9 | dinov2_small | `dinov2_small` | Linear | 44.940 | 73.094 | 0.5 | 1.95312 | high |
| 10 | dinov3_vitl16 | `dinov3_vitl16_lvd1689m` | Linear | 48.294 | 70.182 | 0.5 | 1.95312 | high |
| 11 | toklip_l_384 | `toklip_l` | Linear | 3.686 | 7.258 | 0.5 | 1.95312 | high |
| 12 | toklip_s_256 | `toklip_s` | Linear | 4.038 | 8.306 | 0.5 | 1.95312 | high |
| 13 | uniar_bsq | `uniar_bsq` | Linear | 1.514 | 3.818 | 0.5 | 1.95312 | high |
| 14 | unitok_attn | `unitok` | Linear | 27.732 | 50.358 | 0.5 | 1.95312 | high |
| 15 | vilau_256 | `vilau` | Linear | 11.754 | 18.636 | 0.5 | 1.95312 | high |
| 16 | eupe_convnext_b | `eupe_convnext_b` | Linear | 28.726 | 48.464 | 0.5 | 1.95312 | high |
| 17 | eupe_vit_b | `eupe_vit_b` | Linear | 30.050 | 53.432 | 0.5 | 1.95312 | high |
| 18 | eupe_vit_s | `eupe_vit_s` | Linear | 20.704 | 44.494 | 0.5 | 1.95312 | high |
| 19 | eupe_vit_t | `eupe_vit_t` | Linear | 7.616 | 20.386 | 0.5 | 1.95312 | high |
| 20 | ijepa_vith14 | `ijepa` | Linear | 35.156 | 54.034 | 0.5 | 1.95312 | high |
| 21 | mc1_b16_224_2.5b | `mc1_b16_224_2.5b` | Linear | 21.768 | 40.890 | 0.5 | 1.95312 | high |
| 22 | mc1_b16_224_400m | `mc1_b16_224_400m` | Linear | 20.584 | 39.152 | 0.5 | 1.95312 | high |
| 23 | mc1_b32_224_2.5b | `mc1_b32_224_2.5b` | Linear | 16.006 | 32.712 | 0.5 | 1.95312 | high |
| 24 | mc1_b32_224_400m | `mc1_b32_224_400m` | Linear | 18.778 | 37.438 | 0.5 | 1.95312 | high |
| 25 | mc1_g14_224_2.5b | `mc1_g14_224_2.5b` | Linear | 37.012 | 55.780 | 0.5 | 1.95312 | high |
| 26 | mc1_h14_224_2.5b | `mc1_h14_224_2.5b` | Linear | 34.594 | 55.552 | 0.5 | 1.95312 | high |
| 27 | mc1_h14_224_v1.2 | `mc1_h14_224_v1.2` | Linear | 35.364 | 55.562 | 0.5 | 1.95312 | high |
| 28 | mc1_l14_224_2.5b | `mc1_l14_224_2.5b` | Linear | 31.334 | 52.806 | 0.5 | 1.95312 | high |
| 29 | mc1_l14_224_400m | `mc1_l14_224_400m` | Linear | 28.708 | 49.942 | 0.5 | 1.95312 | high |
| 30 | mc2_b16_224 | `mc2_b16_224` | Linear | 23.962 | 45.594 | 0.5 | 1.95312 | high |
| 31 | mc2_b16_384 | `mc2_b16_384` | Linear | 24.408 | 45.638 | 0.5 | 1.95312 | high |
| 32 | mc2_b32_224 | `mc2_b32_224` | Linear | 21.164 | 41.372 | 0.5 | 1.95312 | high |
| 33 | mc2_b32_224_mt5 | `mc2_b32_224_mt5` | Linear | 21.602 | 40.500 | 0.5 | 1.95312 | high |
| 34 | mc2_b32_384 | `mc2_b32_384` | Linear | 21.918 | 42.624 | 0.5 | 1.95312 | high |
| 35 | mc2_g14_224 | `mc2_g14_224` | Linear | 28.854 | 44.822 | 0.5 | 1.95312 | high |
| 36 | mc2_g14_378 | `mc2_g14_378` | Linear | 28.940 | 44.652 | 0.5 | 1.95312 | high |
| 37 | mc2_h14_378 | `mc2_h14_378` | Linear | 33.252 | 53.380 | 0.5 | 1.95312 | high |
| 38 | mc2_l14_224 | `mc2_l14_224` | Linear | 29.972 | 49.944 | 0.5 | 1.95312 | high |
| 39 | mc2_m16_224 | `mc2_m16_224` | Linear | 16.790 | 35.030 | 0.5 | 1.95312 | high |
| 40 | mc2_m16_224_mt5 | `mc2_m16_224_mt5` | Linear | 17.018 | 35.074 | 0.5 | 1.95312 | high |
| 41 | mc2_m16_384 | `mc2_m16_384` | Linear | 16.764 | 34.818 | 0.5 | 1.95312 | high |
| 42 | mc2_s16_224 | `mc2_s16_224` | Linear | 12.974 | 29.730 | 0.5 | 1.95312 | high |
| 43 | mc2_s16_224_mt5 | `mc2_s16_224_mt5` | Linear | 11.114 | 26.070 | 0.5 | 1.95312 | high |
| 44 | mc2_s16_384 | `mc2_s16_384` | Linear | 13.646 | 30.722 | 0.5 | 1.95312 | high |
| 45 | pe_core_b16_224 | `pe_core_b16_224` | Linear | 11.104 | 23.438 | 0.5 | 1.95312 | high |
| 46 | pe_core_g14_448 | `pe_core_g14_448` | Linear | 47.976 | 65.522 | 0.5 | 1.95312 | high |
| 47 | pe_lang_l14_448 | `pe_lang_l14_448` | Linear | 0.878 | 2.186 | 0.5 | 1.95312 | high |
| 48 | pixio_vitb16 | `pixio_vitb16` | BN→Linear | 22.444 | 45.164 | 0.5 | 1.95312 | high |
| 49 | pixio_vith16 | `pixio_vith16` | BN→Linear | 20.724 | 41.746 | 0.5 | 1.95312 | high |
| 50 | pixio_vitl16 | `pixio_vitl16` | BN→Linear | 24.888 | 46.530 | 0.5 | 1.95312 | high |
| 51 | raev2_dinov3l_k7 | `raev2_dinov3l_k7` | Linear | 36.674 | 62.894 | 0.5 | 1.95312 | high |
| 52 | siglip2_b16_224 | `siglip2_b16_224` | Linear | 8.504 | 15.456 | 0.5 | 1.95312 | high |
| 53 | siglip2_b16_256 | `siglip2_b16_256` | Linear | 7.896 | 15.736 | 0.5 | 1.95312 | high |
| 54 | siglip2_b16_384 | `siglip2_b16_384` | Linear | 7.854 | 14.562 | 0.5 | 1.95312 | high |
| 55 | siglip2_b16_512 | `siglip2_b16_512` | Linear | 6.876 | 14.092 | 0.5 | 1.95312 | high |
| 56 | siglip2_b32_256 | `siglip2_b32_256` | Linear | 6.792 | 13.540 | 0.5 | 1.95312 | high |
| 57 | siglip2_g16_256 | `siglip2_g16_256` | Linear | 24.970 | 40.408 | 0.5 | 1.95312 | high |
| 58 | siglip2_g16_384 | `siglip2_g16_384` | Linear | 22.624 | 37.474 | 0.5 | 1.95312 | high |
| 59 | siglip2_l16_256 | `siglip2_l16_256` | Linear | 16.018 | 28.450 | 0.5 | 1.95312 | high |
| 60 | siglip2_l16_384 | `siglip2_l16_384` | Linear | 15.240 | 27.124 | 0.5 | 1.95312 | high |
| 61 | siglip2_l16_512 | `siglip2_l16_512` | Linear | 15.170 | 27.334 | 0.5 | 1.95312 | high |
| 62 | siglip2_sm14_224 | `siglip2_sm14_224` | Linear | 17.208 | 30.196 | 0.5 | 1.95312 | high |
| 63 | siglip2_sm14_384 | `siglip2_sm14_384` | Linear | 15.970 | 28.488 | 0.5 | 1.95312 | high |
| 64 | siglip2_sm16_256 | `siglip2_sm16_256` | Linear | 16.646 | 29.300 | 0.5 | 1.95312 | high |
| 65 | siglip2_sm16_384 | `siglip2_sm16_384` | Linear | 16.308 | 28.426 | 0.5 | 1.95312 | high |
| 66 | siglip2_sm16_512 | `siglip2_sm16_512` | Linear | 16.796 | 29.444 | 0.5 | 1.95312 | high |
| 67 | webssl_dino1b_full2b_224 | `webssl_dino1b_full2b_224` | Linear | 35.976 | 57.944 | 0.2 | 0.78125 | interior |
| 68 | webssl_mae1b_full2b_224 | `webssl_mae1b_full2b_224` | BN→Linear | 2.052 | 5.102 | 0.5 | 1.95312 | high |
| 69 | webssl_mae300m_full2b_224 | `webssl_mae300m_full2b_224` | BN→Linear | 0.350 | 1.188 | 0.5 | 1.95312 | high |
| 70 | webssl_mae3b_full2b_224 | `webssl_mae3b_full2b_224` | BN→Linear | 0.704 | 2.764 | 0.5 | 1.95312 | high |

## Provenance

- Manifest: `scripts/linear_probe_match_budget_5shot/tokenizers.tsv`.
- Per-model metrics: each model's `results_eval_linear.json` under this output
  directory.
- FLOPs profiler: `scripts/linear_probe_match_budget_5shot/profile_flops.py`.
