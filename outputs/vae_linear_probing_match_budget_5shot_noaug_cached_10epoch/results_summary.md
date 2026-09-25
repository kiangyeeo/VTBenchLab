# ImageNet-1K 5-shot no-augmentation cached 10-epoch linear probing

## Completion and protocol

- Status: **70/70 complete**.
- Result validation: every tokenizer reached **iteration 50**, contains all
  **13 learning-rate heads**, and reports protocol version
  `tokenizer_linear_probe_imagenet_5shot_noaug_cached_10epoch_allbn_v1`.
- Training set: fixed class-balanced ImageNet-1K 5-shot support set, **5 images
  per class / 5,000 distinct images total**.
- Probe: `BatchNorm1d(affine=False) -> Linear` for all 70 tokenizers.
- Augmentation: **none**. Train and validation both use the deterministic eval
  transform belonging to each encoder.
- Cached execution: frozen train features and frozen validation features are
  each extracted once. The 5,000 cached train features are reused for **10
  epochs**, with batch size 1,000: 5 updates/epoch and 50 updates total.
- Evaluation: official 50,000-image ImageNet-1K validation split. Validation
  features are cached once and reused for the epoch-wise head evaluations.

## Accuracy summary

- Mean Top-1: **61.083%**; median Top-1: **62.493%**.
- Mean Top-5: **84.115%**; median Top-5: **87.346%**.
- Best Top-1: **79.198%**, `pe_core_g14_448`.
- Lowest Top-1: **12.212%**, `webssl_mae1b_full2b_224`.
- All **70/70** selected learning rates are interior points of the tested grid;
  none selected its lower or upper boundary.

## Epoch-5 intermediate results

All **70/70** histories contain the epoch-5 evaluation at `iteration=25`, with
results for all 13 learning-rate heads. The table reports the best head selected
independently at epoch 5. `Delta 5→10` is the final epoch-10 result minus the
epoch-5 result in percentage points.

- Epoch-5 mean Top-1: **60.358%**.
- Epoch-5 mean Top-5: **83.579%**.
- Mean improvement from epoch 5 to epoch 10: **+0.725 Top-1 points** and
  **+0.536 Top-5 points**.
- Top-1 improved for **64/70** tokenizers from epoch 5 to epoch 10.
- These are intermediate points from the fixed 10-epoch cosine schedule. They
  are not equivalent to a separate 5-epoch experiment whose cosine schedule
  reaches zero at epoch 5.

| # | Tokenizer | Epoch-5 Top-1 (%) | Epoch-5 Top-5 (%) | Epoch-5 best base LR | Effective LR | Top-1 delta 5→10 | Top-5 delta 5→10 |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | `clip_openai__l14` | 65.192 | 89.256 | 0.1 | 0.390625 | +0.428 | +0.204 |
| 2 | `dino_vitb16` | 54.818 | 80.652 | 0.1 | 0.390625 | -0.326 | -0.630 |
| 3 | `dino_vitb8` | 59.596 | 84.574 | 0.1 | 0.390625 | -0.302 | -0.572 |
| 4 | `dino_vits16` | 50.382 | 75.196 | 0.1 | 0.390625 | -0.004 | +0.140 |
| 5 | `dino_vits8` | 59.192 | 83.190 | 0.1 | 0.390625 | -0.060 | +0.058 |
| 6 | `dinov2_base` | 68.876 | 91.136 | 0.05 | 0.1953125 | +0.376 | +0.026 |
| 7 | `dinov2_giant` | 72.984 | 93.028 | 0.05 | 0.1953125 | +0.172 | -0.324 |
| 8 | `dinov2_large` | 73.252 | 93.058 | 0.05 | 0.1953125 | -0.044 | -0.078 |
| 9 | `dinov2_small` | 59.496 | 85.532 | 0.1 | 0.390625 | +0.334 | +0.174 |
| 10 | `dinov3_vitl16` | 75.428 | 93.922 | 0.05 | 0.1953125 | -0.670 | -0.312 |
| 11 | `toklip_l_384` | 59.016 | 83.558 | 0.1 | 0.390625 | +1.208 | +0.524 |
| 12 | `toklip_s_256` | 55.392 | 80.028 | 0.05 | 0.1953125 | +1.334 | +1.032 |
| 13 | `uniar_bsq` | 42.454 | 64.928 | 0.02 | 0.078125 | +2.666 | +2.890 |
| 14 | `unitok_attn` | 63.064 | 88.080 | 0.1 | 0.390625 | +0.712 | +0.220 |
| 15 | `vilau_256` | 62.874 | 85.860 | 0.1 | 0.390625 | +0.858 | +0.558 |
| 16 | `eupe_convnext_b` | 58.914 | 85.342 | 0.1 | 0.390625 | +0.910 | +0.418 |
| 17 | `eupe_vit_b` | 73.114 | 93.596 | 0.05 | 0.1953125 | +0.250 | +0.028 |
| 18 | `eupe_vit_s` | 61.198 | 87.848 | 0.1 | 0.390625 | +0.730 | +0.128 |
| 19 | `eupe_vit_t` | 37.538 | 68.294 | 0.2 | 0.78125 | +0.912 | +0.692 |
| 20 | `ijepa_vith14` | 58.710 | 78.662 | 0.1 | 0.390625 | +0.290 | -0.082 |
| 21 | `mc1_b16_224_2.5b` | 57.870 | 84.564 | 0.1 | 0.390625 | +0.512 | +0.500 |
| 22 | `mc1_b16_224_400m` | 56.554 | 83.406 | 0.1 | 0.390625 | +0.678 | +0.532 |
| 23 | `mc1_b32_224_2.5b` | 53.136 | 80.836 | 0.1 | 0.390625 | +0.556 | +0.572 |
| 24 | `mc1_b32_224_400m` | 51.216 | 79.190 | 0.1 | 0.390625 | +0.492 | +0.406 |
| 25 | `mc1_g14_224_2.5b` | 70.792 | 91.534 | 0.05 | 0.1953125 | +0.436 | +0.268 |
| 26 | `mc1_h14_224_2.5b` | 68.988 | 90.706 | 0.05 | 0.1953125 | +0.696 | +0.434 |
| 27 | `mc1_h14_224_v1.2` | 69.618 | 90.902 | 0.05 | 0.1953125 | +0.826 | +0.448 |
| 28 | `mc1_l14_224_2.5b` | 67.320 | 90.374 | 0.1 | 0.390625 | +0.598 | +0.324 |
| 29 | `mc1_l14_224_400m` | 64.238 | 88.094 | 0.1 | 0.390625 | +0.472 | +0.312 |
| 30 | `mc2_b16_224` | 59.512 | 85.594 | 0.1 | 0.390625 | +0.790 | +0.498 |
| 31 | `mc2_b16_384` | 61.060 | 86.662 | 0.1 | 0.390625 | +0.764 | +0.592 |
| 32 | `mc2_b32_224` | 52.916 | 80.474 | 0.1 | 0.390625 | +0.524 | +0.524 |
| 33 | `mc2_b32_224_mt5` | 54.412 | 81.594 | 0.1 | 0.390625 | +0.476 | +0.520 |
| 34 | `mc2_b32_384` | 56.224 | 83.164 | 0.1 | 0.390625 | +0.710 | +0.616 |
| 35 | `mc2_g14_224` | 70.108 | 90.664 | 0.05 | 0.1953125 | +0.768 | +0.486 |
| 36 | `mc2_g14_378` | 71.556 | 91.528 | 0.05 | 0.1953125 | +0.814 | +0.438 |
| 37 | `mc2_h14_378` | 70.000 | 91.336 | 0.1 | 0.390625 | +0.652 | +0.286 |
| 38 | `mc2_l14_224` | 68.486 | 90.414 | 0.1 | 0.390625 | +0.652 | +0.352 |
| 39 | `mc2_m16_224` | 53.554 | 81.556 | 0.2 | 0.78125 | +0.722 | +0.390 |
| 40 | `mc2_m16_224_mt5` | 53.142 | 81.348 | 0.2 | 0.78125 | +0.750 | +0.258 |
| 41 | `mc2_m16_384` | 55.558 | 83.264 | 0.2 | 0.78125 | +0.924 | +0.344 |
| 42 | `mc2_s16_224` | 49.314 | 77.602 | 0.2 | 0.78125 | +0.404 | +0.500 |
| 43 | `mc2_s16_224_mt5` | 47.544 | 76.720 | 0.2 | 0.78125 | +0.390 | +0.362 |
| 44 | `mc2_s16_384` | 52.128 | 80.250 | 0.2 | 0.78125 | +0.402 | +0.330 |
| 45 | `pe_core_b16_224` | 64.590 | 89.332 | 0.1 | 0.390625 | +0.740 | +0.354 |
| 46 | `pe_core_g14_448` | 78.972 | 95.528 | 0.05 | 0.1953125 | +0.226 | +0.106 |
| 47 | `pe_lang_l14_448` | 55.664 | 82.760 | 0.1 | 0.390625 | +1.864 | +1.262 |
| 48 | `pixio_vitb16` | 42.242 | 70.410 | 0.1 | 0.390625 | +1.254 | +1.114 |
| 49 | `pixio_vith16` | 40.104 | 67.592 | 0.1 | 0.390625 | +1.514 | +1.122 |
| 50 | `pixio_vitl16` | 46.384 | 73.176 | 0.1 | 0.390625 | +1.806 | +1.334 |
| 51 | `raev2_dinov3l_k7` | 62.512 | 85.600 | 0.1 | 0.390625 | +0.006 | +0.026 |
| 52 | `siglip2_b16_224` | 68.342 | 91.268 | 0.1 | 0.390625 | +0.562 | +0.206 |
| 53 | `siglip2_b16_256` | 68.916 | 91.630 | 0.1 | 0.390625 | +0.770 | +0.236 |
| 54 | `siglip2_b16_384` | 70.428 | 92.248 | 0.1 | 0.390625 | +0.618 | +0.236 |
| 55 | `siglip2_b16_512` | 70.794 | 92.430 | 0.1 | 0.390625 | +0.754 | +0.212 |
| 56 | `siglip2_b32_256` | 62.596 | 87.274 | 0.1 | 0.390625 | +0.622 | +0.164 |
| 57 | `siglip2_g16_256` | 77.474 | 94.760 | 0.05 | 0.1953125 | +0.350 | +0.084 |
| 58 | `siglip2_g16_384` | 77.664 | 94.772 | 0.05 | 0.1953125 | +0.452 | +0.062 |
| 59 | `siglip2_l16_256` | 74.090 | 93.636 | 0.1 | 0.390625 | +0.496 | +0.146 |
| 60 | `siglip2_l16_384` | 74.300 | 93.710 | 0.05 | 0.1953125 | +0.648 | +0.330 |
| 61 | `siglip2_l16_512` | 74.950 | 94.158 | 0.1 | 0.390625 | +0.586 | +0.074 |
| 62 | `siglip2_sm14_224` | 74.940 | 94.164 | 0.1 | 0.390625 | +0.734 | +0.018 |
| 63 | `siglip2_sm14_384` | 75.658 | 94.208 | 0.05 | 0.1953125 | +0.594 | +0.272 |
| 64 | `siglip2_sm16_256` | 74.974 | 93.964 | 0.05 | 0.1953125 | +0.592 | +0.254 |
| 65 | `siglip2_sm16_384` | 75.570 | 94.192 | 0.05 | 0.1953125 | +0.630 | +0.324 |
| 66 | `siglip2_sm16_512` | 75.728 | 94.370 | 0.05 | 0.1953125 | +0.674 | +0.292 |
| 67 | `webssl_dino1b_full2b_224` | 60.404 | 86.960 | 0.05 | 0.1953125 | +2.064 | +1.038 |
| 68 | `webssl_mae1b_full2b_224` | 10.140 | 24.200 | 0.05 | 0.1953125 | +2.072 | +3.752 |
| 69 | `webssl_mae300m_full2b_224` | 10.992 | 26.034 | 0.1 | 0.390625 | +2.018 | +3.462 |
| 70 | `webssl_mae3b_full2b_224` | 9.926 | 24.646 | 0.05 | 0.1953125 | +3.288 | +5.624 |

## Correlation with the three MLLMs in the supplied table

The MLLM values in this section are transcribed exclusively from the newly
supplied 70-encoder table image; no local MLLM result file is used. The image
rows were mapped one-to-one to the canonical manifest: **70 rows, 70 unique
tokenizer IDs, no missing or extra IDs**. Every correlation therefore uses the
same exact **n=70** cohort.

For each MLLM, Spearman's rho compares ranks (ties use average ranks), while
Pearson's r compares the raw MLLM aggregate scores with ImageNet linear-probe
Top-1. `Top-1` follows the supplied table's definition: the ground-truth MLLM
score of the encoder ranked first by linear probing, rather than the
linear-probe accuracy itself.

| Linear-probing checkpoint | Qwen3-1.7B rho | Qwen3-1.7B r | Qwen3-1.7B Top-1 | Qwen2.5-1.5B rho | Qwen2.5-1.5B r | Qwen2.5-1.5B Top-1 | SmolLM2-1.7B rho | SmolLM2-1.7B r | SmolLM2-1.7B Top-1 | n |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Epoch 5 (`iteration=25`) | 0.684 | 0.681 | 48.58 | 0.755 | 0.685 | 51.88 | 0.730 | 0.732 | 42.41 | 70 |
| **Epoch 10 (`iteration=50`)** | **0.693** | **0.693** | **48.58** | **0.770** | **0.702** | **51.88** | **0.744** | **0.745** | **42.41** | **70** |

Epoch 10 is the stronger checkpoint for the complete 70-tokenizer table: all
three Spearman coefficients and all three Pearson coefficients improve over
epoch 5. The Top-1 encoder at both checkpoints is `pe_core_g14_448`
(linear-probe Top-1 78.972% at epoch 5 and 79.198% at epoch 10), so the three
MLLM Top-1 values are unchanged.

## Average FLOPs

These numbers use the same measured frozen-encoder surfaces and actual input
resolutions as the preceding 70-tokenizer 5-shot run. Removing augmentation
does not change the input tensor shapes. Ten cached head-training epochs do
**not** repeat encoder computation: each train and validation image passes
through its frozen encoder once.

The main budget convention below counts one multiply-accumulate as one
operation (MAC), matching the earlier approximately 303P estimate. The strict
PyTorch operator-FLOPs convention counts multiply and add separately and is
therefore exactly 2x. BatchNorm and linear-head computation are excluded, as in
the earlier 303P budget; they are feature-space operations and are small
relative to the encoders.

| Scope | Average per evaluated encoder (MAC convention) | Average per evaluated encoder (strict operator FLOPs) | Total across 70 encoders (MAC convention) | Total across 70 encoders (strict operator FLOPs) |
|---|---:|---:|---:|---:|
| One image | 183.668922 GMACs | 367.337844 GFLOPs | — | — |
| Train cache: 5,000 images | **0.918345 PMACs** | **1.836689 PFLOPs** | 64.284123 PMACs | 128.568245 PFLOPs |
| Validation cache: 50,000 images | 9.183446 PMACs | 18.366892 PFLOPs | 642.841227 PMACs | 1,285.682454 PFLOPs |
| Train + validation caches: 55,000 images | **10.101791 PMACs** | **20.203581 PFLOPs** | **707.125350 PMACs** | **1,414.250699 PFLOPs** |

Thus, the average **training-side frozen-encoder cost is about 0.92P per
evaluated encoder**, still approximately the intended 1P budget. If the full
50,000-image validation encoder pass is included, the average one-time total is
about **10.10P per evaluated encoder**. The 10 epochs affect only the cached
BN/linear heads, not these encoder-FLOPs figures.

## Per-tokenizer results

Accuracy values are percentages on the official ImageNet-1K validation set.
Rows follow the canonical 70-tokenizer manifest. `Best base LR` is the configured
learning rate before global-batch scaling; `Effective LR = base LR × 1000 / 256`.

| # | Tokenizer | Model ID | Top-1 (%) | Top-5 (%) | Best base LR | Effective LR | LR position |
|---:|---|---|---:|---:|---:|---:|---|
| 1 | `clip_openai__l14` | `clip_openai__l14` | 65.620 | 89.460 | 0.1 | 0.390625 | interior |
| 2 | `dino_vitb16` | `dinov1_vitb16` | 54.492 | 80.022 | 0.05 | 0.1953125 | interior |
| 3 | `dino_vitb8` | `dinov1_vitb8` | 59.294 | 84.002 | 0.05 | 0.1953125 | interior |
| 4 | `dino_vits16` | `dinov1_vits16` | 50.378 | 75.336 | 0.1 | 0.390625 | interior |
| 5 | `dino_vits8` | `dinov1_vits8` | 59.132 | 83.248 | 0.1 | 0.390625 | interior |
| 6 | `dinov2_base` | `dinov2_base` | 69.252 | 91.162 | 0.05 | 0.1953125 | interior |
| 7 | `dinov2_giant` | `dinov2_giant` | 73.156 | 92.704 | 0.02 | 0.078125 | interior |
| 8 | `dinov2_large` | `dinov2_large` | 73.208 | 92.980 | 0.05 | 0.1953125 | interior |
| 9 | `dinov2_small` | `dinov2_small` | 59.830 | 85.706 | 0.1 | 0.390625 | interior |
| 10 | `dinov3_vitl16` | `dinov3_vitl16_lvd1689m` | 74.758 | 93.610 | 0.05 | 0.1953125 | interior |
| 11 | `toklip_l_384` | `toklip_l` | 60.224 | 84.082 | 0.05 | 0.1953125 | interior |
| 12 | `toklip_s_256` | `toklip_s` | 56.726 | 81.060 | 0.05 | 0.1953125 | interior |
| 13 | `uniar_bsq` | `uniar_bsq` | 45.120 | 67.818 | 0.02 | 0.078125 | interior |
| 14 | `unitok_attn` | `unitok` | 63.776 | 88.300 | 0.05 | 0.1953125 | interior |
| 15 | `vilau_256` | `vilau` | 63.732 | 86.418 | 0.1 | 0.390625 | interior |
| 16 | `eupe_convnext_b` | `eupe_convnext_b` | 59.824 | 85.760 | 0.1 | 0.390625 | interior |
| 17 | `eupe_vit_b` | `eupe_vit_b` | 73.364 | 93.624 | 0.05 | 0.1953125 | interior |
| 18 | `eupe_vit_s` | `eupe_vit_s` | 61.928 | 87.976 | 0.1 | 0.390625 | interior |
| 19 | `eupe_vit_t` | `eupe_vit_t` | 38.450 | 68.986 | 0.2 | 0.78125 | interior |
| 20 | `ijepa_vith14` | `ijepa` | 59.000 | 78.580 | 0.05 | 0.1953125 | interior |
| 21 | `mc1_b16_224_2.5b` | `mc1_b16_224_2.5b` | 58.382 | 85.064 | 0.1 | 0.390625 | interior |
| 22 | `mc1_b16_224_400m` | `mc1_b16_224_400m` | 57.232 | 83.938 | 0.1 | 0.390625 | interior |
| 23 | `mc1_b32_224_2.5b` | `mc1_b32_224_2.5b` | 53.692 | 81.408 | 0.1 | 0.390625 | interior |
| 24 | `mc1_b32_224_400m` | `mc1_b32_224_400m` | 51.708 | 79.596 | 0.1 | 0.390625 | interior |
| 25 | `mc1_g14_224_2.5b` | `mc1_g14_224_2.5b` | 71.228 | 91.802 | 0.05 | 0.1953125 | interior |
| 26 | `mc1_h14_224_2.5b` | `mc1_h14_224_2.5b` | 69.684 | 91.140 | 0.05 | 0.1953125 | interior |
| 27 | `mc1_h14_224_v1.2` | `mc1_h14_224_v1.2` | 70.444 | 91.350 | 0.05 | 0.1953125 | interior |
| 28 | `mc1_l14_224_2.5b` | `mc1_l14_224_2.5b` | 67.918 | 90.698 | 0.1 | 0.390625 | interior |
| 29 | `mc1_l14_224_400m` | `mc1_l14_224_400m` | 64.710 | 88.406 | 0.1 | 0.390625 | interior |
| 30 | `mc2_b16_224` | `mc2_b16_224` | 60.302 | 86.092 | 0.1 | 0.390625 | interior |
| 31 | `mc2_b16_384` | `mc2_b16_384` | 61.824 | 87.254 | 0.1 | 0.390625 | interior |
| 32 | `mc2_b32_224` | `mc2_b32_224` | 53.440 | 80.998 | 0.1 | 0.390625 | interior |
| 33 | `mc2_b32_224_mt5` | `mc2_b32_224_mt5` | 54.888 | 82.114 | 0.1 | 0.390625 | interior |
| 34 | `mc2_b32_384` | `mc2_b32_384` | 56.934 | 83.780 | 0.1 | 0.390625 | interior |
| 35 | `mc2_g14_224` | `mc2_g14_224` | 70.876 | 91.150 | 0.05 | 0.1953125 | interior |
| 36 | `mc2_g14_378` | `mc2_g14_378` | 72.370 | 91.966 | 0.05 | 0.1953125 | interior |
| 37 | `mc2_h14_378` | `mc2_h14_378` | 70.652 | 91.622 | 0.05 | 0.1953125 | interior |
| 38 | `mc2_l14_224` | `mc2_l14_224` | 69.138 | 90.766 | 0.1 | 0.390625 | interior |
| 39 | `mc2_m16_224` | `mc2_m16_224` | 54.276 | 81.946 | 0.1 | 0.390625 | interior |
| 40 | `mc2_m16_224_mt5` | `mc2_m16_224_mt5` | 53.892 | 81.606 | 0.1 | 0.390625 | interior |
| 41 | `mc2_m16_384` | `mc2_m16_384` | 56.482 | 83.608 | 0.1 | 0.390625 | interior |
| 42 | `mc2_s16_224` | `mc2_s16_224` | 49.718 | 78.102 | 0.2 | 0.78125 | interior |
| 43 | `mc2_s16_224_mt5` | `mc2_s16_224_mt5` | 47.934 | 77.082 | 0.2 | 0.78125 | interior |
| 44 | `mc2_s16_384` | `mc2_s16_384` | 52.530 | 80.580 | 0.2 | 0.78125 | interior |
| 45 | `pe_core_b16_224` | `pe_core_b16_224` | 65.330 | 89.686 | 0.1 | 0.390625 | interior |
| 46 | `pe_core_g14_448` | `pe_core_g14_448` | 79.198 | 95.634 | 0.05 | 0.1953125 | interior |
| 47 | `pe_lang_l14_448` | `pe_lang_l14_448` | 57.528 | 84.022 | 0.1 | 0.390625 | interior |
| 48 | `pixio_vitb16` | `pixio_vitb16` | 43.496 | 71.524 | 0.1 | 0.390625 | interior |
| 49 | `pixio_vith16` | `pixio_vith16` | 41.618 | 68.714 | 0.1 | 0.390625 | interior |
| 50 | `pixio_vitl16` | `pixio_vitl16` | 48.190 | 74.510 | 0.1 | 0.390625 | interior |
| 51 | `raev2_dinov3l_k7` | `raev2_dinov3l_k7` | 62.518 | 85.626 | 0.1 | 0.390625 | interior |
| 52 | `siglip2_b16_224` | `siglip2_b16_224` | 68.904 | 91.474 | 0.1 | 0.390625 | interior |
| 53 | `siglip2_b16_256` | `siglip2_b16_256` | 69.686 | 91.866 | 0.1 | 0.390625 | interior |
| 54 | `siglip2_b16_384` | `siglip2_b16_384` | 71.046 | 92.484 | 0.1 | 0.390625 | interior |
| 55 | `siglip2_b16_512` | `siglip2_b16_512` | 71.548 | 92.642 | 0.1 | 0.390625 | interior |
| 56 | `siglip2_b32_256` | `siglip2_b32_256` | 63.218 | 87.438 | 0.1 | 0.390625 | interior |
| 57 | `siglip2_g16_256` | `siglip2_g16_256` | 77.824 | 94.844 | 0.05 | 0.1953125 | interior |
| 58 | `siglip2_g16_384` | `siglip2_g16_384` | 78.116 | 94.834 | 0.05 | 0.1953125 | interior |
| 59 | `siglip2_l16_256` | `siglip2_l16_256` | 74.586 | 93.782 | 0.05 | 0.1953125 | interior |
| 60 | `siglip2_l16_384` | `siglip2_l16_384` | 74.948 | 94.040 | 0.05 | 0.1953125 | interior |
| 61 | `siglip2_l16_512` | `siglip2_l16_512` | 75.536 | 94.232 | 0.05 | 0.1953125 | interior |
| 62 | `siglip2_sm14_224` | `siglip2_sm14_224` | 75.674 | 94.182 | 0.05 | 0.1953125 | interior |
| 63 | `siglip2_sm14_384` | `siglip2_sm14_384` | 76.252 | 94.480 | 0.05 | 0.1953125 | interior |
| 64 | `siglip2_sm16_256` | `siglip2_sm16_256` | 75.566 | 94.218 | 0.05 | 0.1953125 | interior |
| 65 | `siglip2_sm16_384` | `siglip2_sm16_384` | 76.200 | 94.516 | 0.05 | 0.1953125 | interior |
| 66 | `siglip2_sm16_512` | `siglip2_sm16_512` | 76.402 | 94.662 | 0.05 | 0.1953125 | interior |
| 67 | `webssl_dino1b_full2b_224` | `webssl_dino1b_full2b_224` | 62.468 | 87.998 | 0.1 | 0.390625 | interior |
| 68 | `webssl_mae1b_full2b_224` | `webssl_mae1b_full2b_224` | 12.212 | 27.952 | 0.05 | 0.1953125 | interior |
| 69 | `webssl_mae300m_full2b_224` | `webssl_mae300m_full2b_224` | 13.010 | 29.496 | 0.1 | 0.390625 | interior |
| 70 | `webssl_mae3b_full2b_224` | `webssl_mae3b_full2b_224` | 13.214 | 30.270 | 0.05 | 0.1953125 | interior |

## Provenance

- Canonical manifest:
  `scripts/linear_probe_match_budget_5shot_noaug_cached_10epoch/tokenizers.tsv`
- Run log:
  `outputs/vae_linear_probing_match_budget_5shot_noaug_cached_10epoch/run_all.log`
- Per-tokenizer source files: each canonical output directory's
  `results_eval_linear.json`, `metrics_history.jsonl`, and `protocol.json`.
- FLOPs profiler:
  `scripts/linear_probe_match_budget_5shot/profile_flops.py`.
