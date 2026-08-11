# SUN397 fixed-surface tokenizer linear probing

This experiment reuses the feature extractors and optimization choices from
`scripts/linear_probe_tokenizers` while replacing ImageNet-1K with SUN397.

- train / validation / test: 76,127 / 10,875 / 21,750 images;
- 397 classes and the model-native train/evaluation transforms;
- one frozen, preselected feature surface per tokenizer;
- global batch 1024 and 13 parallel learning-rate heads;
- SGD, momentum 0.9, no weight decay, and cosine decay to zero;
- 75 updates per approximate epoch (76,800 sampled images, 0.88% over one pass);
- validation every epoch; final validation selects one head, then test is
  evaluated exactly once with that head.
- micro top-1/top-5 match the ImageNet metric and micro top-1 selects the LR;
  macro top-1/top-5 are also saved because the local split is imbalanced.

The local data are the `dpdl-benchmark/sun397` all-image 70/10/20-style
split. This is **not** the canonical SUN397 protocol averaging ten partitions
with 50 train and 50 test images per class, so do not compare its absolute
accuracy directly with canonical SUN397 papers.

The default is 10 epochs and seed 0. Results are isolated by model and seed
under `outputs/sun397_linear_probing_dinov2_single_surface/`.

At 10 epochs, one run processes 768,000 training samples, 108,750 validation
samples across the ten epoch boundaries, and 21,750 final test samples. The
898,500 frozen-backbone image forwards are about 6.76% of the current
ImageNet-1K 10-epoch protocol. Based on the existing ImageNet run logs,
MetaCLIP should take roughly 10 minutes, the five legacy anchors roughly 6.7
GPU-hours in total, and the 37-model completed cohort roughly 42 GPU-hours.
These are projections; parquet I/O and the available GPU can shift them.

Run one model:

```bash
conda activate dino
CUDA_VISIBLE_DEVICES=0 \
  bash scripts/linear_probe_tokenizers_sun397/run_model.sh metaclip
```

Run a second seed or change the epoch count:

```bash
CUDA_VISIBLE_DEVICES=0 \
  bash scripts/linear_probe_tokenizers_sun397/run_model.sh metaclip \
  --seed 1 --epochs 10
```

For an initial stability audit, run seed 0 across the desired model cohort and
use `metrics_history.jsonl` to compare every epoch with epoch 10. Only after
the seed-0 ranking looks useful should seeds 1 and 2 be run on representative
cross-family anchors.

The launcher defaults frozen-feature microbatches to 256 for TokLIP-S,
TokLIP-L, and UniTok, and 16 for VQGAN; other models use 1024. Override with
`--feature-microbatch-size` when the visible GPU requires it. This changes
only feature-forward chunking, not the optimization batch.

The final validation table is `results_eval_linear.json`; the held-out result
is `results_test_linear.json`. Intermediate checkpoints reuse one filename,
while validation metrics are appended once per epoch.
