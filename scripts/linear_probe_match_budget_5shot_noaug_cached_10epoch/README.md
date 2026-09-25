# ImageNet-1K deterministic cached 5-shot probing

This directory runs the canonical 70-tokenizer panel with a fixed deterministic
five-shot support set and persistent feature caches.

- Support set: exactly 5 distinct images per class (5,000 total), support seed 0.
- Augmentation: none. Train and validation both use each encoder's deterministic
  evaluation transform.
- Feature extraction: the 5,000 support features and 50,000 validation features
  are each extracted once and stored as FP32 NumPy arrays.
- Head training: batch size 1,000, 10 epochs, 5 updates per epoch, 50 updates
  total, with one 10-epoch cosine schedule.
- Heads: all 70 tokenizers use the same non-affine `BatchNorm1d -> Linear`
  protocol.
- Validation: cached full ImageNet-1K validation features are evaluated after
  every epoch; the encoder is not run again.
- Outputs and caches:
  `outputs/vae_linear_probing_match_budget_5shot_noaug_cached_10epoch`.

Run all 70 models on one GPU:

```bash
bash scripts/linear_probe_match_budget_5shot_noaug_cached_10epoch/run_all.sh
```

Run selected models by rank, requested alias, or internal model ID:

```bash
bash scripts/linear_probe_match_budget_5shot_noaug_cached_10epoch/run_all.sh \
  1 dinov3_vitl16 webssl_mae300m_full2b_224
```

Use multiple GPUs with static round-robin sharding:

```bash
CACHED_10E_GPUS=0,1,2,3 \
  bash scripts/linear_probe_match_budget_5shot_noaug_cached_10epoch/run_all.sh
```

Completed 50-update result files are skipped automatically. An interrupted run
reuses a complete feature cache and resumes its head checkpoint.
