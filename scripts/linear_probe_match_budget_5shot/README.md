# ImageNet-1K 5-shot match-budget probing

This directory runs the 64 unique tokenizers from the original 42-model panel
plus the additional 23-model request (one model occurs in both lists).

- Train support: exactly 5 distinct images per ImageNet-1K class (5,000 total).
- Training: a fresh Linear/BN-Linear head, batch size 1,000, one epoch, exactly
  5 optimizer updates. No 4-shot head checkpoint is reused.
- Validation: the complete official 50,000-image validation set with batch size
  1,024 and `eval_num_workers=0` to avoid host-memory prefetch OOM.
- TokLIP-L uses a validation-only frozen-encoder microbatch of 128 instead of
  256 to fit an 80 GiB GPU. The outer validation batch and metrics are unchanged.
- Heads: Pixio and Web-SSL MAE use non-affine `BatchNorm1d -> Linear`; every
  other tokenizer uses plain `Linear`.
- Output: `outputs/vae_linear_probing_match_budget_5shot`.

Run all 64 models on one GPU:

```bash
bash scripts/linear_probe_match_budget_5shot/run_all.sh
```

Run on several GPUs using static round-robin sharding:

```bash
MATCH_BUDGET_5SHOT_GPUS=0,1,2,3 \
  bash scripts/linear_probe_match_budget_5shot/run_all.sh
```

Run selected models by combined rank, requested alias, or internal model ID:

```bash
bash scripts/linear_probe_match_budget_5shot/run_all.sh 1 unitok_attn mc2_b16_384
```

Dry-run the complete panel:

```bash
MATCH_BUDGET_5SHOT_DRY_RUN=1 \
  bash scripts/linear_probe_match_budget_5shot/run_all.sh
```
