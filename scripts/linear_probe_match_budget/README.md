# ImageNet-1K 4-shot match-budget probing

This directory runs exactly the 42 vision encoders in `tokenizers.tsv`.

- Train support: exactly 4 distinct images per ImageNet-1K class (4,000 total).
- Support selection: deterministic and shared across models (`support_seed=0`).
- Training: one pass, batch size 1,000, exactly 4 optimizer updates.
- Validation: the official 50,000-image ImageNet validation set, batch size 1,024.
- Validation loading uses the main process by default (`eval_num_workers=0`) to
  prevent multi-worker prefetch from exhausting host memory at 512px. This does
  not change validation membership, order, batches, model forwards, or metrics.
- Heads: Pixio and Web-SSL MAE use the non-affine `BatchNorm1d -> Linear`
  implementation from `scripts/linear_probe_tokenizers_bn`; all others use the
  plain `Linear` implementation from `scripts/linear_probe_tokenizers`.
- LR grid, frozen feature surfaces, transforms, and model loaders are inherited
  from the existing tokenizer probing implementation.

Run all models on one GPU:

```bash
bash scripts/linear_probe_match_budget/run_all.sh
```

Run the additional requested 23-model panel on one GPU:

```bash
bash scripts/linear_probe_match_budget/run_extra_23.sh
```

The extra panel accepts its 1-based order, requested alias, or internal model
ID when selecting individual models. For example:

```bash
bash scripts/linear_probe_match_budget/run_extra_23.sh 6 vilau_256 mc2_b16_384
```

The aliases `unitok_attn`, `vilau_256`, `toklip_s_256`, and `toklip_l_384`
map to the existing probe model IDs `unitok`, `vilau`, `toklip_s`, and
`toklip_l`. All 23 use the plain Linear head.

Run on several GPUs with static round-robin sharding:

```bash
MATCH_BUDGET_GPUS=0,1,2,3 bash scripts/linear_probe_match_budget/run_all.sh
```

Run selected models (model ids or ranks):

```bash
bash scripts/linear_probe_match_budget/run_all.sh siglip2_sm14_384 38
```

Dry-run the complete panel:

```bash
MATCH_BUDGET_DRY_RUN=1 bash scripts/linear_probe_match_budget/run_all.sh
```

Useful environment overrides are `MATCH_BUDGET_DATA`,
`MATCH_BUDGET_EXTRA`, `MATCH_BUDGET_OUT_ROOT`, `MATCH_BUDGET_NUM_WORKERS`,
`MATCH_BUDGET_EVAL_NUM_WORKERS` (default `0`), `MATCH_BUDGET_CONDA_ENV`
(default `dino`), `MATCH_BUDGET_SEED`, and `MATCH_BUDGET_SUPPORT_SEED`.

Completed results are detected and skipped. Output defaults to
`outputs/vae_linear_probing_match_budget_4shot`.

The approximately 1 PFLOP budget covers the average model's 4,000 training
encoder forwards. The full 50,000-image validation forward pass is separate.
