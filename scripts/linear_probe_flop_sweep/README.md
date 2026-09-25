# ImageNet FLOP-budget sweep

This experiment covers the canonical 70-tokenizer panel with deterministic,
nested ImageNet training subsets and one shared protocol:

- no data augmentation;
- FP32 frozen features cached once;
- non-affine `BatchNorm1d -> Linear` for every tokenizer;
- 13 learning rates, batch size 1,000, one independent epoch per point;
- full 50,000-image validation set, reusing the earlier compatible validation
  caches when possible.

The default per-class caps are `1,2,5,10,20,50,100`. Every subset is balanced
and nested. A different set can be passed with `--cap-shots`. Encoders without
a pre-existing full-train cache process only the largest requested support set;
when a smaller compatible compact cache exists, only the additional examples
are encoded. Encoders with a complete cache reuse it without extraction.

The first five examples per class exactly preserve the support set from
`vae_linear_probing_match_budget_5shot_noaug_cached_10epoch`. Every cap point
starts a fresh probe and has its own cosine schedule; there is no warm-start
between budgets.

Run all models:

```bash
bash scripts/linear_probe_flop_sweep/run_all.sh
```

For the 150/200-shot continuation, run each model with
`--cap-shots 150,200`; both points share the same 200-shot cache. The all-model
launcher accepts the same setting through the environment:

```bash
SWEEP_CAP_SHOTS=150,200 bash scripts/linear_probe_flop_sweep/run_all.sh
```

To evaluate the existing budget points on a fixed seed-42 uniform sample of
5,000 validation images while reusing all feature caches:

```bash
SWEEP_OUTPUT_ROOT=outputs/vae_linear_probing_flop_sweep_noaug_cached_1epoch_allbn_val5k_seed42 \
SWEEP_CAP_SHOTS=1,2,5,10,20,50,100,150,200 \
SWEEP_VALIDATION_SAMPLES=5000 SWEEP_VALIDATION_SEED=42 \
bash scripts/linear_probe_flop_sweep/run_all.sh
```

Watch progress:

```bash
tmux attach -t dino
python scripts/linear_probe_flop_sweep/status.py
```

Outputs and resumable caches are stored under
`outputs/vae_linear_probing_flop_sweep_noaug_cached_1epoch_allbn`. Expensive
full-train feature caches are reused from
`outputs/vae_linear_probing_flop_sweep_noaug_cached_5epoch_allbn/_feature_cache`;
the original five-epoch results are retained unchanged.
