# Balanced Food-101 tokenizer linear probing

This experiment uses the 45 configurations listed in `Tokenizer_set_up.md`
and the same fixed feature surfaces, 13-head learning-rate grid, batch 1024,
SGD, and cosine schedule as `scripts/linear_probe_tokenizers`.

Food-101 is exactly balanced at 750 official train and 250 official test
images for each of 101 classes. The official test is kept untouched. A fixed
class-stratified split (seed 0, independent of the training seed) holds out
100 images per class from official train:

- train: 65,650 images, 650 per class;
- validation: 10,100 images, 100 per class;
- official test: 25,250 images, 250 per class.

As in the original Food-101 benchmark, official-train annotations contain
known label noise while official-test annotations are cleaned.

The train and held-out validation views load the source twice so that train
uses model-native random augmentation while validation uses the deterministic
evaluation transform. The index hashes and source Hugging Face fingerprints
are written into every `protocol.json`.
Validation selects the LR head; official test is evaluated once using only
that selected head and is never used for selection. A completed result is
detected and skipped on restart; an evaluation interrupted before its result
is written must be rerun.

At the default 10 epochs there are 65 updates per approximate epoch: 66,560
sampled images, 1.39% over one train pass. Training, per-epoch validation, and
one final test total 791,850 frozen-backbone image forwards, about 5.95% of the
current ImageNet-1K protocol.

Scaling the existing ImageNet logs gives roughly 39--42 GPU-hours for the 35
manifest models that already have usable timing traces. The remaining ten
include unmeasured giant/high-resolution models, so the full 45-configuration
panel is strictly more expensive and does not yet have a defensible total
estimate.

## Models

`tokenizers_from_setup.tsv` is the explicit 45-row mapping from the document
configuration ids to probe model ids. The generic probe's extra `metaclip`,
`dinov3`, `raev2`, `ijepa`, and `vqgan` entries are not allowed here. Four
document ids map to shorter implementation names:

- `unitok_attn` -> `unitok`;
- `vilau_256` -> `vilau`;
- `toklip_s_256` -> `toklip_s`;
- `toklip_l_384` -> `toklip_l`.

`clip_meta__l14` and `mc1_l14_224_2.5b` resolve to the same checkpoint and
feature surface. Both are run because both occur in `Tokenizer_set_up.md`, but
correlation and aggregate statistics must retain only one of them. Thus the
manifest has 45 configurations and 44 independent tokenizer points.

The wrapper supplies conservative, model-specific frozen-feature microbatch
defaults (16--1024) so giant/high-resolution encoders do not all attempt a
1024-image backbone forward. This does not change the optimization batch of
1024. Use `--feature-microbatch-size` to lower a default if a model still
exceeds the visible GPU's memory.

## Run

Use the same `dino` environment as the ImageNet probes. `run_model.sh` accepts
either the document id or the implementation id:

```bash
conda activate dino
CUDA_VISIBLE_DEVICES=0 \
  bash scripts/linear_probe_tokenizers_food101/run_model.sh unitok_attn
```

Run the complete manifest sequentially on one visible GPU:

```bash
CUDA_VISIBLE_DEVICES=0 \
  bash scripts/linear_probe_tokenizers_food101/run_panel.sh
```

For eight GPUs, launch one process per GPU with matching shard settings. For
example, GPU 3 runs:

```bash
CUDA_VISIBLE_DEVICES=3 \
FOOD101_PANEL_SHARD_COUNT=8 \
FOOD101_PANEL_SHARD_INDEX=3 \
  bash scripts/linear_probe_tokenizers_food101/run_panel.sh
```

Set `FOOD101_PANEL_DRY_RUN=1` to inspect a shard's assignments without
starting any model.

All launchers resume by default. Outputs are isolated under
`outputs/food101_linear_probing_dinov2_single_surface/<model>/seed<seed>/`.
Use `metrics_history.jsonl` for epoch-by-epoch rank stability;
`results_eval_linear.json` contains the final held-out validation table and
`results_test_linear.json` contains the one-head official-test result.

Checkpoint resume restores the heads, optimizer, scheduler, and sampler
position, but not DataLoader worker augmentation RNG state. Therefore an
interrupted run is not guaranteed to be bitwise identical to an uninterrupted
run with the same seed; seed-stability comparisons should use uninterrupted
runs where practical.

The previous five-tokenizer Food-101 experiment selected readout/LR on the
official test and used a different training protocol, so its absolute scores
and correlations are not valid comparisons for this experiment.
