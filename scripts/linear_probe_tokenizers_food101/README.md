# Balanced Food-101 tokenizer linear probing

This experiment uses the 45 configurations listed in `Tokenizer_set_up.md`
and the same fixed feature surfaces, 13-head learning-rate grid, batch 1024,
SGD, and cosine schedule as `scripts/linear_probe_tokenizers`.

The canonical panel is globally synchronized by epoch. Every tokenizer first
finishes the epoch-1 checkpoint and validation; only then may any tokenizer
start epoch 2. The same barrier is repeated through epoch 10. The optimizer
and cosine horizon stay fixed at ten epochs throughout -- the per-process
cutoff never shortens or restarts the learning-rate schedule.

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

Scaling the existing ImageNet logs gives roughly 65--85 GPU-hours for the full
45-configuration panel before allowing for unusually slow model/dataset
startup. Because the epoch-barrier protocol reloads each backbone once per
epoch, plan roughly 3--4 days on one A100 or 15--22 hours on eight A100s with
the static sharding below. These are engineering estimates, not Food-101 wall
times measured from a completed panel.

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
either the document id or the implementation id. For example, this advances
one tokenizer through epoch 1 while retaining the ten-epoch cosine horizon:

```bash
conda activate dino
CUDA_VISIBLE_DEVICES=0 \
  bash scripts/linear_probe_tokenizers_food101/run_model.sh \
    unitok_attn --epochs 10 --stop-after-epoch 1
```

The canonical coordinator runs the complete manifest epoch by epoch on one
already-visible GPU:

```bash
CUDA_VISIBLE_DEVICES=0 \
  bash scripts/linear_probe_tokenizers_food101/run_panel.sh
```

For eight GPUs, launch one coordinator and give it the complete GPU list. It
starts eight static shards for one epoch, checks every worker exit status, and
starts the next epoch only after all eight succeed:

```bash
FOOD101_PANEL_GPUS=0,1,2,3,4,5,6,7 \
  bash scripts/linear_probe_tokenizers_food101/run_panel.sh
```

`run_panel_epoch.sh` is the one-epoch shard worker used by the coordinator; do
not launch independent ten-epoch shard loops, because they would not provide a
global barrier. Set `FOOD101_PANEL_DRY_RUN=1` to inspect all 10 x 45 scheduled
jobs without starting a model.

If the coordinator stops during epoch N, restart from that same barrier:

```bash
FOOD101_PANEL_START_EPOCH=N \
FOOD101_PANEL_GPUS=0,1,2,3,4,5,6,7 \
  bash scripts/linear_probe_tokenizers_food101/run_panel.sh
```

Models that already completed epoch N validate their saved history and no-op;
unfinished models resume from epoch N-1. The driver refuses to advance a model
by more than one epoch in one invocation, so a mistakenly skipped barrier
fails instead of silently violating the schedule.

All launchers resume by default. Outputs are isolated under
`outputs/food101_linear_probing_dinov2_single_surface/<model>/seed<seed>/`.
Use `metrics_history.jsonl` for epoch-by-epoch rank stability;
`results_eval_linear.json` contains the latest held-out validation table.
`results_test_linear.json` is created only at epoch 10 and contains the
one-head official-test result.

Each planned epoch boundary restores the heads, optimizer, scheduler, and
sampler position. Restarting the process also restarts DataLoader worker
augmentation RNG, so this synchronized trajectory intentionally differs from
one uninterrupted ten-epoch process. An unplanned interruption within an
epoch is not guaranteed to be bitwise identical after resume.

The previous five-tokenizer Food-101 experiment selected readout/LR on the
official test and used a different training protocol, so its absolute scores
and correlations are not valid comparisons for this experiment.
