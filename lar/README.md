# LAR implementation

This directory implements `LAR_spec (1).md` with deterministic row alignment.
Visual and text arrays always have an adjacent `.ids.txt` file, and
`compute_lar.py` refuses to run unless both ID sequences match exactly.

## Fixed conventions

- COCO IDs are sorted numerically. The common caption/answer set contains 4,618 images.
- Caption uses the annotation with the smallest annotation ID for each image.
- Answer uses the `answer_type == "other"` annotation with the smallest question ID,
  then reads `multiple_choice_answer`.
- Qwen pooling is an attention-mask-aware mean of the final hidden layer; padding is excluded.
- All supported vision adapters discard CLS/register/storage tokens and mean-pool spatial tokens.
- Visual arrays are stored as FP32, centered without row L2 normalization, and converted to
  FP64 immediately before SVD.
- `K=min(d,512)` is used for LAR, Lift, m50/m90, VSA, and spectral baselines.
- `eff_rank` is the eigenvalue participation ratio `1/sum(p_lambda^2)`.
- `RankMe` is `exp(entropy(p_sigma))`, using normalized singular values.
- Full `lam`, `r`, `w`, and singular-value arrays are saved under `results/spectra/`.

## Environment check and smoke test

Run from the workspace root:

```bash
cd /cache/ma-user/VTBenchLab
export HF_ENDPOINT=https://hf-mirror.com
python -m unittest discover -s lar/tests -v
python -m lar.extract_text --domain answer --image-set coco4618 --prepare-only
python -m lar.extract_visual --image-set coco4618 --models dinov1_vits16 \
  --device cuda --limit 8 --num-workers 2
```

The `--limit` output has `.limit8` in its name and is only for checking model loading.

## E3 full-pool run

E1 is deprecated and is not a gate. E3 uses only the common COCO-4618 image rows.

First normalize the target table. The local main table supplies Qwen3, Qwen2.5,
ImageNet retrieval, Qwen2.5 pretraining loss, and some epoch-1 probes. Supplement CSVs
must use the canonical columns printed by `--help`; they supply the full pool's
SmolLM2, CKA, A_score, and missing epoch-1 values.

```bash
python -m lar.prepare_e3_targets \
  --supplement lar/configs/e3_targets_available.csv \
  --supplement-only
python -m lar.prepare_e3_models

python -m lar.extract_text --domain caption --image-set coco4618 --model Qwen/Qwen2.5-1.5B
python -m lar.extract_text --domain answer --image-set coco4618 --model Qwen/Qwen2.5-1.5B
python -m lar.extract_visual \
  --models-config lar/configs/models_e3.yaml \
  --image-set coco4618 --num-workers 8 --skip-existing
python -m lar.compute_lar_pool \
  --models-config lar/configs/models_e3.yaml --resume
python -m lar.eval_e3 --strict-metric-pool
```

`compute_lar_pool` loads caption and answer text matrices once and performs one visual
SVD per encoder. It writes each completed row and full spectrum immediately, so
`--resume` is safe after interruption. `eval_e3` produces `results/e3.json`,
`results/e3_lift_curves.png`, and `results/e3_report.md` in one invocation.

To distribute extraction across eight GPUs, use deterministic config-order shards:

```bash
mkdir -p lar/logs
for shard in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES="$shard" python -m lar.extract_visual \
    --models-config lar/configs/models_e3.yaml --image-set coco4618 \
    --num-shards 8 --shard-index "$shard" --num-workers 8 --skip-existing \
    > "lar/logs/extract_e3_gpu${shard}.log" 2>&1 &
done
wait
```

`--strict-metric-pool` requires complete dual-domain metrics for every configured model
but permits unavailable target/baseline cells; their exact coverage remains in JSON.
Use `--strict-pool` only when every target and baseline cell is also complete. Every
random process derives its seed from the JSON-recorded base seed.

## Recompute legacy spectra

Waste is removed. Existing full spectrum sidecars can be upgraded without a visual
or text forward pass:

```bash
python -m lar.recompute_from_csv \
  --input lar/results/lar_metrics.csv \
  --output lar/results/lar_metrics_v2.csv
```
