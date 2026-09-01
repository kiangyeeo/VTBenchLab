# LAR implementation

This directory implements `LAR_spec (1).md` with deterministic row alignment.
Visual and text arrays always have an adjacent `.ids.txt` file, and
`compute_lar.py` refuses to run unless both ID sequences match exactly.

## Fixed conventions

- COCO IDs are sorted numerically. The common caption/answer set contains 4,618 images.
- Caption uses the annotation with the smallest annotation ID for each image.
- Answer uses the `answer_type == "other"` annotation with the smallest question ID,
  then reads `multiple_choice_answer`.
- ImageNet-10k uses `random.Random(0).sample` over sorted validation paths.
- Qwen pooling is an attention-mask-aware mean of the final hidden layer; padding is excluded.
- All supported vision adapters discard CLS/register/storage tokens and mean-pool spatial tokens.
- `K=min(rank,512)` is used for all LAR/Waste/spectral-baseline scalar calculations.
- `eff_rank` is the eigenvalue participation ratio `1/sum(p_lambda^2)`.
- `RankMe` is `exp(entropy(p_sigma))`, using normalized singular values.
- Full `lam`, `r`, `w`, and singular-value arrays are saved under `results/spectra/`.

## Environment check and smoke test

Run from the workspace root:

```bash
cd /cache/ma-user/VTBenchLab
python -m unittest discover -s lar/tests -v
python -m lar.extract_text --domain answer --image-set coco4618 --prepare-only
python -m lar.extract_visual --image-set coco4618 --models dinov1_vits16 \
  --device cuda --limit 8 --num-workers 2
```

The `--limit` output has `.limit8` in its name and is only for checking model loading.

## Stage E1

First encode the ImageNet prompts. The default model ID downloads from Hugging Face if
it is not already cached; use a local directory with `--model /path/to/Qwen2.5-1.5B`
when the compute node has no network access.

```bash
python -m lar.extract_text --domain imagenet --image-set in1k10k \
  --model Qwen/Qwen2.5-1.5B --batch-size 64
python -m lar.extract_visual --image-set in1k10k --num-workers 8
```

Compute ImageNet Waste for the six enabled models:

```bash
for model in dinov1_vits16 dinov2_small metaclip_b16_2pt5b pe_core_b16_224 webssl_mae300m_full2b_224 toklip_s_semantic_256; do
  python -m lar.compute_lar \
    --feature "lar/features/${model}__in1k10k.npy" \
    --text lar/text/imagenet__in1k10k.npy \
    --name "$model" --text-domain imagenet --image-set in1k10k
done

python -m lar.prepare_e1_probes
python -m lar.eval_e1 --probes lar/configs/e1_probes.csv
```

Stop if `lar/results/e1.json` does not pass the specified rho threshold.

Important: only the WebSSL-MAE and TokLIP probing rows currently use exactly the same
patch-mean representation as LAR. The other four existing BN/no-BN results use CLS,
CLS+patch, or attention pooling. `e1_probes.csv` marks this explicitly. Those four rows
are useful for an exploratory check, but a strict E1 requires rerunning both probing
heads on the same patch-mean features.

## Stages E2 and E3

Extract each text domain once and each model's visual features once:

```bash
python -m lar.extract_text --domain caption --image-set coco4618 --model Qwen/Qwen2.5-1.5B
python -m lar.extract_text --domain answer --image-set coco4618 --model Qwen/Qwen2.5-1.5B
python -m lar.extract_visual --image-set coco4618 --num-workers 8

for model in dinov1_vits16 dinov2_small metaclip_b16_2pt5b pe_core_b16_224 webssl_mae300m_full2b_224 toklip_s_semantic_256; do
  for domain in caption answer; do
    python -m lar.compute_lar \
      --feature "lar/features/${model}__coco4618.npy" \
      --text "lar/text/${domain}__coco4618.npy" \
      --name "$model" --text-domain "$domain" --image-set coco4618
  done
done
```

Run the required caption-5k stability check separately:

```bash
python -m lar.extract_text --domain caption --image-set coco5000 --model Qwen/Qwen2.5-1.5B
python -m lar.extract_visual --image-set coco5000 --num-workers 8

for model in dinov1_vits16 dinov2_small metaclip_b16_2pt5b pe_core_b16_224 webssl_mae300m_full2b_224 toklip_s_semantic_256; do
  python -m lar.compute_lar \
    --feature "lar/features/${model}__coco5000.npy" \
    --text lar/text/caption__coco5000.npy \
    --name "$model" --text-domain caption --image-set coco5000
done
```

E2 expects a CSV containing at least `name,family,MLLM_Avg`:

```bash
python -m lar.eval_e2 --targets /path/to/targets.csv \
  --metric LAR_64 --target MLLM_Avg
```

E3 expects `name,family,MLLM_Avg` plus any requested baseline columns:

```bash
python -m lar.eval_e3 --targets /path/to/targets.csv \
  --target MLLM_Avg --lar-metric LAR_64

python -m lar.eval_e3 --targets /path/to/targets.csv \
  --pc1-columns llm_a llm_b llm_c --lar-metric LAR_64 \
  --output lar/results/e3_pc1.json
```

The ImageNet-50k/two-stage fallback is deliberately not activated until the
caption-4618 versus caption-5000 stability check fails. The current scripts preserve
the data and spectra needed to make that decision first.
