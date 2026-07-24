# Visual Tokenizer Transformer Compatibility

This directory is the independent implementation area for the protocol in
`../visual_tokenizer_transformer_compatibility_no_lm.md`.

The first implemented task is ImageNet-1k single-label classification for:

- `mc1_b16_224_2.5b`
- `mc1_b16_224_400m`

## Fixed representation surface

Both models use exactly the same surface:

```text
224 x 224 image
  -> final MetaCLIP visual Transformer block
  -> native final normalization
  -> remove CLS
  -> retain all 14 x 14 = 196 patch tokens
  -> do not apply the retrieval visual projection
```

The frozen tokenizer therefore returns `[B, 196, 768]`.

The current pooled ImageNet linear-probe implementation is not reused because
it returns one model-selected `[B, D]` feature. The controlled baselines here
all start from the exact same `[B, 196, 768]` sequence.

## Readouts

Every model and seed is evaluated with:

1. `gap_linear`: token mean followed by one linear classifier.
2. `gap_mlp`: token mean followed by a two-layer GELU MLP. Its hidden width is
   chosen automatically to match the Transformer readout's trainable parameter
   count.
3. `transformer`: low-rank `768 -> 128 -> 512` projector, learned CLS token,
   two pre-norm Transformer layers, 8 heads, FFN ratio 4, and a linear
   classifier.

ImageNet is the global-readout task, so the protocol uses one learned CLS token
as specified in section 3.1 of the experiment document. The 8/16 query-token
setting belongs to the later query-conditioned tasks.

No extra positional embedding is added by the readout. The patch-token values
already contain the visual encoder's native positional information.

The Transformer has 6,984,168 trainable parameters. The automatically matched
MLP has 6,985,012 (a 0.012% difference).

## Fixed optimization protocol

- frozen visual tokenizer;
- AdamW, learning rate `3e-4`, weight decay `0.05`;
- global batch size `256`;
- 100-update linear warmup and cosine decay;
- validation at updates `0`, `500`, `2,000`, and `8,000`;
- seeds `0`, `1`, and `2`;
- ImageNet RandomResizedCrop + horizontal flip for training;
- native deterministic resize + center crop for validation;
- Top-1, Top-5, log-update AULC, seed variance, and Transformer gain.

The microbatch can be reduced to fit a GPU. Gradient accumulation always
preserves the configured global batch size.

## Environment and smoke test

The implementation uses the existing `dino` conda environment.

```bash
cd /cache/ma-user/VTBenchLab
conda run --no-capture-output -n dino \
  python transformer_compatibility/smoke_test.py --device cpu
```

This strictly loads both checkpoints and runs all three readouts forward and
backward. It does not train on ImageNet.

## Full controlled experiment

This is the primary command. It runs both models, all three readouts, and all
three seeds, then writes an aggregate summary:

```bash
cd /cache/ma-user/VTBenchLab
bash transformer_compatibility/scripts/run_imagenet_mc1_pair.sh
```

Hardware knobs do not change the global optimization batch:

```bash
MICRO_BATCH_SIZE=16 \
EVAL_BATCH_SIZE=32 \
NUM_WORKERS=8 \
bash transformer_compatibility/scripts/run_imagenet_mc1_pair.sh
```

To run only the new Transformer readout:

```bash
READOUTS=transformer \
bash transformer_compatibility/scripts/run_imagenet_mc1_pair.sh
```

To run one job explicitly:

```bash
conda run --no-capture-output -n dino \
  python transformer_compatibility/train_imagenet.py \
  --model mc1_b16_224_2.5b \
  --readout transformer \
  --seed 0 \
  --micro-batch-size 32 \
  --eval-batch-size 64 \
  --num-workers 8
```

`CUDA_VISIBLE_DEVICES` can be used to place independent jobs on different
GPUs. A run is single-process and uses one visible CUDA device.

## Outputs

Run artifacts are written under:

```text
transformer_compatibility/outputs/imagenet1k/
  <model>/<readout>/seed<seed>/
    protocol.json
    metrics.jsonl
    checkpoint_step500.pt
    checkpoint_step2000.pt
    checkpoint_step8000.pt
    summary.json
```

The launcher writes aggregate files to:

```text
transformer_compatibility/outputs/imagenet1k_summary/
  runs.csv
  aggregate.csv
  transformer_gain.csv
  summary.md
```

To regenerate the summaries:

```bash
conda run --no-capture-output -n dino \
  python transformer_compatibility/summarize_imagenet.py
```

## Extending the experiment

Protocol-critical defaults live in
`configs/imagenet_mc1_protocol.json`. New tokenizer adapters should return the
same sequence contract `[B, N, D]` and declare their surface, token count, grid
shape, checkpoint, and input dimension. Do not modify the existing pooled
linear-probe extractors when adding sequence surfaces.
