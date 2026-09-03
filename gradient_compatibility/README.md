# PE pair gradient compatibility probe

This directory implements a LESS-inspired, falsifiable pilot for the
`pe_lang_g14_448` / `pe_core_g14_448` rank reversal.

The controlled interface is:

```text
PE final normalized 32x32 patch tokens [B,1024,1536]
  -> tokenizer-specific MLP3x-GELU projector [1536 -> 1536]
  -> frozen Qwen2.5-1.5B
  -> fresh, shared LoRA-B gradient coordinates in the last four q/v projections
```

PE-Core's released attention pool is deliberately excluded. Both tokenizers expose
the same sequence surface. The projector is warmed up on fixed COCO captions. The
LoRA probes are attached only after warmup, use identical random A matrices, remain
at B=0, and are never optimized.

For each example the saved visual gradient is:

```text
delta_g = gradient(correct image) - gradient(deranged image)
```

The primary domain score is the mean cosine between each generic `delta_g` and the
target domain's mean `delta_g`. Raw-gradient alignment, visual-signal strength, and
target split-half stability are controls.

## Environment

Use the existing `dino` environment. It already contains PyTorch, Transformers,
timm, PyArrow, Datasets, and Safetensors. The implementation does not require PEFT.

```bash
cd /cache/ma-user/VTBenchLab
conda activate dino
python -m unittest discover -s gradient_compatibility/tests -v
```

## End-to-end smoke

The smoke still loads the real 1.9B PE-Core tower and Qwen, but uses only two
examples per split:

```bash
CONFIG=gradient_compatibility/configs/pe_pair_smoke.json \
  bash gradient_compatibility/scripts/run_pe_pair.sh
```

## Formal seed-0 pilot

```bash
bash gradient_compatibility/scripts/run_pe_pair.sh
```

## SigLIP2-B/16-512 versus MetaCLIP1-bigG/14

This pair uses each timm tower's final normalized native patch grid: SigLIP2
returns `[B,1024,768]` and MetaCLIP1 returns `[B,256,1664]`.  The projectors map
both representations into the same Qwen hidden width; outputs are isolated from
the PE experiment.

```bash
bash gradient_compatibility/scripts/run_siglip2_mc1_pair.sh
```

## RAE v2, SigLIP2-B/16-256, and MetaCLIP1-B/16

The RAE v2 representation is its decoder-facing normalized K=23 DINOv3-L
spatial latent, not a plain final-layer DINO feature. The three-model summary
reports the predicted order and Spearman correlation against the known Qwen2.5
order.

```bash
bash gradient_compatibility/scripts/run_raev2_siglip2_mc1_trio.sh
```

Every stage is resumable at completed-stage granularity. Select stages explicitly:

```bash
STAGES=manifest,tokens bash gradient_compatibility/scripts/run_pe_pair.sh
STAGES=warmup SEEDS=0 bash gradient_compatibility/scripts/run_pe_pair.sh
STAGES=probe,summary SEEDS=0 bash gradient_compatibility/scripts/run_pe_pair.sh
```

Run the two tokenizers separately if desired:

```bash
TOKENIZERS=pe_lang_g14_448 STAGES=tokens,warmup,probe \
  bash gradient_compatibility/scripts/run_pe_pair.sh
TOKENIZERS=pe_core_g14_448 STAGES=tokens,warmup,probe \
  bash gradient_compatibility/scripts/run_pe_pair.sh
STAGES=summary bash gradient_compatibility/scripts/run_pe_pair.sh
```

After seed 0 works, add projector seeds without rebuilding token caches:

```bash
STAGES=warmup,probe,summary SEEDS="0 1 2" \
  bash gradient_compatibility/scripts/run_pe_pair.sh
```

## Artifacts

The formal config writes under `artifacts/pe_pair_pilot/`:

```text
manifest/{records.jsonl,manifest.json}
tokens/<tokenizer>/{cache.json,index.json,shards/*.safetensors}
projectors/<tokenizer>/seed_<seed>/projector_seen_*.pt
gradients/<tokenizer>/seed_<seed>/{generic,caption,vqa,ocr,reasoning}.safetensors
summary/{domain_scores.csv,summary.json,summary.md}
```

The unweighted average across domains is explicitly exploratory. With only two
tokenizers, this run can test whether the proxy recovers the reversal direction and
whether its mechanism is stable; it cannot establish a meaningful rank correlation.

## Reliability-gated loss proxy

The original gradient-cosine score fails both diagnostic cohorts. The follow-up
proxy uses gradient split-half stability only as a label-free domain gate, then
ranks post-warmup correct-image validation loss in the reliable domains. With a
fixed threshold of `0.2`, caption and reasoning are retained and VQA/OCR are
rejected. At 4096 warmup examples this reproduces both known test orders.

That agreement is not yet evidence, and the rule is carried forward as a
pre-registered bet rather than a validated metric. `pe_core_g14_448` diverged rather
than lost, the `reasoning` domain carries `~0.016` nats of visual signal against a
`0.97` nat between-tokenizer spread, the deciding margins are below the protocol's own
`0.09`-`0.19` nat run-to-run variance, and the raw `real` loss carries a
sequence-length term. See `artifacts/reliability_gated_loss_proxy_report.md` for the
numbers. Do not tune the metric further on these five models.

```bash
python -m gradient_compatibility.evaluate_losses \
  --config gradient_compatibility/configs/raev2_siglip2_mc1_trio_pilot.json \
  --domains caption reasoning --batch-size 8

python -m gradient_compatibility.summarize_loss_proxy \
  --config gradient_compatibility/configs/raev2_siglip2_mc1_trio_pilot.json \
  --loss-json gradient_compatibility/artifacts/raev2_siglip2_mc1_trio_pilot/analysis/loss_probe_seed0_seen4096_full_caption-reasoning.json
```

## Frozen 79-tokenizer sweep

The full sweep freezes the choices made on the five calibration tokenizers:

- 4096 COCO-caption projector warmup examples;
- caption and ScienceQA reasoning validation losses (256 examples each);
- the same three-layer GELU projector, optimizer, frozen Qwen2.5-1.5B, prompts,
  seed, and transforms;
- mean within-domain rank of correct-image loss as the final proxy.

The requested model pool is frozen in `configs/full_sweep_models.txt`. Each
registered loader now exposes its native spatial sequence rather than the pooled
linear-probe feature. The blind `predictions.json` is atomically written before
the MLLM CSV is opened; the primary correlation excludes the five calibration
tokenizers.

One A100:

```bash
cd /cache/ma-user/VTBenchLab
GPU_IDS=0 conda run --no-capture-output -n dino \
  bash gradient_compatibility/scripts/run_full_sweep.sh
```

Multiple A100s partition the 79 models deterministically:

```bash
GPU_IDS=0,1,2,3 conda run --no-capture-output -n dino \
  bash gradient_compatibility/scripts/run_full_sweep.sh
```

The same command is model-level resumable. By default, each reconstructible token
cache is removed only after that tokenizer's projector and loss result are safely
written. Set `KEEP_TOKEN_CACHE=1` to retain them. Set `REVEAL_MLLM=0` to stop after
blind predictions. Final files are:

```text
gradient_compatibility/artifacts/full_sweep_v1/summary/predictions.{json,csv}
gradient_compatibility/artifacts/full_sweep_v1/summary/evaluation.{json,csv}
gradient_compatibility/artifacts/full_sweep_v1/logs/worker_*.log
```

### Sequence-length capture

`--clean-token-cache` removes `tokens/<name>/cache.json`, which is the only record of
each tokenizer's visual sequence length — the main confound for a caption-NLL proxy.
Run this read-only poller alongside the sweep to capture shapes in the window between
cache completion and deletion:

```bash
conda run --no-capture-output -n dino \
  python -m gradient_compatibility.capture_token_shapes \
    --config gradient_compatibility/configs/full_sweep.json
```

It writes `analysis/token_shapes.json` and touches nothing the worker owns. Feature
width is separately recoverable from the surviving projector checkpoints, so only
sequence length is time-critical.

### Diagnostics

`summarize_full_sweep` settles the frozen bet. It cannot say whether the result is
real, so run the diagnostics after it:

```bash
conda run --no-capture-output -n dino \
  python -m gradient_compatibility.analyze_full_sweep \
    --config gradient_compatibility/configs/full_sweep.json \
    --ground-truth-csv lar/configs/e3_targets.csv
```

This writes `analysis/diagnostics.json` and never modifies the blind
`summary/predictions.json`. It reports diverged warmup runs, per-domain visual signal,
sequence-length and feature-width confounds, four score variants
(`frozen_mean_rank`, `caption_real`, `caption_real_minus_zero`,
`caption_real_minus_shuffled`), and — the question that decides whether this proxy is
worth anything — whether the residuals against the MLLM still decompose into per-family
additive offsets the way ImageNet linear probing does. Correlations are reported under
one-per-family, leave-one-family-out, and family-stratified top-1 regret, with a family
coverage table, not as a whole-table Spearman. Add `--allow-partial` to inspect a sweep
that is still running.
