# Tokenizer ImageNet linear probing

This directory implements a single-surface tokenizer baseline derived from the
DINOv2 linear-head optimization protocol. It intentionally does not call or
modify `dinov2/eval/linear.py`, whose `{1,4}`-block and CLS/patch-average search
is specific to ViT/DINO-style intermediate features.

## Fixed protocol


| Parameter | Value |
|---|---|
| Dataset | ImageNet-1k train / validation |
| Hardware | one visible GPU, one process |
| Optimization global batch | 1024 |
| Frozen-backbone feature microbatch | 256; 4 chunks concatenated to `[1024,D]` before the heads |
| Gradient accumulation | 1 (none) |
| Updates | 12,500 (`10 x 1,250`) |
| Loss | cross entropy |
| Optimizer | SGD, momentum 0.9, weight decay 0 |
| Schedule | cosine to zero, no warm-up |
| Linear head | bias enabled; weight `N(0, 0.01)`; zero bias |
| Linear-head precision | FP32 |
| CUDA math | TF32 enabled consistently for matmul and cuDNN |
| Feature normalization | none |
| Class weighting | none |
| Seed | 0 |
| Validation | batch 256; every 1,250 updates and after update 12,500 |

The 13 paper base learning rates are:

```text
0.0001 0.0002 0.0005 0.001 0.002 0.005 0.01 0.02 0.05 0.1 0.2 0.3 0.5
```

DINOv2 scaling is retained: `effective_lr = base_lr * global_batch / 256`.
At batch 1024 the effective grid is:

```text
0.0004 0.0008 0.002 0.004 0.008 0.02 0.04 0.08 0.2 0.4 0.8 1.2 2.0
```

The DataLoader still produces one optimization batch of 1,024 images.  The
frozen backbone processes that CPU batch in four 256-image chunks, and the
resulting ordinary FP32 tensors are concatenated to one `[1024,D]` tensor before
all 13 heads run.  There is one loss/backward/optimizer/scheduler step per 1,024
images; feature microbatching is not gradient accumulation and does not change
the LR scaling or update count.  The best validation top-1 head is reported.
`protocol.json` records the complete configuration and prevents incompatible
checkpoints from being mixed.

## Fixed representations

| Launcher | Representation |
|---|---|
| `run_metaclip.sh` | final normalized CLS before MetaCLIP's 768-to-512 projection |
| `run_toklip_s.sh` | mean of final normalized TokLIP-S semantic tokens |
| `run_toklip_l.sh` | mean of final normalized TokLIP-L semantic tokens |
| `run_unitok.sh` | quantized/dequantized tokens, then mean and `fc_norm`, before projection |
| `run_vilau.sh` | mean of penultimate VILA-U SigLIP block tokens |

The token means above define the selected tokenizer representations. They are
not an additional DINOv2 avgpool readout candidate. TokLIP deliberately does
not use the CLIP pipeline's `forward_head` representation.

Training augmentation is uniformly RandomResizedCrop plus horizontal flip.
Native model resolution and normalization are retained. MetaCLIP uses its
native deterministic validation transform but no training ColorJitter.
MetaCLIP, TokLIP and UniTok backbones run in FP32; VILA-U runs in BF16. Every
selected representation is converted to FP32 before entering its linear head.
TF32 is explicitly enabled for all five jobs, matching DINOv2's CUDA setting
and preventing tokenizer imports from silently changing this per model.

## Commands

Run the jobs sequentially on one GPU:

```bash
conda activate dino
cd /cache/ma-user/VTBenchLab

CUDA_VISIBLE_DEVICES=0 bash scripts/linear_probe_tokenizers/run_metaclip.sh
CUDA_VISIBLE_DEVICES=0 bash scripts/linear_probe_tokenizers/run_toklip_s.sh
CUDA_VISIBLE_DEVICES=0 bash scripts/linear_probe_tokenizers/run_toklip_l.sh
CUDA_VISIBLE_DEVICES=0 bash scripts/linear_probe_tokenizers/run_unitok.sh
CUDA_VISIBLE_DEVICES=0 bash scripts/linear_probe_tokenizers/run_vilau.sh
```

Defaults can be changed without altering the optimization protocol:

```bash
DATA=/path/to/imagenet1k \
EXTRA=/path/to/imagenet1k/extra \
OUT_ROOT=/path/to/new/output/root \
NUM_WORKERS=8 \
CUDA_VISIBLE_DEVICES=0 \
bash scripts/linear_probe_tokenizers/run_metaclip.sh
```

The default output root is
`outputs/vae_linear_probing_dinov2_single_paperlr/`, with the original five
representation-specific directory names below it. Existing
`outputs/vae_linear_probing/` runs are not read or modified.
