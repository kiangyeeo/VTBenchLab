# Batch-normalized tokenizer linear probing

This directory is the BatchNorm variant of
`scripts/linear_probe_tokenizers`. The original implementation and outputs are
not modified.

The probe used here is:

```text
frozen feature [B,D]
  -> BatchNorm1d(D, affine=False, eps=1e-6)
  -> Linear(D, 1000)
```

The frozen backbone uses the feature microbatch selected by each launcher. Its
FP32 features are concatenated into the full optimization batch of 1024 before
BatchNorm is applied. BatchNorm running statistics are checkpointed and are
used during validation.

## Run one model

```bash
cd /cache/ma-user/VTBenchLab
conda activate dino
CUDA_VISIBLE_DEVICES=0 bash scripts/linear_probe_tokenizers_bn/run_metaclip.sh
```

Replace `run_metaclip.sh` with one of:

- `run_dinov1_vits16.sh`
- `run_dinov2_small.sh`
- `run_mc2_s16_224.sh`
- `run_pe_core_b16_224.sh`
- `run_pe_core_t16_384.sh`
- `run_siglip2_b16_224.sh`
- `run_toklip_s.sh`
- `run_toklip_l.sh`
- `run_unitok.sh`
- `run_vqgan.sh`
- `run_vilau.sh`
- `run_webssl_mae300m_full2b_224.sh`
- `run_webssl_mae700m_full2b_224.sh`
- `run_webssl_mae1b_full2b_224.sh`
- `run_webssl_mae2b_full2b_224.sh`
- `run_webssl_mae3b_full2b_224.sh`

## Fast representative family subset

The following launchers form a relatively fast cross-family subset. The wall
times are approximate measurements from the corresponding completed no-BN runs
on this workspace and are only intended for relative selection:

| Family | Launcher | Previous wall time |
| --- | --- | ---: |
| DINOv1 | `run_dinov1_vits16.sh` | 2.21 h |
| DINOv2 | `run_dinov2_small.sh` | 2.22 h |
| MetaCLIP 1 | `run_metaclip.sh` | 2.22 h |
| MetaCLIP 2 | `run_mc2_s16_224.sh` | 2.31 h |
| Perception Encoder | `run_pe_core_b16_224.sh` | 2.22 h |
| Perception Encoder (tiny) | `run_pe_core_t16_384.sh` | 3.49 h |
| SigLIP 2 | `run_siglip2_b16_224.sh` | 5.33 h |
| WebSSL-MAE | `run_webssl_mae300m_full2b_224.sh` | 3.74 h |

The existing TokLIP, UniTok, VILA-U, and larger WebSSL-MAE launchers remain
available, but were not selected for this fast subset because their completed
no-BN runs took longer.

All WebSSL-MAE launchers use the final normalized encoder CLS token, matching
the official ImageNet linear-probing readout. Corrected results are written to
directories ending in `_cls`; the older patch-mean outputs are preserved under
their original directory names and are never resumed by these launchers.

The WebSSL-MAE launchers retain the feature microbatch defaults used by the
corresponding no-BN runs: 16, 8, 4, 2, and 1024 respectively. Override one
when needed, for example:

```bash
FEATURE_MICROBATCH_SIZE=1 CUDA_VISIBLE_DEVICES=0 \
  bash scripts/linear_probe_tokenizers_bn/run_webssl_mae2b_full2b_224.sh
```

Every run trains the same 13-learning-rate grid as the no-BN baseline. Results
are written under:

```text
outputs/vae_linear_probing_dinov2_single_paperlr_bn/
```

Set `OUT_ROOT`, `DATA`, `EXTRA`, or `NUM_WORKERS` before the command to override
their defaults. Pass `--no-resume` to start from scratch in an otherwise empty
output directory. Do not point this implementation at a no-BN output directory
or checkpoint.
