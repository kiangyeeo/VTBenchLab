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

The frozen backbone still runs in microbatches of 256. Its FP32 features are
concatenated into the full optimization batch of 1024 before BatchNorm is
applied. BatchNorm running statistics are checkpointed and are used during
validation.

## Run one model

```bash
cd /cache/ma-user/VTBenchLab
conda activate dino
CUDA_VISIBLE_DEVICES=0 bash scripts/linear_probe_tokenizers_bn/run_metaclip.sh
```

Replace `run_metaclip.sh` with one of:

- `run_toklip_s.sh`
- `run_toklip_l.sh`
- `run_unitok.sh`
- `run_vilau.sh`
- `run_webssl_mae300m_full2b_224.sh`
- `run_webssl_mae700m_full2b_224.sh`
- `run_webssl_mae1b_full2b_224.sh`
- `run_webssl_mae2b_full2b_224.sh`
- `run_webssl_mae3b_full2b_224.sh`

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
