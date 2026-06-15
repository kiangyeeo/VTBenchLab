# DINOv2 ViT-L/14 k-NN evaluation on ImageNet-1k (A100-80G, CUDA 12.4)

## What you need
- **Model:** `dinov2_vitl14_pretrain.pth` (original Meta backbone, NOT the HF transformers
  format) + config `dinov2/configs/eval/vitl14_pretrain.yaml` (already in repo).
- **Dataset:** ImageNet-1k, both `train` (1,281,167 imgs, the k-NN feature bank) and
  `val` (50,000 imgs), in ImageFolder layout + generated `extra/*.npy` metadata.

## Steps

### 0. Environment
```bash
conda create -n dinov2 python=3.10 -y && conda activate dinov2
cd /cache/ma-user/VTBenchLab/dinov2
pip install -r requirements.txt
pip install -e .                      # installs the dinov2 package
```

### 1. Download the backbone (~1.1 GB)
```bash
bash knn_tools/download_model.sh
# -> /cache/ma-user/VTBenchLab/checkpoints/dinov2/dinov2_vitl14_pretrain.pth
```

### 2. Download + lay out ImageNet-1k via HF mirror (~140 GB cache + ~140 GB output)
```bash
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli login                 # once; ILSVRC/imagenet-1k is gated
python knn_tools/prepare_imagenet.py --out /cache/ma-user/VTBenchLab/data/imagenet1k
# Produces train/ val/ labels.txt in the exact naming DINOv2 requires.
# (Use --repo <non-gated-mirror> to skip the login step.)
```

### 3. Generate the extra metadata (.npy)
```bash
PYTHONPATH=. python knn_tools/dump_extra.py \
    --root  /cache/ma-user/VTBenchLab/data/imagenet1k \
    --extra /cache/ma-user/VTBenchLab/data/imagenet1k/extra
```

### 4. Run k-NN
```bash
bash knn_tools/run_knn.sh
# Results -> /cache/ma-user/VTBenchLab/outputs/dinov2_knn_vitl14/results_eval_knn.json
```
Expected: 20-NN top-1 ≈ **83.5%** (paper). The job prints top-1/top-5 for k=10,20,100,200.

## Notes
- `run/eval/knn.py` is a SLURM/submitit launcher; on a single node we call
  `dinov2/eval/knn.py` directly via `torchrun`. Increase `--nproc_per_node` for multi-GPU.
- `cuml` is only for `log_regression.py`, not k-NN — left optional in requirements.txt.
