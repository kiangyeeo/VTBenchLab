# Quick Start: CLIP Benchmark

This folder runs CLIP-like models through `clip_benchmark`.

Use these two unified input files:

```bash
MODEL_LIST=/cache/ma-user/VTBenchLab/CLIP_benchmark/model_lists/models.txt
DATASET_LIST=/cache/ma-user/VTBenchLab/CLIP_benchmark/clip_benchmark/datasets/webdatasets.txt
```

`DATASET_LIST` currently contains 41 `wds/...` datasets. `MODEL_LIST` contains
OpenAI CLIP plus the selected OpenCLIP/LAION models.

## 1. Environment Setup

```bash
cd /cache/ma-user/VTBenchLab/CLIP_benchmark
pip install -r requirements.txt
pip install -e .
```

## 2. Download Model Weights

`clip_benchmark eval` downloads weights automatically, but this command warms the
model cache first.

```bash
cd /cache/ma-user/VTBenchLab/CLIP_benchmark

export MODEL_LIST=/cache/ma-user/VTBenchLab/CLIP_benchmark/model_lists/models.txt
export MODEL_CACHE_DIR=/cache/ma-user/tmp/open_clip_cache
export HF_ENDPOINT=https://hf-mirror.com
mkdir -p "$MODEL_CACHE_DIR"

python - <<'PY'
import gc
import os
import open_clip

model_list = os.environ["MODEL_LIST"]
cache_dir = os.environ["MODEL_CACHE_DIR"]

with open(model_list) as f:
    models = [line.strip().split(",", 1) for line in f if line.strip()]

for model_name, pretrained in models:
    print(f"Downloading/loading {model_name},{pretrained}")
    model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
        model_name,
        pretrained=pretrained,
        cache_dir=cache_dir,
    )
    del model, preprocess_train, preprocess_val
    gc.collect()
print("Done")
PY
```

## 3. Run All WDS Datasets

This loop runs every dataset in `DATASET_LIST` and gives each dataset its own
WebDataset cache directory.

```bash
cd /cache/ma-user/VTBenchLab/CLIP_benchmark

export REPO_ROOT=/cache/ma-user/VTBenchLab
export MODEL_LIST=/cache/ma-user/VTBenchLab/CLIP_benchmark/model_lists/models.txt
export DATASET_LIST=/cache/ma-user/VTBenchLab/CLIP_benchmark/clip_benchmark/datasets/webdatasets.txt
export MODEL_CACHE_DIR=/cache/ma-user/tmp/open_clip_cache
export WDS_CACHE_ROOT=/cache/ma-user/tmp/clip_benchmark_wds_cache_by_dataset
export OUT_DIR="$REPO_ROOT/outputs/clip_benchmark"
export HF_ENDPOINT=https://hf-mirror.com
export HF_DATASET_ROOT="$HF_ENDPOINT/datasets/clip-benchmark/wds_{dataset_cleaned}/tree/main"

mkdir -p "$MODEL_CACHE_DIR" "$WDS_CACHE_ROOT" "$OUT_DIR/logs"

while read -r ds; do
  [ -z "$ds" ] && continue
  clean="${ds#wds/}"
  clean="${clean//\//-}"
  cache_dir="$WDS_CACHE_ROOT/$clean"
  mkdir -p "$cache_dir"

  echo "===== Running $ds ====="

  python -m clip_benchmark.cli eval \
    --pretrained_model "$MODEL_LIST" \
    --dataset "$ds" \
    --dataset_root "$HF_DATASET_ROOT" \
    --batch_size 64 \
    --num_workers 4 \
    --model_cache_dir "$MODEL_CACHE_DIR" \
    --wds_cache_dir "$cache_dir" \
    --output "$OUT_DIR/benchmark_{dataset}_{pretrained}_{model}_{language}_{task}.json" \
    --skip_existing \
    > "$OUT_DIR/logs/${clean}.log" 2>&1

  echo "===== Finished $ds ====="
done < "$DATASET_LIST"
```

## 4. Aggregate Results

```bash
cd /cache/ma-user/VTBenchLab

PYTHONPATH=/cache/ma-user/VTBenchLab/CLIP_benchmark \
python -m clip_benchmark.cli build \
  /cache/ma-user/VTBenchLab/outputs/clip_benchmark/*.json \
  --output /cache/ma-user/VTBenchLab/outputs/clip_benchmark/benchmark.csv
```
