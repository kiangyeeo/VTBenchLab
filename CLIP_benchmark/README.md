# Quick Start: CLIP Benchmark
**1. Environment Setup**

```bash
cd CLIP_benchmark
pip install -r requirements.txt
pip install -e .
```

**2. Run the CLIP Benchmark experiment**

```bash
mkdir -p VTBenchLab/outputs/clip_benchmark
mkdir -p VTBenchLab/outputs/clip_benchmark/logs
mkdir -p /path/to/your/tmp/open_clip_cache
mkdir -p /path/to/your/tmp/clip_benchmark_wds_cache_by_dataset

while read -r ds; do
  clean="${ds#wds/}"
  clean="${clean//\//-}"

  cache_dir="/path/to/your/tmp/clip_benchmark_wds_cache_by_dataset/$clean"
  mkdir -p "$cache_dir"

  echo "===== Running $ds ====="

  clip_benchmark eval \
    --pretrained_model ViT-B-32,openai \
    --dataset "$ds" \
    --dataset_root "https://huggingface.co/datasets/clip-benchmark/wds_{dataset_cleaned}/tree/main" \
    --batch_size 64 \
    --num_workers 4 \
    --model_cache_dir /path/to/your/tmp/open_clip_cache \
    --wds_cache_dir "$cache_dir" \
    --output "VTBenchLab/outputs/clip_benchmark/benchmark_{dataset}_{pretrained}_{model}_{language}_{task}.json" \
    --skip_existing \
    > "VTBenchLab/outputs/clip_benchmark/logs/${clean}.log" 2>&1

  echo "===== Finished $ds ====="
done < VTBenchLab/CLIP_benchmark/clip_benchmark/datasets/webdatasets.txtmkdir -p 
```

**3. Aggregate Results**

```bash
clip_benchmark build \
  VTBenchLab/outputs/clip_benchmark/*.json \
  --output VTBenchLab/outputs/clip_benchmark/benchmark.csvxxxxxxxxxx clip_benchmark build \ VTBenchLab/outputs/clip_benchmark/*.json \  --output VTBenchLab/outputs/clip_benchmark/benchmark.csv
```

Final results will be saved in:

```bash
VTBenchLab/outputs/clip_benchmark/benchmark.csv
```





