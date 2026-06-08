# Quick Start: GVTBench
**1. Environment Setup**

```bash
cd GVT/gvt
pip install -r requirements.txt
pip install setuptools==59.5.0 protobuf==3.20.3
conda install -c conda-forge openjdk=8
```

**2. Download Model Weights**

```bash
GVT/gvt/scripts/download_weights.sh \
  --output-dir /path/to/your/checkpoints \
  --retry 10 \
```

**3. Prepare COCO Caption Evaluator Dependencies**

```bash
GVT/gvt/scripts/download_eval_deps.sh
```

**4. No-VCR Running**

```bash
GVT/gvt/scripts/run_all_eval.sh \
  --data-root /path/to/your/data/gvt/arrow \
  --vicuna-path /path/to/your/checkpoints/vicuna-7b-v1.1 \
  --load-path /path/to/your/checkpoints/gvt.pth \
  --batch-size 16 \
  --output-dir /path/to/your/outputs/gvt_eval_v11 \
  --tasks task_eval_coco_count,task_eval_coco_multiclass,task_eval_coco_caption,task_eval_vqav2 \
  --skip-missing-data
```

**5. VCR Running**

```bash
GVT/gvt/scripts/run_all_eval.sh \
  --data-root /path/to/your/data/gvt/arrow \
  --vicuna-path /path/to/your/checkpoints/vicuna-7b-v1.1 \
  --load-path /path/to/your/checkpoints/gvt.pth \
  --batch-size 16 \
  --output-dir /path/to/your/outputs/gvt_eval_v11 \
  --tasks task_eval_vcr_count,task_eval_vcr_multiclass
```

**6. Aggregate Results**

```bash
python GVT/tools/collect_results.py \
  --result-dir /path/to/your/outputs/gvt_eval_v11/pred_results \
  --output-json /path/to/your/outputs/gvt_eval_v11/summary_results.json \
  --output-md /path/to/your/outputs/gvt_eval_v11/summary_results.md
```

Final results will be saved in:

```bash
VTBenchLab/outputs/gvt_eval_v11/summary_results.md
```





