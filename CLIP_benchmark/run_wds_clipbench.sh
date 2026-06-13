#!/usr/bin/env bash
set -u

REPO_ROOT="${REPO_ROOT:-/cache/ma-user/VTBenchLab}"
PROJECT_DIR="${PROJECT_DIR:-$REPO_ROOT/CLIP_benchmark}"
MODEL_LIST="${MODEL_LIST:-$PROJECT_DIR/model_lists/models.txt}"
DATASET_LIST="${DATASET_LIST:-$PROJECT_DIR/clip_benchmark/datasets/webdatasets.txt}"
MODEL_CACHE_DIR="${MODEL_CACHE_DIR:-/cache/ma-user/tmp/open_clip_cache}"
WDS_CACHE_ROOT="${WDS_CACHE_ROOT:-/cache/ma-user/tmp/clip_benchmark_wds_cache_by_dataset}"
OUT_DIR="${OUT_DIR:-$REPO_ROOT/outputs/clip_benchmark}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

# Conservative defaults. Override when stable, e.g. BATCH_SIZE=256 MAX_WORKERS=6 bash run_wds_clipbench.sh
BATCH_SIZE="${BATCH_SIZE:-128}"
MAX_WORKERS="${MAX_WORKERS:-4}"

mkdir -p "$MODEL_CACHE_DIR" "$WDS_CACHE_ROOT" "$OUT_DIR/logs"

echo "MODEL_LIST=$MODEL_LIST"
echo "DATASET_LIST=$DATASET_LIST"
echo "MODEL_CACHE_DIR=$MODEL_CACHE_DIR"
echo "WDS_CACHE_ROOT=$WDS_CACHE_ROOT"
echo "OUT_DIR=$OUT_DIR"
echo "HF_ENDPOINT=$HF_ENDPOINT"
echo "BATCH_SIZE=$BATCH_SIZE"
echo "MAX_WORKERS=$MAX_WORKERS"

trap 'echo; echo "Interrupted. Existing JSON results are kept; rerun the script to resume."; exit 130' INT TERM

python "$PROJECT_DIR/clipbench_progress.py" --incomplete-only

while read -r ds; do
  [ -z "$ds" ] && continue

  clean="${ds#wds/}"
  clean="${clean//\//-}"
  cache_dir="$WDS_CACHE_ROOT/$clean"
  mkdir -p "$cache_dir"

  nshards="$(
    curl -L -s --fail "$HF_ENDPOINT/datasets/clip-benchmark/wds_${clean}/raw/main/test/nshards.txt" \
      || true
  )"
  if ! [[ "$nshards" =~ ^[0-9]+$ ]]; then
    nshards=1
  fi
  workers=$(( nshards < MAX_WORKERS ? nshards : MAX_WORKERS ))
  workers=$(( workers < 1 ? 1 : workers ))

  echo "===== Running $ds | shards=$nshards workers=$workers batch=$BATCH_SIZE ====="
  python "$PROJECT_DIR/clipbench_progress.py" --incomplete-only | sed -n '1,4p'

  python -m clip_benchmark.cli eval \
    --pretrained_model "$MODEL_LIST" \
    --dataset "$ds" \
    --dataset_root "$HF_ENDPOINT/datasets/clip-benchmark/wds_{dataset_cleaned}/tree/main" \
    --batch_size "$BATCH_SIZE" \
    --num_workers "$workers" \
    --model_cache_dir "$MODEL_CACHE_DIR" \
    --wds_cache_dir "$cache_dir" \
    --output "$OUT_DIR/benchmark_{dataset}_{pretrained}_{model}_{language}_{task}.json" \
    --skip_existing \
    > "$OUT_DIR/logs/${clean}.log" 2>&1

  status=$?
  if [ "$status" -ne 0 ]; then
    echo "===== Failed $ds with exit code $status; see $OUT_DIR/logs/${clean}.log ====="
  else
    echo "===== Finished $ds ====="
  fi
  python "$PROJECT_DIR/clipbench_progress.py" --incomplete-only | sed -n '1,4p'
done < "$DATASET_LIST"

echo "All dataset commands finished."
python "$PROJECT_DIR/clipbench_progress.py"
