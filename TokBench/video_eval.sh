#!/bin/bash
# Video evaluation adapted for a single A100.
#  - one GPU (device 0)
#  - text datasets run SEQUENTIALLY (upstream left `wait` out -> compute raced JSON)
#  - recon dir name unified with video_scripts/resize.sh (video_reconstruction_results)
#  - RES is 256 or 480 (override via env: `RES=480 bash video_eval.sh`)
set -e

# ---- config (override via env) ----
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # .../VisualTokenizer/TokBench
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

DATA_ROOT="${DATA_ROOT:-$REPO_ROOT/tokbench_data}"          # videos/ + video_annotations/
RECON_ROOT="${RECON_ROOT:-$REPO_ROOT/video_reconstruction_results}"
OUT_DIR="${OUT_DIR:-$REPO_ROOT/video_outputs}"
TOKENIZER_NAME="${MODEL_NAME:-resize}"
RES="${RES:-256}"

cd "$REPO_ROOT"

python check_eval_requirements.py

# ---- text recognition (T-ACC / T-NED) ----
dataset_names=("ch3" "ds")
for dataset_name in "${dataset_names[@]}"; do
  python eval_text.py \
    --img_folder "${RECON_ROOT}/${TOKENIZER_NAME}/text_data/${dataset_name}_${RES}/" \
    --gt_path    "${DATA_ROOT}/video_annotations/text_${dataset_name}.json" \
    --dataset "${dataset_name}" \
    --data_type "video" \
    --batch_size 64 \
    --method_name "$TOKENIZER_NAME" \
    --setting "$RES" \
    --save_dir "$OUT_DIR"
done

# ---- face similarity (F-Sim) ----
python eval_face.py \
    --original_image_path "${DATA_ROOT}/videos/face_data/face_clip_3s" \
    --reconstruction_image_path "${RECON_ROOT}/${TOKENIZER_NAME}/face_data/face_clip_3s_${RES}/" \
    --tokenizer "$TOKENIZER_NAME" \
    --data_type "video" \
    --meta_path "${DATA_ROOT}/video_annotations/videoface_meta.json" \
    --setting "$RES" \
    --save_dir "$OUT_DIR"

# ---- aggregate by scale bucket ----
python compute_all_metrics.py --setting "$RES" --data_type video --output_path "$OUT_DIR"
