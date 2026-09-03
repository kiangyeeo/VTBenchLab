#!/usr/bin/env bash
set -euo pipefail

cd /cache/ma-user/VTBenchLab

CONFIG="${CONFIG:-gradient_compatibility/configs/full_sweep.json}"
GPU_IDS="${GPU_IDS:-0}"
STAGES="${STAGES:-tokens,warmup,loss,summary}"
SEED="${SEED:-0}"
LOSS_BATCH_SIZE="${LOSS_BATCH_SIZE:-4}"
KEEP_TOKEN_CACHE="${KEEP_TOKEN_CACHE:-0}"
REVEAL_MLLM="${REVEAL_MLLM:-1}"
TOKENIZERS="${TOKENIZERS:-all}"
LOG_DIR="gradient_compatibility/artifacts/full_sweep_v1/logs"

python -m gradient_compatibility.preflight --config "${CONFIG}"
python -m gradient_compatibility.data --config "${CONFIG}"

IFS=',' read -r -a GPUS <<< "${GPU_IDS}"
NUM_WORKERS="${#GPUS[@]}"
mkdir -p "${LOG_DIR}"

worker_stages="${STAGES//,summary/}"
worker_stages="${worker_stages//summary,/}"
worker_stages="${worker_stages//summary/}"
pids=()
if [[ -n "${worker_stages//,/}" ]]; then
  for index in "${!GPUS[@]}"; do
    gpu="${GPUS[$index]}"
    clean_args=()
    if [[ "${KEEP_TOKEN_CACHE}" != "1" && ",${worker_stages}," == *",loss,"* ]]; then
      clean_args+=(--clean-token-cache)
    fi
    echo "Launching worker ${index}/${NUM_WORKERS} on physical GPU ${gpu}"
    CUDA_VISIBLE_DEVICES="${gpu}" PYTHONUNBUFFERED=1 \
      python -m gradient_compatibility.sweep_worker \
        --config "${CONFIG}" \
        --device cuda:0 \
        --worker-index "${index}" \
        --num-workers "${NUM_WORKERS}" \
        --tokenizers ${TOKENIZERS} \
        --stages "${worker_stages}" \
        --seed "${SEED}" \
        --loss-batch-size "${LOSS_BATCH_SIZE}" \
        "${clean_args[@]}" \
        > >(tee "${LOG_DIR}/worker_${index}.log") 2>&1 &
    pids+=("$!")
  done

  failed=0
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      failed=1
    fi
  done
  if [[ "${failed}" != "0" ]]; then
    echo "At least one worker failed. Re-run the same command after inspecting ${LOG_DIR}." >&2
    exit 1
  fi
fi

if [[ ",${STAGES}," == *",summary,"* ]]; then
  summary_args=()
  if [[ "${REVEAL_MLLM}" == "1" ]]; then
    summary_args+=(--ground-truth-csv lar/configs/e3_targets.csv)
  fi
  python -m gradient_compatibility.summarize_full_sweep \
    --config "${CONFIG}" "${summary_args[@]}"
fi
