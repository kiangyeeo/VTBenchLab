#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd "${script_dir}/.." && pwd)"

data_root=""
vicuna_path=""
load_path=""
batch_size="16"
output_dir="benchmark_outputs"
tasks="task_eval_coco_count,task_eval_coco_multiclass,task_eval_vcr_count,task_eval_vcr_multiclass,task_eval_coco_caption,task_eval_vqav2"
skip_missing_data="false"
allow_vicuna_mismatch="false"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --data-root)
      data_root="$2"
      shift 2
      ;;
    --vicuna-path)
      vicuna_path="$2"
      shift 2
      ;;
    --load-path)
      load_path="$2"
      shift 2
      ;;
    --batch-size)
      batch_size="$2"
      shift 2
      ;;
    --output-dir)
      output_dir="$2"
      shift 2
      ;;
    --tasks)
      tasks="$2"
      shift 2
      ;;
    --skip-missing-data)
      skip_missing_data="true"
      shift
      ;;
    --allow-vicuna-mismatch)
      allow_vicuna_mismatch="true"
      shift
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

if [[ -z "${data_root}" || -z "${vicuna_path}" || -z "${load_path}" ]]; then
  echo "Usage: $0 --data-root PATH --vicuna-path PATH --load-path PATH [--batch-size 16] [--output-dir benchmark_outputs] [--tasks task_a,task_b] [--skip-missing-data]" >&2
  exit 2
fi

abs_path() {
  if [[ "$1" = /* ]]; then
    echo "$1"
  else
    echo "$(pwd)/$1"
  fi
}

data_root="$(abs_path "${data_root}")"
vicuna_path="$(abs_path "${vicuna_path}")"
load_path="$(abs_path "${load_path}")"
output_dir="$(abs_path "${output_dir}")"

if [[ ! -d "${data_root}" ]]; then
  echo "[error] data root does not exist: ${data_root}" >&2
  exit 1
fi
if [[ ! -d "${vicuna_path}" ]]; then
  echo "[error] Vicuna path does not exist: ${vicuna_path}" >&2
  exit 1
fi
if [[ ! -f "${vicuna_path}/config.json" ]]; then
  echo "[error] Vicuna path is missing config.json: ${vicuna_path}" >&2
  exit 1
fi
python - "${vicuna_path}/config.json" "${allow_vicuna_mismatch}" <<'PY'
import json
import sys
from pathlib import Path

config_path = Path(sys.argv[1])
allow_mismatch = sys.argv[2] == "true"
config = json.loads(config_path.read_text(encoding="utf-8"))
name = str(config.get("_name_or_path", "")).lower()

if "vicuna" not in name:
    print(
        "[warn] Vicuna config _name_or_path does not contain 'vicuna': "
        f"{config.get('_name_or_path', '')}",
        file=sys.stderr,
    )

if "v1.5" in name and not allow_mismatch:
    raise SystemExit(
        "[error] The official GVT README asks for FastChat Vicuna-7B but does "
        "not specify v1.5. lmsys/vicuna-7b-v1.5 is Llama-2 based and has "
        "already produced garbled outputs in this pipeline. Use a Llama-1 "
        "based FastChat Vicuna-7B checkpoint, such as the v1.1/original "
        "converted weights, or pass "
        "--allow-vicuna-mismatch only for debugging."
    )
PY
if [[ ! -f "${load_path}" ]]; then
  echo "[error] GVT checkpoint does not exist: ${load_path}" >&2
  exit 1
fi
if [[ ! -d "${data_root}/eval_gt" ]]; then
  echo "[error] eval_gt directory is missing under data root: ${data_root}/eval_gt" >&2
  exit 1
fi

mkdir -p "${output_dir}/pred_results/count" "${output_dir}/output"

echo "[config] data_root=${data_root}"
echo "[config] vicuna_path=${vicuna_path}"
echo "[config] load_path=${load_path}"
echo "[config] batch_size=${batch_size}"
echo "[config] output_dir=${output_dir}"

required_arrow_for_task() {
  case "$1" in
    task_eval_coco_count) echo "coco_oc.arrow" ;;
    task_eval_coco_multiclass) echo "coco_mci.arrow" ;;
    task_eval_vcr_count) echo "vcr_oc.arrow" ;;
    task_eval_vcr_multiclass) echo "vcr_mci.arrow" ;;
    task_eval_coco_caption) echo "coco_caption_karpathy_val.arrow" ;;
    task_eval_vqav2) echo "vqav2_rest_val.arrow" ;;
    *) echo "" ;;
  esac
}

task_list_contains_caption="false"
IFS=',' read -ra task_list <<< "${tasks}"
for task in "${task_list[@]}"; do
  task="$(echo "${task}" | xargs)"
  if [[ "${task}" == "task_eval_coco_caption" ]]; then
    task_list_contains_caption="true"
  fi
done

if [[ "${task_list_contains_caption}" == "true" ]]; then
  if ! command -v java >/dev/null 2>&1; then
    echo "[error] task_eval_coco_caption requires Java for PTBTokenizer/SPICE, but 'java' was not found." >&2
    echo "        Install Java first, for example: conda install -c conda-forge openjdk=8" >&2
    exit 1
  fi
  echo "[prepare] COCO Caption Java evaluator dependencies"
  "${project_root}/scripts/download_eval_deps.sh"
  required_caption_jars=(
    "${project_root}/gvt/modules/evaluations/tokenizer/stanford-corenlp-3.4.1.jar"
    "${project_root}/gvt/modules/evaluations/spice/spice-1.0.jar"
    "${project_root}/gvt/modules/evaluations/spice/lib/ejml-0.23.jar"
    "${project_root}/gvt/modules/evaluations/spice/lib/slf4j-api-1.7.12.jar"
    "${project_root}/gvt/modules/evaluations/spice/lib/slf4j-simple-1.7.21.jar"
    "${project_root}/gvt/modules/evaluations/spice/lib/lmdbjni-0.4.6.jar"
    "${project_root}/gvt/modules/evaluations/spice/lib/lmdbjni-linux64-0.4.6.jar"
    "${project_root}/gvt/modules/evaluations/spice/lib/lmdbjni-osx64-0.4.6.jar"
    "${project_root}/gvt/modules/evaluations/spice/lib/lmdbjni-win64-0.4.6.jar"
    "${project_root}/gvt/modules/evaluations/spice/lib/fst-2.47.jar"
    "${project_root}/gvt/modules/evaluations/spice/lib/jackson-core-2.5.3.jar"
    "${project_root}/gvt/modules/evaluations/spice/lib/javassist-3.19.0-GA.jar"
    "${project_root}/gvt/modules/evaluations/spice/lib/objenesis-2.4.jar"
    "${project_root}/gvt/modules/evaluations/spice/lib/guava-19.0.jar"
    "${project_root}/gvt/modules/evaluations/spice/lib/json-simple-1.1.1.jar"
    "${project_root}/gvt/modules/evaluations/spice/lib/Meteor-1.5.jar"
    "${project_root}/gvt/modules/evaluations/spice/lib/SceneGraphParser-1.0.jar"
    "${project_root}/gvt/modules/evaluations/spice/lib/stanford-corenlp-3.6.0.jar"
    "${project_root}/gvt/modules/evaluations/spice/lib/stanford-corenlp-3.6.0-models.jar"
  )
  for jar in "${required_caption_jars[@]}"; do
    if [[ ! -f "${jar}" ]]; then
      echo "[error] task_eval_coco_caption requires COCO Caption Java jars." >&2
      echo "        Missing: ${jar}" >&2
      echo "        The run script tried to prepare them automatically; check network access and retry." >&2
      exit 1
    fi
  done
fi

for task in "${task_list[@]}"; do
  task="$(echo "${task}" | xargs)"
  [[ -z "${task}" ]] && continue

  required_arrow="$(required_arrow_for_task "${task}")"
  if [[ -n "${required_arrow}" && ! -f "${data_root}/${required_arrow}" ]]; then
    if [[ "${skip_missing_data}" == "true" ]]; then
      echo "[skip] ${task}: missing ${data_root}/${required_arrow}"
      continue
    fi
    echo "[error] ${task}: missing ${data_root}/${required_arrow}" >&2
    exit 1
  fi

  echo "[eval] ${task}"
  (
    cd "${project_root}"
    python run.py with \
      "${task}" \
      num_gpus=1 num_nodes=1 \
      test_only=True \
      test_on_val=True \
      image_size=224 \
      num_latents=32 \
      per_gpu_batchsize="${batch_size}" \
      data_root="${data_root}" \
      vicuna_path="${vicuna_path}" \
      load_path="${load_path}" \
      output_dir="${output_dir}" \
      log_dir="${output_dir}/output"
  )
done

python "${project_root}/../tools/collect_results.py" \
  --result-dir "${output_dir}/pred_results" \
  --output-json "${output_dir}/summary_results.json" \
  --output-md "${output_dir}/summary_results.md"
