#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../../.." && pwd)"

output_dir="${repo_root}/checkpoints"
# GVT's README says Vicuna-7B via FastChat but does not pin a weights version.
# The default below is the safer Llama-1-based public Vicuna-7B candidate; do
# not use lmsys/vicuna-7b-v1.5 unless you intentionally want a Llama-2-based
# mismatch for debugging.
vicuna_repo="lmsys/vicuna-7b-v1.1"
gvt_file_id="14ficAR-WL8M0-rZaAz5bSh2DnzgaSpqS"
retry_count="5"
force_download="false"
hf_endpoint="${HF_ENDPOINT:-}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --output-dir)
      output_dir="$2"
      shift 2
      ;;
    --vicuna-repo)
      vicuna_repo="$2"
      shift 2
      ;;
    --skip-vicuna)
      skip_vicuna="true"
      shift
      ;;
    --skip-gvt)
      skip_gvt="true"
      shift
      ;;
    --retry)
      retry_count="$2"
      shift 2
      ;;
    --force)
      force_download="true"
      shift
      ;;
    --hf-endpoint)
      hf_endpoint="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

skip_vicuna="${skip_vicuna:-false}"
skip_gvt="${skip_gvt:-false}"

mkdir -p "${output_dir}"

retry_cmd() {
  local attempt=1
  local max_attempts="$1"
  shift

  until "$@"; do
    if [[ "${attempt}" -ge "${max_attempts}" ]]; then
      echo "[error] command failed after ${attempt} attempts: $*" >&2
      return 1
    fi
    echo "[warn] command failed, retrying ${attempt}/${max_attempts}: $*" >&2
    sleep $((attempt * 10))
    attempt=$((attempt + 1))
  done
}

if [[ "${skip_vicuna}" != "true" ]]; then
  vicuna_dir="${output_dir}/$(basename "${vicuna_repo}")"
  if [[ "${force_download}" == "true" ]]; then
    rm -rf "${vicuna_dir}"
  fi
  echo "[download] Vicuna from Hugging Face: ${vicuna_repo}"
  echo "[resume] Re-running this command will reuse Hugging Face cache and existing files in ${vicuna_dir}"
  if [[ -n "${hf_endpoint}" ]]; then
    echo "[config] HF_ENDPOINT=${hf_endpoint}"
  fi
  retry_cmd "${retry_count}" env HF_ENDPOINT="${hf_endpoint}" VICUNA_REPO="${vicuna_repo}" VICUNA_DIR="${vicuna_dir}" python -c '
import inspect
import os
from huggingface_hub import snapshot_download

kwargs = {
    "repo_id": os.environ["VICUNA_REPO"],
    "local_dir": os.environ["VICUNA_DIR"],
    "local_dir_use_symlinks": False,
}
if "resume_download" in inspect.signature(snapshot_download).parameters:
    kwargs["resume_download"] = True
snapshot_download(**kwargs)
'
  echo "[ok] Vicuna path: ${vicuna_dir}"
fi

if [[ "${skip_gvt}" != "true" ]]; then
  gvt_path="${output_dir}/gvt.pth"
  gvt_part_path="${gvt_path}.part"
  if [[ -f "${gvt_path}" && "${force_download}" != "true" ]]; then
    echo "[skip] exists: ${gvt_path}"
  else
    if [[ "${force_download}" == "true" ]]; then
      rm -f "${gvt_path}" "${gvt_part_path}"
    fi
  echo "[download] GVT checkpoint from Google Drive"
    echo "[resume] partial file: ${gvt_part_path}"
    gdown_args=(--id "${gvt_file_id}" -O "${gvt_part_path}")
    if gdown --help 2>&1 | grep -q -- "--continue"; then
      gdown_args=(--continue "${gdown_args[@]}")
    else
      echo "[warn] installed gdown does not advertise --continue; retries will restart this file if interrupted" >&2
    fi
    retry_cmd "${retry_count}" gdown "${gdown_args[@]}"
    mv "${gvt_part_path}" "${gvt_path}"
  echo "[ok] GVT checkpoint path: ${gvt_path}"
  fi
fi

echo
echo "Use these evaluation args:"
echo "  --vicuna-path ${output_dir}/$(basename "${vicuna_repo}")"
echo "  --load-path ${output_dir}/gvt.pth"
