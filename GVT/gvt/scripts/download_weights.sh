#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../../.." && pwd)"

output_dir="${repo_root}/checkpoints"
vicuna_repo="lmsys/vicuna-7b-v1.5"
gvt_file_id="14ficAR-WL8M0-rZaAz5bSh2DnzgaSpqS"

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
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

skip_vicuna="${skip_vicuna:-false}"
skip_gvt="${skip_gvt:-false}"

mkdir -p "${output_dir}"

if [[ "${skip_vicuna}" != "true" ]]; then
  vicuna_dir="${output_dir}/$(basename "${vicuna_repo}")"
  echo "[download] Vicuna from Hugging Face: ${vicuna_repo}"
  python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='${vicuna_repo}', local_dir='${vicuna_dir}', local_dir_use_symlinks=False)"
  echo "[ok] Vicuna path: ${vicuna_dir}"
fi

if [[ "${skip_gvt}" != "true" ]]; then
  gvt_path="${output_dir}/gvt.pth"
  echo "[download] GVT checkpoint from Google Drive"
  gdown --id "${gvt_file_id}" -O "${gvt_path}"
  echo "[ok] GVT checkpoint path: ${gvt_path}"
fi

echo
echo "Use these evaluation args:"
echo "  --vicuna-path ${output_dir}/$(basename "${vicuna_repo}")"
echo "  --load-path ${output_dir}/gvt.pth"
