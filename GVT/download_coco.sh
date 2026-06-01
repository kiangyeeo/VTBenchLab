#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE="${GVT_DATA_ROOT:-${SCRIPT_DIR}/../data/gvt}"
RAW_DIR="${BASE}/raw"

# Set any of these to 0 to skip a dataset, e.g. DOWNLOAD_VCR=0 bash GVT/download_coco.sh
DOWNLOAD_COCO="${DOWNLOAD_COCO:-1}"
DOWNLOAD_VQA="${DOWNLOAD_VQA:-1}"
DOWNLOAD_KARPATHY="${DOWNLOAD_KARPATHY:-1}"
DOWNLOAD_VCR="${DOWNLOAD_VCR:-1}"

download() {
  local url="$1"
  local out_dir="$2"
  local filename
  local out_path
  filename="$(basename "${url}")"
  out_path="${out_dir}/${filename}"
  mkdir -p "${out_dir}"

  if command -v wget >/dev/null 2>&1; then
    wget -c --tries=3 -P "${out_dir}" "${url}"
  elif command -v curl >/dev/null 2>&1; then
    curl -fL --retry 3 -C - -o "${out_path}" "${url}"
  else
    echo "Error: neither wget nor curl is installed." >&2
    exit 1
  fi

  validate_download "${out_path}" "${url}"
}

validate_download() {
  local path="$1"
  local url="$2"
  local prefix

  if [[ ! -s "${path}" ]]; then
    echo "Error: downloaded file is empty: ${path}" >&2
    exit 1
  fi

  prefix="$(LC_ALL=C head -c 16 "${path}" || true)"
  case "${prefix}" in
    "<?xml"*|"<Error"*|"<!DOCTYPE"*|"<html"*|"<HTML"*)
      echo "Error: ${url} did not return a dataset archive." >&2
      echo "       Saved file looks like an XML/HTML error page: ${path}" >&2
      echo "       For VCR, the official S3 links currently return AccessDenied even after accepting the web license." >&2
      echo "       Download VCR manually from a valid source, then place vcr1annots.zip and vcr1images.zip in $(dirname "${path}")." >&2
      exit 1
      ;;
  esac
}

unzip_into() {
  local zip_path="$1"
  local out_dir="$2"
  mkdir -p "${out_dir}"
  unzip -n "${zip_path}" -d "${out_dir}"
}

echo "GVT data root: ${BASE}"

if [[ "${DOWNLOAD_COCO}" == "1" ]]; then
  COCO_DIR="${RAW_DIR}/coco"
  echo
  echo "[COCO] downloading images and annotations to ${COCO_DIR}"


  download "http://images.cocodataset.org/zips/train2014.zip" "${COCO_DIR}"
  download "http://images.cocodataset.org/zips/val2014.zip" "${COCO_DIR}"
  download "http://images.cocodataset.org/zips/val2017.zip" "${COCO_DIR}"
  download "http://images.cocodataset.org/annotations/annotations_trainval2014.zip" "${COCO_DIR}"
  download "http://images.cocodataset.org/annotations/annotations_trainval2017.zip" "${COCO_DIR}"

  unzip_into "${COCO_DIR}/train2014.zip" "${COCO_DIR}"
  unzip_into "${COCO_DIR}/val2014.zip" "${COCO_DIR}"
  unzip_into "${COCO_DIR}/val2017.zip" "${COCO_DIR}"
  unzip_into "${COCO_DIR}/annotations_trainval2014.zip" "${COCO_DIR}"
  unzip_into "${COCO_DIR}/annotations_trainval2017.zip" "${COCO_DIR}"
fi

if [[ "${DOWNLOAD_VQA}" == "1" ]]; then
  VQA_DIR="${RAW_DIR}/vqa"
  echo
  echo "[VQAv2] downloading validation questions and annotations to ${VQA_DIR}"

  download "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Val_mscoco.zip" "${VQA_DIR}"
  download "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip" "${VQA_DIR}"

  unzip_into "${VQA_DIR}/v2_Questions_Val_mscoco.zip" "${VQA_DIR}"
  unzip_into "${VQA_DIR}/v2_Annotations_Val_mscoco.zip" "${VQA_DIR}"
fi

if [[ "${DOWNLOAD_KARPATHY}" == "1" ]]; then
  KARPATHY_DIR="${RAW_DIR}/coco/karpathy"
  echo
  echo "[Karpathy COCO captions] downloading split json to ${KARPATHY_DIR}"

  download "https://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip" "${KARPATHY_DIR}"
  unzip_into "${KARPATHY_DIR}/caption_datasets.zip" "${KARPATHY_DIR}"
fi

if [[ "${DOWNLOAD_VCR}" == "1" ]]; then
  VCR_DIR="${RAW_DIR}/vcr"
  echo
  echo "[VCR] downloading annotations and images to ${VCR_DIR}"
  echo "[VCR] by downloading, you are responsible for complying with the VCR dataset license: https://visualcommonsense.com/download/"

  download "https://s3.us-west-2.amazonaws.com/ai2-rowanz/vcr1annots.zip" "${VCR_DIR}"
  download "https://s3.us-west-2.amazonaws.com/ai2-rowanz/vcr1images.zip" "${VCR_DIR}"

  unzip_into "${VCR_DIR}/vcr1annots.zip" "${VCR_DIR}"
  unzip_into "${VCR_DIR}/vcr1images.zip" "${VCR_DIR}"
fi

echo
echo "Done. Expected raw-data layout:"
echo "  COCO:      ${RAW_DIR}/coco/{train2014,val2014,val2017,annotations}"
echo "  VQAv2:     ${RAW_DIR}/vqa/v2_OpenEnded_mscoco_val2014_questions.json"
echo "             ${RAW_DIR}/vqa/v2_mscoco_val2014_annotations.json"
echo "  Karpathy:  ${RAW_DIR}/coco/karpathy/dataset_coco.json"
echo "  VCR:       ${RAW_DIR}/vcr/vcr1images"
echo
echo "Next: run the Arrow preparation scripts with --save-dir ${BASE}/arrow"
