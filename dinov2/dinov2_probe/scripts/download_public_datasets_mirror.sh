#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../.."

MIRROR="${DATASET_MIRROR_BASE:-}"
ROOT="${DATA_ROOT:-data/dinov2_baseline}"

mkdir -p "$ROOT"
if [[ -z "$MIRROR" ]]; then
  echo "DATASET_MIRROR_BASE is not set. Mirror downloads are site-specific, so this script only documents the expected layout."
  python -m dinov2_probe.check_data --data-root "$ROOT" || true
  exit 0
fi

echo "Mirror base: $MIRROR"
echo "Place/sync public datasets into $ROOT using your mirror's layout."
echo "Then run: python -m dinov2_probe.download_public_datasets --data-root $ROOT"

