#!/usr/bin/env bash
set -euo pipefail

ROOT="/cache/ma-user/VTBenchLab"
EXPORT_DIR="$ROOT/outputs/vae_linear_probing_200shot_features_fp32"
ARCHIVE_DIR="$ROOT/outputs/vae_linear_probing_200shot_features_fp32_archive"
PART_PREFIX="$ARCHIVE_DIR/vae_linear_probing_200shot_features_fp32.tar.zst.part-"

cd "$ROOT"
python scripts/linear_probe_flop_sweep/export_200shot_features.py

mkdir -p "$ARCHIVE_DIR"
if compgen -G "$PART_PREFIX*" > /dev/null; then
    echo "!! archive parts already exist under $ARCHIVE_DIR; refusing to overwrite" >&2
    exit 2
fi

tar --sort=name -C "$EXPORT_DIR" -cf - \
    README.md export_manifest.json manifest.tsv source_indices.npy labels.npy features metadata \
    | zstd -T4 -1 \
    | split -b 10G -d -a 2 - "$PART_PREFIX"

(
    cd "$ARCHIVE_DIR"
    sha256sum vae_linear_probing_200shot_features_fp32.tar.zst.part-* > SHA256SUMS
)

echo ">> archive parts complete: $ARCHIVE_DIR"
echo ">> reassemble with: cat vae_linear_probing_200shot_features_fp32.tar.zst.part-* > vae_linear_probing_200shot_features_fp32.tar.zst"
