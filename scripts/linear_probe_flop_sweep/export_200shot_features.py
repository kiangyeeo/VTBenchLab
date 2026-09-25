#!/usr/bin/env python3
"""Export the canonical 70-tokenizer 200-shot FP32 feature panel.

The export has one shared, sorted ImageNet source-index array and one shared
label array. Every feature matrix follows that exact row order. Existing exact
200-shot compact arrays are hard-linked; full-train arrays are subset into
resumable output memmaps.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "scripts/linear_probe_match_budget_5shot_noaug_cached_10epoch/tokenizers.tsv"
CACHE_ROOT = ROOT / "outputs/vae_linear_probing_flop_sweep_noaug_cached_5epoch_allbn/_feature_cache"
RESULT_ROOT = ROOT / "outputs/vae_linear_probing_flop_sweep_noaug_cached_1epoch_allbn"
EXPORT_ROOT = ROOT / "outputs/vae_linear_probing_200shot_features_fp32"
SHOT = 200
EXPECTED_ROWS = 200_000
COPY_ROWS = 2_048


def atomic_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def atomic_text(path: Path, value: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def load_rows():
    rows = []
    for line in MANIFEST.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        rank, requested_id, model, head = line.split("\t")
        rows.append((int(rank), requested_id, model, head))
    if len(rows) != 70 or [row[0] for row in rows] != list(range(1, 71)):
        raise RuntimeError("Expected canonical tokenizer ranks 1..70")
    return rows


def metadata_candidates(requested_id: str):
    candidates = []
    for path in (CACHE_ROOT / requested_id).glob("*/metadata.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            shape = payload["splits"]["train"]["feature_shape"]
            features = Path(payload["splits"]["train"]["features_path"])
            labels = Path(payload["splits"]["train"]["labels_path"])
            if features.is_file() and labels.is_file():
                candidates.append((int(shape[0]), int(shape[1]), path, payload))
        except (KeyError, OSError, ValueError, json.JSONDecodeError):
            continue
    return candidates


def hardlink_or_copy(source: Path, destination: Path) -> None:
    if destination.is_file():
        return
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.unlink(missing_ok=True)
    try:
        os.link(source, temporary)
    except OSError:
        shutil.copyfile(source, temporary)
    os.replace(temporary, destination)


def export_full_subset(
    source: Path,
    destination: Path,
    source_indices: np.ndarray,
    feature_dim: int,
    progress_path: Path,
    progress_label: str,
) -> None:
    expected_shape = (EXPECTED_ROWS, feature_dim)
    if destination.is_file():
        array = np.load(destination, mmap_mode="r")
        if array.shape == expected_shape and array.dtype == np.float32:
            print(f"{progress_label}: reuse completed subset", flush=True)
            return
        raise RuntimeError(f"Invalid existing output array: {destination}")

    partial = destination.with_name(f".{destination.name}.partial")
    completed = 0
    if partial.is_file() and progress_path.is_file():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if progress.get("shape") == list(expected_shape):
            completed = int(progress.get("completed", 0))
    if not 0 <= completed <= EXPECTED_ROWS:
        completed = 0
    if completed == 0:
        output = np.lib.format.open_memmap(
            partial, mode="w+", dtype=np.float32, shape=expected_shape
        )
        atomic_json(progress_path, {"completed": 0, "shape": list(expected_shape)})
    else:
        output = np.load(partial, mmap_mode="r+")
        if output.shape != expected_shape or output.dtype != np.float32:
            raise RuntimeError(f"Invalid partial output array: {partial}")

    full = np.load(source, mmap_mode="r")
    if full.dtype != np.float32 or full.shape[1] != feature_dim:
        raise RuntimeError(f"Invalid source feature array: {source}")
    for start in range(completed, EXPECTED_ROWS, COPY_ROWS):
        stop = min(start + COPY_ROWS, EXPECTED_ROWS)
        output[start:stop] = full[source_indices[start:stop]]
        if stop % (COPY_ROWS * 25) == 0 or stop == EXPECTED_ROWS:
            output.flush()
            atomic_json(progress_path, {"completed": stop, "shape": list(expected_shape)})
            print(
                f"{progress_label}: {stop}/{EXPECTED_ROWS} ({100.0 * stop / EXPECTED_ROWS:.1f}%)",
                flush=True,
            )
    output.flush()
    del output, full
    os.replace(partial, destination)
    progress_path.unlink(missing_ok=True)


def main() -> int:
    rows = load_rows()
    EXPORT_ROOT.mkdir(parents=True, exist_ok=True)
    feature_root = EXPORT_ROOT / "features"
    metadata_root = EXPORT_ROOT / "metadata"
    feature_root.mkdir(exist_ok=True)
    metadata_root.mkdir(exist_ok=True)

    support_hashes = set()
    for _rank, requested_id, _model, _head in rows:
        protocol_path = RESULT_ROOT / requested_id / "cap_0200/protocol.json"
        result_path = RESULT_ROOT / requested_id / "cap_0200/results_eval_linear.json"
        if not protocol_path.is_file() or not result_path.is_file():
            raise RuntimeError(f"Missing completed 200-shot result for {requested_id}")
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        if int(protocol["epochs"]) != 1 or int(protocol["actual_train_samples"]) != EXPECTED_ROWS:
            raise RuntimeError(f"Unexpected 200-shot protocol for {requested_id}")
        support_hashes.add(protocol["support_indices_sha256"])
    if len(support_hashes) != 1:
        raise RuntimeError(f"200-shot support sets are not identical: {support_hashes}")
    support_hash = support_hashes.pop()

    compact_seed = None
    for rank, requested_id, model, head in rows:
        for count, feature_dim, metadata_path, metadata in metadata_candidates(requested_id):
            source_path = metadata["splits"]["train"].get("source_indices_path")
            if count == EXPECTED_ROWS and source_path and Path(source_path).is_file():
                compact_seed = (metadata_path, metadata)
                break
        if compact_seed is not None:
            break
    if compact_seed is None:
        raise RuntimeError("No exact 200-shot compact cache is available")

    seed_metadata_path, seed_metadata = compact_seed
    seed_train = seed_metadata["splits"]["train"]
    source_indices_source = Path(seed_train["source_indices_path"])
    labels_source = Path(seed_train["labels_path"])
    source_indices = np.load(source_indices_source, mmap_mode="r")
    labels = np.load(labels_source, mmap_mode="r")
    if (
        source_indices.shape != (EXPECTED_ROWS,)
        or source_indices.dtype != np.int64
        or labels.shape != (EXPECTED_ROWS,)
        or labels.dtype != np.int64
        or np.any(source_indices[1:] <= source_indices[:-1])
    ):
        raise RuntimeError("Invalid shared support arrays")
    stable_indices = np.asarray(source_indices, dtype="<i8")
    if hashlib.sha256(stable_indices.tobytes(order="C")).hexdigest() != support_hash:
        raise RuntimeError("Shared source indices do not match the 200-shot protocol hash")
    hardlink_or_copy(source_indices_source, EXPORT_ROOT / "source_indices.npy")
    hardlink_or_copy(labels_source, EXPORT_ROOT / "labels.npy")

    manifest_lines = [
        "rank\ttokenizer\tmodel\tfeature_dim\tdtype\trows\tfeature_file\trepresentation"
    ]
    export_entries = []
    for rank, requested_id, model, _head in rows:
        candidates = metadata_candidates(requested_id)
        exact = [item for item in candidates if item[0] == EXPECTED_ROWS]
        full = [item for item in candidates if item[0] == 1_281_167]
        if exact:
            count, feature_dim, source_metadata_path, source_metadata = exact[0]
            train_meta = source_metadata["splits"]["train"]
            compact_indices = np.load(train_meta["source_indices_path"], mmap_mode="r")
            compact_labels = np.load(train_meta["labels_path"], mmap_mode="r")
            if not np.array_equal(compact_indices, source_indices):
                raise RuntimeError(f"Source-index order mismatch for {requested_id}")
            if not np.array_equal(compact_labels, labels):
                raise RuntimeError(f"Label order mismatch for {requested_id}")
            source_features = Path(train_meta["features_path"])
            source_kind = "exact-200shot-hardlink"
        elif full:
            count, feature_dim, source_metadata_path, source_metadata = full[0]
            train_meta = source_metadata["splits"]["train"]
            full_labels = np.load(train_meta["labels_path"], mmap_mode="r")
            if not np.array_equal(full_labels[source_indices], labels):
                raise RuntimeError(f"Full-cache labels mismatch for {requested_id}")
            source_features = Path(train_meta["features_path"])
            source_kind = "subset-from-full-train-cache"
        else:
            raise RuntimeError(f"No usable cache for {requested_id}")

        stem = f"{rank:03d}_{requested_id}"
        destination = feature_root / f"{stem}.npy"
        if exact:
            hardlink_or_copy(source_features, destination)
            array = np.load(destination, mmap_mode="r")
            if array.shape != (EXPECTED_ROWS, feature_dim) or array.dtype != np.float32:
                raise RuntimeError(f"Invalid exact feature array for {requested_id}")
            print(f"[{rank:02d}/70] {requested_id}: linked exact 200-shot cache", flush=True)
        else:
            export_full_subset(
                source_features,
                destination,
                source_indices,
                feature_dim,
                metadata_root / f".{stem}.progress.json",
                f"[{rank:02d}/70] {requested_id}",
            )

        representation = source_metadata.get("identity", {}).get("representation", "")
        entry = {
            "rank": rank,
            "tokenizer": requested_id,
            "model": model,
            "rows": EXPECTED_ROWS,
            "feature_dim": feature_dim,
            "dtype": "float32",
            "feature_file": f"features/{stem}.npy",
            "row_order": "shared source_indices.npy (strictly increasing official ImageNet train indices)",
            "representation": representation,
            "source_kind": source_kind,
            "source_cache_metadata": str(source_metadata_path.resolve()),
            "source_cache_fingerprint": source_metadata.get("fingerprint"),
        }
        atomic_json(metadata_root / f"{stem}.json", entry)
        export_entries.append(entry)
        manifest_lines.append(
            "\t".join(
                [
                    str(rank), requested_id, model, str(feature_dim), "float32",
                    str(EXPECTED_ROWS), entry["feature_file"], representation,
                ]
            )
        )

    atomic_text(EXPORT_ROOT / "manifest.tsv", "\n".join(manifest_lines) + "\n")
    export_manifest = {
        "format_version": "vtbench_imagenet_200shot_features_v1",
        "tokenizer_count": 70,
        "shots_per_class": SHOT,
        "class_count": 1_000,
        "rows_per_tokenizer": EXPECTED_ROWS,
        "feature_dtype": "float32",
        "support_seed": 0,
        "support_indices_sha256": support_hash,
        "row_order": "shared source_indices.npy; strictly increasing official ImageNet train indices",
        "labels_file": "labels.npy",
        "source_indices_file": "source_indices.npy",
        "seed_cache_metadata": str(seed_metadata_path.resolve()),
        "entries": export_entries,
    }
    atomic_json(EXPORT_ROOT / "export_manifest.json", export_manifest)
    readme = f"""# VTBench ImageNet-1K 200-shot frozen visual features

This package contains FP32 frozen-encoder representations for the canonical 70
tokenizers in `manifest.tsv` rank order. Each matrix contains exactly 200,000
rows (200 deterministic training examples per ImageNet-1K class).

All matrices share the same row order:

- `source_indices.npy`: official ImageNet training-set source indices, sorted;
- `labels.npy`: ImageNet labels for those rows;
- `features/001_*.npy` through `features/070_*.npy`: FP32 feature matrices.

The support-index SHA-256 is `{support_hash}`. Load with NumPy, for example:

```python
import numpy as np
features = np.load("features/001_clip_openai__l14.npy", mmap_mode="r")
labels = np.load("labels.npy", mmap_mode="r")
source_indices = np.load("source_indices.npy", mmap_mode="r")
```

No validation features, classifier weights, or ImageNet pixels are included.
"""
    atomic_text(EXPORT_ROOT / "README.md", readme)
    print(f">> export complete: {EXPORT_ROOT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
