#!/usr/bin/env python
"""Normalize the local two-row main table and optional E3 supplements."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

try:
    from .data import WORKSPACE
except ImportError:
    from data import WORKSPACE


FIELDS = (
    "name", "family", "MLLM_Avg", "qwen3", "qwen2_5", "smollm2",
    "probe_epoch1", "retrieval-ImageNet", "CKA", "pretrain_loss", "A_score",
)
NAME_ALIASES = {
    "toklip_s_semantic_256": "toklip_s_256",
    "toklip_l_semantic_384": "toklip_l_384",
    "unitok": "unitok_attn",
    "vilau_7b_256_semantic_penultimate": "vilau_256",
    "ijepa": "I-JEPA",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--main-table", type=Path,
        default=WORKSPACE / "result" / "VisualTokenizer表现 - 主表 (1).csv",
    )
    parser.add_argument(
        "--epoch-table", type=Path,
        default=WORKSPACE / "outputs" / "vae_linear_probing_dinov2_single_paperlr" /
        "accuracy_by_epoch.md",
    )
    parser.add_argument(
        "--supplement", type=Path, nargs="*", default=(),
        help=(
            "Canonical one-header CSV(s) with name plus any E3 columns. "
            "Non-empty supplement cells override the main table."
        ),
    )
    parser.add_argument(
        "--supplement-only", action="store_true",
        help="Build only from supplement rows; do not retain cells from the older main table.",
    )
    parser.add_argument(
        "--output", type=Path,
        default=WORKSPACE / "lar" / "configs" / "e3_targets.csv",
    )
    parser.add_argument(
        "--require-complete", action="store_true",
        help="Fail if any model lacks any requested target or baseline value.",
    )
    return parser.parse_args()


def clean_number(value: str) -> str:
    stripped = value.strip()
    if not stripped or stripped in {"-", "—", "NA", "N/A", "nan"}:
        return ""
    try:
        return format(float(stripped.rstrip("%")), ".12g")
    except ValueError:
        return ""


def infer_family(name: str) -> str:
    lowered = name.lower()
    if lowered == "i-jepa" or lowered.startswith("ijepa"):
        return "ijepa"
    if lowered in {"clip_openai__l14", "clip_meta__l14", "metaclip_b16_2pt5b"}:
        return "clip"
    for prefix, family in (
        ("siglip2", "siglip2"), ("mc1", "mc1"), ("mc2", "mc2"),
        ("pe_core", "pe_core"), ("pe_lang", "pe_lang"),
        ("dinov2", "dinov2"), ("dinov3", "dinov3"), ("dinov1", "dino"),
        ("dino_", "dino"),
        ("webssl_mae", "webssl_mae"), ("webssl_dino", "dino"),
        ("pixio", "pixio"), ("eupe", "eupe"), ("raev2", "raev2"),
        ("toklip", "toklip"), ("unitok", "unitok"), ("uniar", "uniar"),
        ("vilau", "vilau"),
    ):
        if lowered.startswith(prefix):
            return family
    return "other"


def read_main_table(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.reader(handle))
    if len(rows) < 3 or len(rows[0]) < 18 or len(rows[1]) < 18:
        raise RuntimeError(f"Expected the two-header 18-column main table: {path}")
    expected = {
        0: "Tokenizer", 2: "ImageNet", 3: "ImageNet",
        12: "qwen2.5 1.5B", 15: "qwen3 1.7B", 16: "qwen2.5 1.5B",
    }
    for index, label in expected.items():
        actual = rows[0][index] if index == 0 else rows[1][index]
        if actual.strip() != label:
            raise RuntimeError(f"Unexpected main-table header at column {index}: {actual!r}")
    output = {}
    for source in rows[2:]:
        if not source or not source[0].strip():
            continue
        name = source[0].strip()
        output[name] = {
            "name": name,
            "family": infer_family(name),
            "MLLM_Avg": clean_number(source[16]),
            "qwen3": clean_number(source[15]),
            "qwen2_5": clean_number(source[16]),
            "smollm2": "",
            "probe_epoch1": "",
            "retrieval-ImageNet": clean_number(source[3]),
            "CKA": "",
            "pretrain_loss": clean_number(source[12]),
            "A_score": "",
        }
    return output


def read_epoch_markdown(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    output = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("|") or "tokenizer" in line.lower() or re.match(r"^\|[-:| ]+\|$", line):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) < 2:
            continue
        value = clean_number(cells[1])
        if value:
            output[NAME_ALIASES.get(cells[0], cells[0])] = value
    return output


def merge_supplement(rows: dict[str, dict[str, str]], path: Path) -> None:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        additions = list(csv.DictReader(handle))
    if not additions or "name" not in additions[0]:
        raise RuntimeError(f"Supplement must contain a name column: {path}")
    unknown_columns = sorted(set(additions[0]) - set(FIELDS))
    if unknown_columns:
        print(f"WARNING: ignoring supplement columns in {path}: {unknown_columns}")
    for addition in additions:
        name = addition.get("name", "").strip()
        if not name:
            continue
        row = rows.setdefault(name, {field: "" for field in FIELDS})
        row["name"] = name
        for field in FIELDS[1:]:
            value = addition.get(field, "").strip()
            if value:
                row[field] = value if field == "family" else clean_number(value)
        if not row.get("family"):
            row["family"] = infer_family(name)


def main() -> None:
    args = parse_args()
    if args.supplement_only and not args.supplement:
        raise ValueError("--supplement-only requires at least one --supplement CSV")
    rows = {} if args.supplement_only else read_main_table(args.main_table)
    if not args.supplement_only:
        for name, value in read_epoch_markdown(args.epoch_table).items():
            if name in rows:
                rows[name]["probe_epoch1"] = value
    for supplement in args.supplement:
        merge_supplement(rows, supplement)
    incomplete = {
        name: [field for field in FIELDS[2:] if not row.get(field, "").strip()]
        for name, row in rows.items()
    }
    incomplete = {name: fields for name, fields in incomplete.items() if fields}
    if incomplete and args.require_complete:
        preview = "\n".join(f"{name}: {fields}" for name, fields in list(incomplete.items())[:20])
        raise RuntimeError(f"Incomplete E3 target rows ({len(incomplete)}):\n{preview}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows[name] for name in sorted(rows))
    temporary.replace(args.output)
    coverage = {
        field: sum(bool(row.get(field, "").strip()) for row in rows.values())
        for field in FIELDS[1:]
    }
    print(f"wrote {len(rows)} rows to {args.output}; coverage={coverage}")
    if incomplete:
        print(f"WARNING: {len(incomplete)} rows have at least one missing E3 value")


if __name__ == "__main__":
    main()
