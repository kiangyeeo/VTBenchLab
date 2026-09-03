"""Diagnostics for the full sweep, layered on top of the frozen blind prediction.

``summarize_full_sweep`` writes the pre-registered bet and must keep doing so
untouched. This module never rewrites it. It answers the questions the frozen
mean-rank number cannot:

* Which warmup runs diverged (the image made the LLM strictly worse than no image)?
* How much visual signal does each probe domain actually carry?
* Is the proxy explained by visual sequence length or feature width?
* Do the residuals against the MLLM decompose into per-family additive offsets,
  the way ImageNet linear probing does?
* How does it score under one-per-family / leave-one-family-out / family-stratified
  top-1 regret, rather than a whole-table Spearman?
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .summarize import _spearman
from .summarize_loss_proxy import _average_ranks_lower_is_better
from .utils import atomic_write_json, load_config, resolve_path

# A warmup run is treated as diverged when the correct image does not beat the
# no-image baseline on caption, the only domain with a large visual effect.
DIVERGENCE_EPSILON = 0.0


def _load_rows(config: dict, root: Path, allow_partial: bool) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    missing = []
    for name in config["tokenizers"]:
        path = root / "analysis" / "by_tokenizer" / f"{name}.json"
        if not path.is_file():
            missing.append(name)
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not payload.get("complete"):
            missing.append(name)
            continue
        rows[name] = {row["domain"]: row for row in payload["rows"]}
    if missing and not allow_partial:
        raise RuntimeError(
            f"{len(missing)} tokenizers incomplete; pass --allow-partial to analyze anyway: "
            f"{missing[:5]}..."
        )
    return rows


def _load_shapes(root: Path) -> dict[str, dict]:
    path = root / "analysis" / "token_shapes.json"
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {row["tokenizer"]: row for row in payload["rows"]}


def _load_input_dims(root: Path, seed: int) -> dict[str, int]:
    """Recover feature width from the surviving projector checkpoints."""
    dims: dict[str, int] = {}
    projector_root = root / "projectors"
    if not projector_root.is_dir():
        return dims
    for identity_source in sorted(projector_root.glob(f"*/seed_{seed}/projector_seen_*.pt")):
        name = identity_source.parents[1].name
        if name in dims:
            continue
        try:
            import torch

            payload = torch.load(identity_source, map_location="cpu")
            dims[name] = int(payload["identity"]["input_dim"])
        except Exception:  # noqa: BLE001 - a missing dim only drops one confound column
            continue
    return dims


def _group_offset_share(residuals: np.ndarray, groups: list[str]) -> float:
    """Fraction of residual variance explained by a per-group additive constant."""
    total = float(np.sum((residuals - residuals.mean()) ** 2))
    if total <= 0:
        return float("nan")
    centered = residuals.copy()
    for group in set(groups):
        mask = np.asarray([g == group for g in groups])
        centered[mask] -= residuals[mask].mean()
    return 1.0 - float(np.sum(centered**2)) / total


def _ols_r2(design: np.ndarray, target: np.ndarray) -> float:
    coefficients, *_ = np.linalg.lstsq(design, target, rcond=None)
    residuals = target - design @ coefficients
    total = float(np.sum((target - target.mean()) ** 2))
    if total <= 0:
        return float("nan")
    return 1.0 - float(np.sum(residuals**2)) / total


def _family_design(families: list[str]) -> np.ndarray:
    unique = sorted(set(families))
    columns = [np.asarray([1.0 if f == u else 0.0 for f in families]) for u in unique]
    return np.stack(columns, axis=1)


def family_offset_diagnosis(
    score: np.ndarray, truth: np.ndarray, families: list[str]
) -> dict:
    """Mirror of the ImageNet-probing diagnosis: does family identity absorb the residual?"""
    n = len(truth)
    intercept = np.ones((n, 1))
    score_column = score.reshape(-1, 1)
    family_columns = _family_design(families)
    score_only = _ols_r2(np.hstack([intercept, score_column]), truth)
    with_offsets = _ols_r2(np.hstack([family_columns, score_column]), truth)
    with_slopes = _ols_r2(
        np.hstack([family_columns, family_columns * score_column]), truth
    )
    coefficients, *_ = np.linalg.lstsq(np.hstack([intercept, score_column]), truth, rcond=None)
    residuals = truth - np.hstack([intercept, score_column]) @ coefficients
    return {
        "r2_score_only": score_only,
        "r2_plus_family_offsets": with_offsets,
        "r2_plus_family_slopes": with_slopes,
        "residual_variance_explained_by_family": _group_offset_share(residuals, families),
        "family_offsets": {
            family: float(residuals[[f == family for f in families]].mean())
            for family in sorted(set(families))
        },
        "mllm_range": float(truth.max() - truth.min()),
    }


def protocol_scores(
    score: np.ndarray, truth: np.ndarray, families: list[str], draws: int, seed: int
) -> dict:
    """Whole-table Spearman is optimistic; these three protocols are not."""
    rng = np.random.default_rng(seed)
    by_family: dict[str, list[int]] = {}
    for index, family in enumerate(families):
        by_family.setdefault(family, []).append(index)

    sampled = []
    for _ in range(draws):
        picked = [int(rng.choice(members)) for members in by_family.values()]
        sampled.append(_spearman(list(score[picked]), list(truth[picked])))
    sampled_array = np.asarray([v for v in sampled if np.isfinite(v)])

    within = {
        family: _spearman(list(score[members]), list(truth[members]))
        for family, members in by_family.items()
        if len(members) >= 3
    }

    regrets = []
    for members in by_family.values():
        best_by_score = members[int(np.argmax(score[members]))]
        regrets.append(float(truth[members].max() - truth[best_by_score]))

    lofo_spearman = []
    lofo_regret = []
    for family, members in by_family.items():
        others = [i for i in range(len(truth)) if families[i] != family]
        if len(others) < 3 or len(members) < 2:
            continue
        design = np.stack([np.ones(len(others)), score[others]], axis=1)
        coefficients, *_ = np.linalg.lstsq(design, truth[others], rcond=None)
        predicted = coefficients[0] + coefficients[1] * score[members]
        lofo_spearman.append(_spearman(list(predicted), list(truth[members])))
        lofo_regret.append(
            float(truth[members].max() - truth[members[int(np.argmax(predicted))]])
        )

    top_quartile = np.argsort(-truth)[: max(2, len(truth) // 4)]
    return {
        "whole_table_spearman": _spearman(list(score), list(truth)),
        "one_per_family_spearman_mean": float(sampled_array.mean()),
        "one_per_family_spearman_std": float(sampled_array.std()),
        "one_per_family_draws": int(sampled_array.size),
        "within_family_spearman": within,
        "family_stratified_top1_regret_mean": float(np.mean(regrets)),
        "family_stratified_top1_regret_max": float(np.max(regrets)),
        "leave_one_family_out_spearman_mean": (
            float(np.nanmean(lofo_spearman)) if lofo_spearman else float("nan")
        ),
        "leave_one_family_out_top1_regret_mean": (
            float(np.mean(lofo_regret)) if lofo_regret else float("nan")
        ),
        "top_quartile_spearman": _spearman(
            list(score[top_quartile]), list(truth[top_quartile])
        ),
    }


def build_table(config: dict, root: Path, allow_partial: bool, seed: int) -> list[dict]:
    rows = _load_rows(config, root, allow_partial)
    shapes = _load_shapes(root)
    dims = _load_input_dims(root, seed)
    domains = list(config["protocol"]["reliable_domains"])
    losses = {
        domain: {name: float(rows[name][domain]["real"]) for name in rows}
        for domain in domains
    }
    ranks = {domain: _average_ranks_lower_is_better(losses[domain]) for domain in domains}

    table = []
    for name in sorted(rows):
        spec = config["tokenizers"][name]
        caption = rows[name]["caption"]
        record = {
            "tokenizer": name,
            "registry_name": spec["registry_name"],
            "family": spec["family"],
            "calibration_model": name in set(config["protocol"]["calibration_tokenizers"]),
            "seq_len": shapes.get(name, {}).get("seq_len"),
            "feature_dim": shapes.get(name, {}).get("feature_dim", dims.get(name)),
            "frozen_mean_rank": float(np.mean([ranks[d][name] for d in domains])),
            "caption_real": float(caption["real"]),
            "caption_real_minus_zero": float(caption["real_minus_zero"]),
            "caption_real_minus_shuffled": float(caption["real_minus_shuffled"]),
            "diverged": bool(float(caption["real_minus_zero"]) >= DIVERGENCE_EPSILON),
        }
        for domain in domains:
            record[f"{domain}_real"] = float(rows[name][domain]["real"])
            record[f"{domain}_real_minus_shuffled"] = float(
                rows[name][domain]["real_minus_shuffled"]
            )
            record[f"{domain}_real_minus_zero"] = float(rows[name][domain]["real_minus_zero"])
        table.append(record)
    return table


def domain_signal_audit(table: list[dict], domains: list[str]) -> dict:
    """A domain whose correct-vs-deranged gap is ~0 cannot rank anything."""
    audit = {}
    for domain in domains:
        gaps = np.asarray(
            [abs(row[f"{domain}_real_minus_shuffled"]) for row in table if not row["diverged"]]
        )
        spread = np.asarray([row[f"{domain}_real"] for row in table if not row["diverged"]])
        audit[domain] = {
            "median_abs_real_minus_shuffled": float(np.median(gaps)) if gaps.size else float("nan"),
            "max_abs_real_minus_shuffled": float(gaps.max()) if gaps.size else float("nan"),
            "loss_spread_across_tokenizers": float(spread.max() - spread.min())
            if spread.size
            else float("nan"),
            "n": int(gaps.size),
        }
    return audit


def analyze(
    config: dict, ground_truth_csv: Path | None, allow_partial: bool, seed: int, draws: int
) -> Path:
    root = resolve_path(config, config["runtime"]["artifact_root"])
    table = build_table(config, root, allow_partial, seed)
    domains = list(config["protocol"]["reliable_domains"])

    diverged = [row["tokenizer"] for row in table if row["diverged"]]
    usable = [row for row in table if not row["diverged"]]
    report = {
        "schema_version": 1,
        "note": "Diagnostics only. The frozen blind prediction in summary/predictions.json is untouched.",
        "tokenizers_analyzed": len(table),
        "diverged_tokenizers": diverged,
        "divergence_rule": "caption real_minus_zero >= 0 (image no better than no image)",
        "domain_signal_audit": domain_signal_audit(table, domains),
        "rows": table,
    }

    if ground_truth_csv is not None and len(usable) >= 4:
        with ground_truth_csv.open("r", encoding="utf-8-sig", newline="") as handle:
            truth_rows = {row["name"]: row for row in csv.DictReader(handle)}
        kept, truth_values = [], []
        for row in usable:
            value = truth_rows.get(row["registry_name"], {}).get("qwen2_5", "")
            if value != "":
                kept.append(row)
                truth_values.append(float(value))
        truth = np.asarray(truth_values)
        families = [row["family"] for row in kept]
        # Higher score must mean better, so ranks and losses are negated.
        candidates = {
            "frozen_mean_rank": -np.asarray([row["frozen_mean_rank"] for row in kept]),
            "caption_real": -np.asarray([row["caption_real"] for row in kept]),
            "caption_real_minus_zero": -np.asarray(
                [row["caption_real_minus_zero"] for row in kept]
            ),
            "caption_real_minus_shuffled": -np.asarray(
                [row["caption_real_minus_shuffled"] for row in kept]
            ),
        }
        # Shapes can be missing for tokenizers whose cache was cleaned before the
        # poller started, so each confound is measured on its own covered subset.
        confounds = {}
        for key in ("seq_len", "feature_dim"):
            covered = [i for i, row in enumerate(kept) if row[key] is not None]
            confounds[key] = (
                None
                if len(covered) < 4
                else (covered, np.asarray([float(kept[i][key]) for i in covered]))
            )

        report["evaluation"] = {
            "n_scored": len(kept),
            "n_excluded_diverged": len(diverged),
            "family_coverage": {
                family: sum(1 for f in families if f == family) for family in sorted(set(families))
            },
            "scores": {
                name: {
                    "protocols": protocol_scores(score, truth, families, draws, seed),
                    "family_offset": family_offset_diagnosis(score, truth, families),
                    "confounds": {
                        key: (
                            None
                            if entry is None
                            else {
                                "n_covered": len(entry[0]),
                                "spearman_score_vs_confound": _spearman(
                                    list(score[entry[0]]), list(entry[1])
                                ),
                                "spearman_confound_vs_mllm": _spearman(
                                    list(entry[1]), list(truth[entry[0]])
                                ),
                            }
                        )
                        for key, entry in confounds.items()
                    },
                }
                for name, score in candidates.items()
            },
        }

    output_path = root / "analysis" / "diagnostics.json"
    atomic_write_json(output_path, report)
    print(f"Wrote {output_path}")
    print(f"  analyzed={len(table)} diverged={len(diverged)} {diverged if diverged else ''}")
    for domain, audit in report["domain_signal_audit"].items():
        print(
            f"  {domain}: median |real-shuffled|={audit['median_abs_real_minus_shuffled']:.4f} "
            f"loss spread={audit['loss_spread_across_tokenizers']:.4f}"
        )
    if "evaluation" in report:
        for name, block in report["evaluation"]["scores"].items():
            protocols = block["protocols"]
            offsets = block["family_offset"]
            print(
                f"  {name}: whole={protocols['whole_table_spearman']:.3f} "
                f"one-per-family={protocols['one_per_family_spearman_mean']:.3f} "
                f"regret={protocols['family_stratified_top1_regret_mean']:.2f} "
                f"family-residual={offsets['residual_variance_explained_by_family']:.3f}"
            )
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Full-sweep diagnostics")
    parser.add_argument("--config", required=True)
    parser.add_argument("--ground-truth-csv")
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--draws", type=int, default=2000)
    args = parser.parse_args()
    config, _ = load_config(args.config)
    truth_path = (
        None if args.ground_truth_csv is None else Path(args.ground_truth_csv).expanduser().resolve()
    )
    analyze(config, truth_path, args.allow_partial, args.seed, args.draws)


if __name__ == "__main__":
    main()
