from __future__ import annotations

from typing import Any


def compute_vtm_lcg_scores(
    losses: dict[str, float],
    *,
    epsilon: float = 1.0e-12,
) -> dict[str, Any]:
    required = {"L_mean", "L_visual", "L_visual_text"}
    missing = required - set(losses)
    if missing:
        raise KeyError(f"Missing required losses: {sorted(missing)}")
    l_mean = float(losses["L_mean"])
    l_visual = float(losses["L_visual"])
    l_visual_text = float(losses["L_visual_text"])
    if l_mean <= 0 or l_visual < 0 or l_visual_text < 0:
        raise ValueError(f"Losses must be non-negative and L_mean positive: {losses}")

    result: dict[str, Any] = {
        "VTMC": l_visual / max(l_mean, epsilon),
        "VTM": 1.0 - l_visual / max(l_mean, epsilon),
        "LCG": (l_visual - l_visual_text) / max(l_visual, epsilon),
    }
    if "L_visual_shuffled_text" in losses:
        shuffled = float(losses["L_visual_shuffled_text"])
        result["LCG_specific"] = (
            shuffled - l_visual_text
        ) / max(l_visual, epsilon)
        result["caption_specificity_pass"] = l_visual_text < shuffled
    if "L_visual_spatial_shuffle" in losses:
        spatial = float(losses["L_visual_spatial_shuffle"])
        result["spatial_loss_increase"] = (
            spatial - l_visual
        ) / max(l_visual, epsilon)
        result["spatial_structure_pass"] = l_visual < spatial
    if "L_no_visible" in losses:
        no_visible = float(losses["L_no_visible"])
        result["no_visible_to_mean_ratio"] = no_visible / max(l_mean, epsilon)
    result["VTM_positive"] = result["VTM"] > 0
    result["LCG_positive"] = result["LCG"] > 0
    return result

