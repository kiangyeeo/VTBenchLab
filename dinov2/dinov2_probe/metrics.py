from __future__ import annotations

from typing import Any, Dict, Iterable, List, Sequence, Tuple

import torch


def _target_contains(target: Any, pred: int) -> bool:
    if isinstance(target, torch.Tensor):
        if target.ndim == 0:
            return int(target.item()) == pred
        return pred in {int(x) for x in target.tolist()}
    if isinstance(target, (list, tuple, set)):
        return pred in {int(x) for x in target}
    return int(target) == pred


def topk_accuracy(logits: torch.Tensor, targets: Any, topk: Sequence[int] = (1, 5)) -> Dict[str, float]:
    maxk = min(max(topk), logits.shape[1])
    _, pred = logits.topk(maxk, dim=1)
    if isinstance(targets, torch.Tensor) and targets.ndim == 1:
        target = targets.to(pred.device).view(-1, 1)
        correct = pred.eq(target)
        return {f"top{k}": correct[:, : min(k, maxk)].any(dim=1).float().mean().item() * 100.0 for k in topk}

    target_list = targets.tolist() if isinstance(targets, torch.Tensor) else list(targets)
    out = {}
    for k in topk:
        kk = min(k, maxk)
        hits = 0
        for row, target in zip(pred[:, :kk].cpu().tolist(), target_list):
            hits += any(_target_contains(target, p) for p in row)
        out[f"top{k}"] = 100.0 * hits / max(len(target_list), 1)
    return out


def tensor_targets(targets: Any) -> torch.Tensor:
    if isinstance(targets, torch.Tensor):
        if targets.ndim != 1:
            raise ValueError("Training targets must be a 1D integer tensor")
        return targets.long()
    if not all(isinstance(t, (int, float)) for t in targets):
        raise ValueError("Training targets must be integer labels, not multi-label lists")
    return torch.tensor([int(t) for t in targets], dtype=torch.long)

