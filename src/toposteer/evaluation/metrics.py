from __future__ import annotations

import math
from collections import defaultdict
from typing import Iterable

import torch


def compute_flip_accuracy(pos_scores: torch.Tensor, neg_scores: torch.Tensor) -> float:
    return float((pos_scores > neg_scores).float().mean().item())


def heatmap_mass_gap(pos_scores: torch.Tensor, neg_scores: torch.Tensor) -> float:
    return float((pos_scores - neg_scores).mean().item())


def compute_binary_iou(pred_mask: torch.Tensor, gt_mask: torch.Tensor, eps: float = 1e-6) -> float:
    pred = (pred_mask > 0).float()
    gt = (gt_mask > 0).float()
    inter = (pred * gt).sum().item()
    union = ((pred + gt) > 0).float().sum().item()
    return float(inter / max(union, eps))


def aggregate_scalar_metrics(rows: list[dict]) -> dict[str, float]:
    bucket = defaultdict(list)
    for row in rows:
        for key, value in row.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                if math.isfinite(float(value)):
                    bucket[key].append(float(value))
    return {k: sum(v) / len(v) for k, v in bucket.items() if v}


def normalized_trapz_area(xs: Iterable[float], ys: Iterable[float]) -> float | None:
    """
    Normalized trapezoidal area under a 1D curve.

    Returns None when fewer than 2 distinct x-values are available.
    The output is normalized by the x-range so a constant curve y=c has area c.
    """
    pairs = sorted((float(x), float(y)) for x, y in zip(xs, ys))
    if len(pairs) < 2:
        return None

    deduped: list[tuple[float, float]] = []
    for x, y in pairs:
        if deduped and abs(deduped[-1][0] - x) < 1e-12:
            deduped[-1] = (x, y)
        else:
            deduped.append((x, y))

    if len(deduped) < 2:
        return None

    x0, _ = deduped[0]
    x1, _ = deduped[-1]
    if abs(x1 - x0) < 1e-12:
        return None

    area = 0.0
    for (xa, ya), (xb, yb) in zip(deduped[:-1], deduped[1:]):
        area += 0.5 * (ya + yb) * (xb - xa)
    return float(area / (x1 - x0))
