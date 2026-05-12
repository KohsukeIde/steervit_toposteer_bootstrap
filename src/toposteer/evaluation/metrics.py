from __future__ import annotations

import math
from collections import defaultdict
from typing import Iterable

import numpy as np
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


def bootstrap_mean_ci(
    values: Iterable[float],
    samples: int = 2000,
    seed: int = 0,
    ci: float = 0.95,
) -> dict[str, float | int | None]:
    arr = np.asarray(
        [float(v) for v in values if v is not None and math.isfinite(float(v))],
        dtype=np.float64,
    )
    if arr.size == 0:
        return {"mean": None, "ci_low": None, "ci_high": None, "num_values": 0, "num_bootstrap_samples": int(samples)}
    mean = float(arr.mean())
    if arr.size == 1 or samples <= 1:
        return {
            "mean": mean,
            "ci_low": mean,
            "ci_high": mean,
            "num_values": int(arr.size),
            "num_bootstrap_samples": int(samples),
        }

    rng = np.random.default_rng(int(seed))
    boots = np.empty(int(samples), dtype=np.float64)
    for i in range(int(samples)):
        idx = rng.integers(0, arr.size, size=arr.size)
        boots[i] = arr[idx].mean()
    alpha = max(0.0, min(1.0, (1.0 - float(ci)) / 2.0))
    return {
        "mean": mean,
        "ci_low": float(np.quantile(boots, alpha)),
        "ci_high": float(np.quantile(boots, 1.0 - alpha)),
        "num_values": int(arr.size),
        "num_bootstrap_samples": int(samples),
    }


def summarize_bootstrap_metrics(
    rows: list[dict],
    keys: Iterable[str],
    samples: int = 2000,
    seed: int = 0,
    ci: float = 0.95,
) -> dict[str, dict[str, float | int | None]]:
    out = {}
    for key in keys:
        values = []
        for row in rows:
            value = row.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)):
                values.append(float(value))
        out[str(key)] = bootstrap_mean_ci(values, samples=samples, seed=seed, ci=ci)
    return out
