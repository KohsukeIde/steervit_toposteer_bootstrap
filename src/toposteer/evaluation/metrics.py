from __future__ import annotations

import math
from collections import defaultdict

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
