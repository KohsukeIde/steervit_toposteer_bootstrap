from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F


def record_image_key(record: dict[str, Any]) -> str:
    meta = record.get("meta", {}) or {}
    for key in ("image_id", "vg_image_id", "coco_image_id"):
        value = meta.get(key)
        if value is not None:
            return f"{key}:{value}"
    for nested_key in ("left_meta", "right_meta"):
        nested = meta.get(nested_key, {}) or {}
        for key in ("image_id", "vg_image_id", "coco_image_id"):
            value = nested.get(key)
            if value is not None:
                return f"{key}:{value}"
    image_path = record.get("image_path")
    if image_path:
        return f"path:{image_path}"
    return f"id:{record.get('id', 'unknown')}"


def masked_token_pool(token_features: torch.Tensor, patch_masks: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    token_features: (N, D)
    patch_masks: (R, N) soft occupancies
    returns: (R, D)
    """
    if token_features.ndim != 2:
        raise ValueError(f"Expected token_features to have shape (N, D), got {tuple(token_features.shape)}")
    if patch_masks.ndim != 2:
        raise ValueError(f"Expected patch_masks to have shape (R, N), got {tuple(patch_masks.shape)}")
    if token_features.size(0) != patch_masks.size(1):
        raise ValueError(
            f"Token/mask mismatch: token_features has {token_features.size(0)} tokens, "
            f"patch_masks expects {patch_masks.size(1)}"
        )

    weights = patch_masks.float()
    denom = weights.sum(dim=1, keepdim=True).clamp_min(eps)
    weights = weights / denom
    return weights @ token_features.float()


def cosine_similarity_matrix(features: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    if features.ndim != 2:
        raise ValueError(f"Expected features to have shape (N, D), got {tuple(features.shape)}")
    if features.size(0) == 0:
        return features.new_zeros((0, 0))
    feats = F.normalize(features.float(), dim=-1, eps=eps)
    return feats @ feats.t()


def topk_neighbor_indices(sim: torch.Tensor, k: int, exclude_self: bool = True) -> torch.Tensor:
    if sim.ndim != 2 or sim.size(0) != sim.size(1):
        raise ValueError(f"Expected square similarity matrix, got {tuple(sim.shape)}")
    n = int(sim.size(0))
    if n == 0:
        return torch.empty((0, 0), dtype=torch.long, device=sim.device)

    max_neighbors = n - 1 if exclude_self else n
    k_eff = min(int(k), max_neighbors)
    if k_eff <= 0:
        return torch.empty((n, 0), dtype=torch.long, device=sim.device)

    work = sim.clone()
    if exclude_self:
        work.fill_diagonal_(-float("inf"))
    return torch.topk(work, k=k_eff, dim=1, largest=True, sorted=True).indices


def neighbor_jaccard_per_node(sim_a: torch.Tensor, sim_b: torch.Tensor, k: int) -> np.ndarray:
    idx_a = topk_neighbor_indices(sim_a, k).detach().cpu().numpy()
    idx_b = topk_neighbor_indices(sim_b, k).detach().cpu().numpy()
    if idx_a.shape[1] == 0:
        return np.full((idx_a.shape[0],), np.nan, dtype=np.float64)

    out = np.zeros((idx_a.shape[0],), dtype=np.float64)
    for row in range(idx_a.shape[0]):
        set_a = set(int(x) for x in idx_a[row].tolist())
        set_b = set(int(x) for x in idx_b[row].tolist())
        union = set_a | set_b
        if not union:
            out[row] = np.nan
            continue
        out[row] = len(set_a & set_b) / len(union)
    return out


def neighbor_flip_rate_per_node(sim_a: torch.Tensor, sim_b: torch.Tensor, k: int) -> np.ndarray:
    jacc = neighbor_jaccard_per_node(sim_a, sim_b, k)
    return 1.0 - jacc


def spearman_rank_correlation_per_node(sim_a: torch.Tensor, sim_b: torch.Tensor) -> np.ndarray:
    if sim_a.shape != sim_b.shape:
        raise ValueError("Similarity matrices must have the same shape.")
    if sim_a.ndim != 2 or sim_a.size(0) != sim_a.size(1):
        raise ValueError(f"Expected square similarity matrices, got {tuple(sim_a.shape)}")

    sa = sim_a.detach().cpu().numpy().astype(np.float64, copy=True)
    sb = sim_b.detach().cpu().numpy().astype(np.float64, copy=True)
    n = sa.shape[0]
    if n < 3:
        return np.full((n,), np.nan, dtype=np.float64)

    diag = np.eye(n, dtype=bool)
    sa[diag] = -np.inf
    sb[diag] = -np.inf

    order_a = np.argsort(-sa, axis=1)[:, :-1]
    order_b = np.argsort(-sb, axis=1)[:, :-1]
    m = order_a.shape[1]
    if m < 2:
        return np.full((n,), np.nan, dtype=np.float64)

    rows = np.arange(n)[:, None]
    ranks_a = np.full((n, n), -1, dtype=np.int64)
    ranks_b = np.full((n, n), -1, dtype=np.int64)
    ranks_a[rows, order_a] = np.arange(m)[None, :]
    ranks_b[rows, order_b] = np.arange(m)[None, :]

    denom = float(m * (m * m - 1))
    out = np.zeros((n,), dtype=np.float64)
    for i in range(n):
        valid = ranks_a[i] >= 0
        d = ranks_a[i, valid] - ranks_b[i, valid]
        out[i] = 1.0 - (6.0 * float((d.astype(np.float64) ** 2).sum())) / denom
    return out


def node_rank_to_target(sim: torch.Tensor, source_idx: int, target_idx: int) -> int | None:
    if sim.ndim != 2 or sim.size(0) != sim.size(1):
        raise ValueError(f"Expected square similarity matrix, got {tuple(sim.shape)}")
    n = int(sim.size(0))
    if source_idx < 0 or source_idx >= n or target_idx < 0 or target_idx >= n or source_idx == target_idx:
        return None
    work = sim[source_idx].detach().cpu().numpy().astype(np.float64, copy=True)
    work[source_idx] = -np.inf
    order = np.argsort(-work)
    hits = np.where(order == int(target_idx))[0]
    if hits.size == 0:
        return None
    return int(hits[0]) + 1


def subset_mean(values: np.ndarray, subset_mask: np.ndarray | Iterable[bool] | None) -> float | None:
    values = np.asarray(values, dtype=np.float64)
    if subset_mask is None:
        finite = np.isfinite(values)
        if not finite.any():
            return None
        return float(values[finite].mean())
    mask = np.asarray(list(subset_mask) if not isinstance(subset_mask, np.ndarray) else subset_mask, dtype=bool)
    if mask.shape[0] != values.shape[0]:
        raise ValueError(f"Subset mask length {mask.shape[0]} does not match values length {values.shape[0]}")
    finite = np.isfinite(values) & mask
    if not finite.any():
        return None
    return float(values[finite].mean())


def bank_overlap_scores(bank_masks: torch.Tensor, query_mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    if bank_masks.ndim != 2:
        raise ValueError(f"Expected bank_masks shape (R, N), got {tuple(bank_masks.shape)}")
    if query_mask.ndim != 1:
        raise ValueError(f"Expected query_mask shape (N,), got {tuple(query_mask.shape)}")
    numer = (bank_masks.float() * query_mask.float().unsqueeze(0)).sum(dim=1)
    denom = bank_masks.float().sum(dim=1).clamp_min(eps)
    return numer / denom


def entity_local_far_masks(
    bank_masks: torch.Tensor,
    patch_mask_pos: torch.Tensor,
    patch_mask_neg: torch.Tensor,
    local_threshold: float = 0.05,
    far_threshold: float = 0.0,
    force_local_indices: Iterable[int] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    overlap_pos = bank_overlap_scores(bank_masks, patch_mask_pos)
    overlap_neg = bank_overlap_scores(bank_masks, patch_mask_neg)
    overlap = torch.maximum(overlap_pos, overlap_neg).detach().cpu().numpy().astype(np.float64)

    local = overlap >= float(local_threshold)
    far = overlap <= float(far_threshold)
    if force_local_indices is not None:
        for idx in force_local_indices:
            if 0 <= int(idx) < local.shape[0]:
                local[int(idx)] = True
                far[int(idx)] = False
    return local, far, overlap


def patch_local_far_masks(
    patch_mask_pos: torch.Tensor,
    patch_mask_neg: torch.Tensor,
    local_threshold: float = 0.25,
    far_threshold: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    overlap = torch.maximum(patch_mask_pos.float(), patch_mask_neg.float()).detach().cpu().numpy().astype(np.float64)
    local = overlap >= float(local_threshold)
    far = overlap <= float(far_threshold)
    return local, far, overlap


def compute_rearrangement_metrics(
    sim_a: torch.Tensor,
    sim_b: torch.Tensor,
    ks: Iterable[int],
    prefix: str,
    local_mask: np.ndarray | None = None,
    far_mask: np.ndarray | None = None,
    include_rank_correlation: bool = False,
) -> dict[str, float | int]:
    metrics: dict[str, float | int] = {f"{prefix}_num_nodes": int(sim_a.size(0))}
    if local_mask is not None:
        metrics[f"{prefix}_num_local_nodes"] = int(np.asarray(local_mask, dtype=bool).sum())
    if far_mask is not None:
        metrics[f"{prefix}_num_far_nodes"] = int(np.asarray(far_mask, dtype=bool).sum())

    for k in ks:
        jacc = neighbor_jaccard_per_node(sim_a, sim_b, int(k))
        flip = 1.0 - jacc
        metrics[f"{prefix}_neighbor_jaccard_k{k}_all"] = subset_mean(jacc, None)
        metrics[f"{prefix}_neighbor_flip_rate_k{k}_all"] = subset_mean(flip, None)

        local_mean = subset_mean(flip, local_mask)
        far_mean = subset_mean(flip, far_mask)
        metrics[f"{prefix}_neighbor_flip_rate_k{k}_local"] = local_mean
        metrics[f"{prefix}_neighbor_flip_rate_k{k}_far"] = far_mean

        local_jacc = subset_mean(jacc, local_mask)
        far_jacc = subset_mean(jacc, far_mask)
        metrics[f"{prefix}_neighbor_jaccard_k{k}_local"] = local_jacc
        metrics[f"{prefix}_neighbor_jaccard_k{k}_far"] = far_jacc

        if local_mean is not None and far_mean is not None:
            metrics[f"{prefix}_localized_edit_diff_k{k}"] = float(local_mean - far_mean)
            if abs(far_mean) > 1e-12:
                metrics[f"{prefix}_localized_edit_ratio_k{k}"] = float(local_mean / far_mean)

    if include_rank_correlation:
        rho = spearman_rank_correlation_per_node(sim_a, sim_b)
        metrics[f"{prefix}_rank_spearman_all"] = subset_mean(rho, None)
        metrics[f"{prefix}_rank_spearman_local"] = subset_mean(rho, local_mask)
        metrics[f"{prefix}_rank_spearman_far"] = subset_mean(rho, far_mask)
    return metrics


def pca_project_2d(features: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(features, torch.Tensor):
        x = features.detach().cpu().numpy().astype(np.float64, copy=False)
    else:
        x = np.asarray(features, dtype=np.float64)

    if x.ndim != 2:
        raise ValueError(f"Expected 2D feature array, got shape {x.shape}")
    if x.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float64)
    if x.shape[1] == 1:
        return np.concatenate([x, np.zeros((x.shape[0], 1), dtype=np.float64)], axis=1)

    x = x - x.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(x, full_matrices=False)
    basis = vt[:2].T
    coords = x @ basis
    if coords.shape[1] == 1:
        coords = np.concatenate([coords, np.zeros((coords.shape[0], 1), dtype=np.float64)], axis=1)
    return coords[:, :2]
