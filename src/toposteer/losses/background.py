from __future__ import annotations

import torch
import torch.nn.functional as F


def background_cosine_drift(
    prompt_dense: torch.Tensor,
    base_dense: torch.Tensor,
    protect_mask_patch: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    prompt_dense, base_dense: (B, N, D)
    protect_mask_patch: (B, N), where 1 means "keep stable"
    """
    cos = F.cosine_similarity(prompt_dense.float(), base_dense.float(), dim=-1)
    drift = 1.0 - cos
    protect_mask_patch = protect_mask_patch.float()
    loss = (drift * protect_mask_patch).sum(dim=1) / protect_mask_patch.sum(dim=1).clamp_min(eps)
    return loss.mean()
