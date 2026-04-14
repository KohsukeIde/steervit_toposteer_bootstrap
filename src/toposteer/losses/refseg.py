from __future__ import annotations

import torch
import torch.nn.functional as F


def soft_patch_ce(logits: torch.Tensor, soft_targets: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    logits: (B, N)
    soft_targets: (B, N), usually normalized to sum to 1
    """
    soft_targets = soft_targets.float()
    soft_targets = soft_targets / soft_targets.sum(dim=1, keepdim=True).clamp_min(eps)
    log_probs = F.log_softmax(logits, dim=1)
    loss = -(soft_targets * log_probs).sum(dim=1).mean()
    return loss
