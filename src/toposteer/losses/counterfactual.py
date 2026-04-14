from __future__ import annotations

import torch
import torch.nn.functional as F


def pairwise_mask_scores(
    patch_probs: torch.Tensor,
    patch_mask_pos: torch.Tensor,
    patch_mask_neg: torch.Tensor,
    mode: str = "mean",
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    patch_probs: (B, N), typically softmax over patch logits
    patch_mask_pos / patch_mask_neg: (B, N), binary or soft occupancies
    """
    patch_mask_pos = patch_mask_pos.float()
    patch_mask_neg = patch_mask_neg.float()

    if mode == "mean":
        pos_score = (patch_probs * patch_mask_pos).sum(dim=1) / patch_mask_pos.sum(dim=1).clamp_min(eps)
        neg_score = (patch_probs * patch_mask_neg).sum(dim=1) / patch_mask_neg.sum(dim=1).clamp_min(eps)
    elif mode == "sum":
        pos_score = (patch_probs * patch_mask_pos).sum(dim=1)
        neg_score = (patch_probs * patch_mask_neg).sum(dim=1)
    else:
        raise ValueError(f"Unsupported score mode: {mode}")
    return pos_score, neg_score


def counterfactual_margin_loss(
    patch_probs_pos_prompt: torch.Tensor,
    patch_mask_pos: torch.Tensor,
    patch_mask_neg: torch.Tensor,
    patch_probs_neg_prompt: torch.Tensor | None = None,
    margin: float = 0.10,
    mode: str = "mean",
) -> torch.Tensor:
    """
    Primary direction:
        prompt_pos should score its positive mask above its negative mask.
    Optional reverse direction:
        prompt_neg should score the negative mask above the positive mask.
    """
    pos_score, neg_score = pairwise_mask_scores(
        patch_probs_pos_prompt,
        patch_mask_pos=patch_mask_pos,
        patch_mask_neg=patch_mask_neg,
        mode=mode,
    )
    loss = F.relu(margin - pos_score + neg_score)

    if patch_probs_neg_prompt is not None:
        neg_prompt_pos_score, neg_prompt_neg_score = pairwise_mask_scores(
            patch_probs_neg_prompt,
            patch_mask_pos=patch_mask_neg,
            patch_mask_neg=patch_mask_pos,
            mode=mode,
        )
        loss = loss + F.relu(margin - neg_prompt_pos_score + neg_prompt_neg_score)

    return loss.mean()
