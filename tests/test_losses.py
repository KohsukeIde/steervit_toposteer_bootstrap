import torch

from toposteer.losses.background import background_cosine_drift
from toposteer.losses.counterfactual import counterfactual_margin_loss
from toposteer.losses.refseg import soft_patch_ce


def test_soft_patch_ce_finite():
    logits = torch.tensor([[1.0, 2.0, 0.0]])
    target = torch.tensor([[0.0, 1.0, 0.0]])
    loss = soft_patch_ce(logits, target)
    assert torch.isfinite(loss)


def test_counterfactual_margin_loss_small_when_order_correct():
    probs = torch.tensor([[0.8, 0.2]])
    pos_mask = torch.tensor([[1.0, 0.0]])
    neg_mask = torch.tensor([[0.0, 1.0]])
    loss = counterfactual_margin_loss(probs, pos_mask, neg_mask, margin=0.1)
    assert loss.item() == 0.0


def test_background_cosine_drift_zero_when_equal():
    a = torch.randn(2, 4, 8)
    b = a.clone()
    protect = torch.ones(2, 4)
    loss = background_cosine_drift(a, b, protect)
    assert abs(loss.item()) < 1e-6


def test_counterfactual_margin_loss_empty_batch_returns_zero():
    probs = torch.zeros(0, 2)
    pos_mask = torch.zeros(0, 2)
    neg_mask = torch.zeros(0, 2)
    loss = counterfactual_margin_loss(probs, pos_mask, neg_mask, margin=0.1)
    assert loss.item() == 0.0
