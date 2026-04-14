import torch

from toposteer.utils.mask_ops import patchify_soft_mask, mask_iou


def test_patchify_soft_mask_shape_and_normalization():
    mask = torch.zeros(1, 1, 8, 8)
    mask[:, :, :4, :4] = 1.0
    patch = patchify_soft_mask(mask, patch_grid=(2, 2), normalize=True)
    assert patch.shape == (1, 4)
    assert torch.allclose(patch.sum(dim=1), torch.ones(1), atol=1e-5)


def test_mask_iou():
    a = torch.zeros(1, 8, 8)
    b = torch.zeros(1, 8, 8)
    a[:, :4, :4] = 1
    b[:, :4, :4] = 1
    assert abs(mask_iou(a, b) - 1.0) < 1e-6
