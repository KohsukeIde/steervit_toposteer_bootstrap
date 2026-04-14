import torch

from toposteer.datasets.unified_refexp import collate_refexp


def test_collate_refexp_mixed_batch_preserves_neg_validity():
    batch = [
        {
            "id": "a",
            "source": "custom",
            "family": "attr",
            "image_path": "/tmp/a.jpg",
            "image": torch.zeros(3, 8, 8),
            "mask_pos": torch.ones(1, 8, 8),
            "mask_neg": torch.zeros(1, 8, 8),
            "prompt_pos": "red car",
            "prompt_neg": "blue car",
            "meta": {},
            "has_neg": True,
            "has_prompt_neg": True,
        },
        {
            "id": "b",
            "source": "custom",
            "family": "plain",
            "image_path": "/tmp/b.jpg",
            "image": torch.zeros(3, 8, 8),
            "mask_pos": torch.ones(1, 8, 8),
            "mask_neg": None,
            "prompt_pos": "chair",
            "prompt_neg": None,
            "meta": {},
            "has_neg": False,
            "has_prompt_neg": False,
        },
    ]

    out = collate_refexp(batch)
    assert out["masks_neg"].shape == (2, 1, 8, 8)
    assert out["valid_neg_mask"].tolist() == [True, False]
    assert out["valid_prompt_neg_mask"].tolist() == [True, False]
    assert torch.allclose(out["masks_neg"][1], torch.zeros(1, 8, 8))
