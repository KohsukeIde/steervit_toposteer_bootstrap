import numpy as np
import torch

from toposteer.evaluation import (
    compute_rearrangement_metrics,
    cosine_similarity_matrix,
    masked_token_pool,
    neighbor_flip_rate_per_node,
    node_rank_to_target,
    patch_local_far_masks,
)


def test_masked_token_pool_weighted_average():
    tokens = torch.tensor([[1.0, 0.0], [0.0, 2.0], [1.0, 1.0]])
    masks = torch.tensor([[1.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    pooled = masked_token_pool(tokens, masks)
    assert pooled.shape == (2, 2)
    assert torch.allclose(pooled[0], torch.tensor([0.5, 1.0]))
    assert torch.allclose(pooled[1], torch.tensor([1.0, 1.0]))


def test_rearrangement_metrics_local_flip_exceeds_far_flip():
    # Four nodes arranged on a line. Prompt switch swaps the local neighborhood around node 0 and 1.
    feats_a = torch.tensor(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [0.0, 1.0],
            [0.0, 0.9],
        ]
    )
    feats_b = torch.tensor(
        [
            [0.0, 1.0],
            [0.1, 0.9],
            [1.0, 0.0],
            [0.9, 0.1],
        ]
    )
    sim_a = cosine_similarity_matrix(feats_a)
    sim_b = cosine_similarity_matrix(feats_b)
    local_mask = np.array([True, True, False, False])
    far_mask = ~local_mask

    metrics = compute_rearrangement_metrics(
        sim_a,
        sim_b,
        ks=[1, 2],
        prefix="entity_pos_neg",
        local_mask=local_mask,
        far_mask=far_mask,
        include_rank_correlation=True,
    )
    assert metrics["entity_pos_neg_neighbor_flip_rate_k1_local"] >= metrics["entity_pos_neg_neighbor_flip_rate_k1_far"]
    assert metrics["entity_pos_neg_rank_spearman_local"] <= metrics["entity_pos_neg_rank_spearman_far"]


def test_patch_local_far_masks_and_rank_lookup():
    pos = torch.tensor([1.0, 1.0, 0.0, 0.0])
    neg = torch.tensor([0.0, 0.0, 1.0, 1.0])
    local, far, overlap = patch_local_far_masks(pos, neg, local_threshold=0.5, far_threshold=0.0)
    assert local.tolist() == [True, True, True, True]
    assert far.tolist() == [False, False, False, False]
    assert overlap.shape[0] == 4

    sim = torch.tensor(
        [
            [1.0, 0.9, 0.2],
            [0.9, 1.0, 0.1],
            [0.2, 0.1, 1.0],
        ]
    )
    assert node_rank_to_target(sim, 0, 1) == 1
    flip = neighbor_flip_rate_per_node(sim, sim.clone(), k=1)
    assert np.allclose(flip, 0.0)
