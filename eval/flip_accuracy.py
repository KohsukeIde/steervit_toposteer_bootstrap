#!/usr/bin/env python
from __future__ import annotations

import argparse

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from toposteer.datasets import UnifiedRefExpDataset, collate_refexp
from toposteer.evaluation import compute_flip_accuracy, heatmap_mass_gap
from toposteer.losses.counterfactual import pairwise_mask_scores
from toposteer.models import SteerViTTrainable
from toposteer.utils import patchify_soft_mask


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute flip accuracy on a paired manifest.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--gate-factor", type=float, default=1.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = SteerViTTrainable(args.checkpoint, device=args.device, trainable_modules=())
    model.eval()
    model.set_gate_factor(args.gate_factor)

    dataset = UnifiedRefExpDataset(args.manifest, image_transform=model.get_transforms())
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_refexp)

    pos_scores_all = []
    neg_scores_all = []

    for batch in tqdm(loader, desc="Evaluating flip accuracy"):
        if batch["masks_neg"] is None:
            raise ValueError("flip_accuracy.py requires a manifest with mask_neg_path entries.")
        images = batch["images"].to(model.device_name)
        masks_pos = batch["masks_pos"].to(model.device_name)
        masks_neg = batch["masks_neg"].to(model.device_name)

        out = model.forward_outputs(images, texts=batch["prompts_pos"])
        patch_mask_pos = patchify_soft_mask(masks_pos, model.patch_grid, normalize=False)
        patch_mask_neg = patchify_soft_mask(masks_neg, model.patch_grid, normalize=False)
        pos_scores, neg_scores = pairwise_mask_scores(out.probs, patch_mask_pos, patch_mask_neg, mode="mean")

        pos_scores_all.append(pos_scores.detach().cpu())
        neg_scores_all.append(neg_scores.detach().cpu())

    pos_scores = torch.cat(pos_scores_all)
    neg_scores = torch.cat(neg_scores_all)

    print(f"flip_accuracy={compute_flip_accuracy(pos_scores, neg_scores):.4f}")
    print(f"mean_gap={heatmap_mass_gap(pos_scores, neg_scores):.4f}")


if __name__ == "__main__":
    main()
