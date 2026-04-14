#!/usr/bin/env python
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from toposteer.config import deep_update, load_yaml, parse_override_pairs
from toposteer.datasets import UnifiedRefExpDataset, collate_refexp
from toposteer.evaluation import aggregate_scalar_metrics, compute_flip_accuracy, heatmap_mass_gap
from toposteer.losses.counterfactual import pairwise_mask_scores
from toposteer.models import SteerViTTrainable
from toposteer.utils import ensure_dir, patchify_soft_mask, seed_everything, write_json, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gate sweep diagnostic over a unified manifest.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--override", nargs="*", default=[])
    return parser.parse_args()


def save_overlay(image_tensor: torch.Tensor, heatmap_2d: torch.Tensor, out_path: Path, alpha: float = 0.55) -> None:
    img = image_tensor.detach().cpu().permute(1, 2, 0).numpy()
    img = (img - img.min()) / max(img.max() - img.min(), 1e-6)
    hm = heatmap_2d.detach().cpu().numpy()
    hm = (hm - hm.min()) / max(hm.max() - hm.min(), 1e-6)

    plt.figure(figsize=(5, 5))
    plt.imshow(img)
    plt.imshow(hm, cmap="jet", alpha=alpha)
    plt.axis("off")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(pad=0)
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0)
    plt.close()


def summarize_rows(rows: list[dict]) -> dict:
    summary = aggregate_scalar_metrics(rows)
    if rows and any(r["has_neg"] for r in rows):
        pos_scores = torch.tensor([r["pos_score"] for r in rows if r["has_neg"]], dtype=torch.float32)
        neg_scores = torch.tensor([r["neg_score"] for r in rows if r["has_neg"]], dtype=torch.float32)
        summary["flip_accuracy"] = compute_flip_accuracy(pos_scores, neg_scores)
        summary["mean_gap"] = heatmap_mass_gap(pos_scores, neg_scores)
    return summary


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    cfg = deep_update(cfg, parse_override_pairs(args.override))
    if args.device is not None:
        cfg["device"] = args.device

    seed_everything(cfg.get("seed", 42))
    output_dir = ensure_dir(args.output_dir)
    overlay_dir = ensure_dir(output_dir / "overlays")

    model = SteerViTTrainable(
        checkpoint=args.checkpoint,
        device=cfg.get("device", "cuda"),
        trainable_modules=(),
    )
    transform = model.get_transforms()

    dataset = UnifiedRefExpDataset(
        manifest_path=args.manifest,
        image_transform=transform,
        limit=args.limit or cfg.get("limit"),
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg.get("batch_size", 4),
        shuffle=False,
        num_workers=cfg.get("num_workers", 2),
        collate_fn=collate_refexp,
    )

    all_summary = []
    per_factor_records: dict[float, list[dict]] = {}

    for gate_factor in cfg.get("gate_factors", [0.0, 1.0]):
        model.eval()
        model.set_gate_factor(float(gate_factor))

        rows = []
        overlay_budget = int(cfg.get("num_overlay_samples", 16))

        with torch.inference_mode():
            for batch in tqdm(loader, desc=f"Gate sweep @ {gate_factor}"):
                images = batch["images"].to(model.device_name)
                prompts_pos = batch["prompts_pos"]
                masks_pos = batch["masks_pos"].to(model.device_name)

                out = model.forward_outputs(images, texts=prompts_pos)
                patch_mask_pos = patchify_soft_mask(masks_pos, model.patch_grid, normalize=False)
                if batch["masks_neg"] is not None:
                    masks_neg = batch["masks_neg"].to(model.device_name)
                    patch_mask_neg = patchify_soft_mask(masks_neg, model.patch_grid, normalize=False)
                else:
                    patch_mask_neg = torch.zeros_like(patch_mask_pos)

                pos_scores, neg_scores = pairwise_mask_scores(
                    out.probs,
                    patch_mask_pos=patch_mask_pos,
                    patch_mask_neg=patch_mask_neg,
                    mode=cfg.get("score_mode", "mean"),
                )

                for i in range(images.size(0)):
                    row = {
                        "id": batch["ids"][i],
                        "family": batch["families"][i],
                        "gate_factor": float(gate_factor),
                        "pos_score": float(pos_scores[i].item()),
                        "neg_score": float(neg_scores[i].item()),
                        "gap": float((pos_scores[i] - neg_scores[i]).item()),
                        "has_neg": bool(batch["masks_neg"] is not None),
                    }
                    rows.append(row)

                    if cfg.get("save_overlays", True) and overlay_budget > 0:
                        heatmap = out.probs[i].view(model.patch_grid).unsqueeze(0).unsqueeze(0)
                        heatmap = torch.nn.functional.interpolate(
                            heatmap,
                            size=model.image_size,
                            mode="bilinear",
                            align_corners=False,
                        ).squeeze()
                        save_overlay(
                            images[i].detach().cpu(),
                            heatmap,
                            overlay_dir / f"{batch['ids'][i]}_gate{gate_factor:.2f}.png",
                            alpha=float(cfg.get("overlay_alpha", 0.55)),
                        )
                        overlay_budget -= 1

        summary = summarize_rows(rows)

        family_rows = defaultdict(list)
        for row in rows:
            family_rows[row.get("family", "unknown")].append(row)
        summary["by_family"] = {family: summarize_rows(fam_rows) for family, fam_rows in sorted(family_rows.items())}

        summary["gate_factor"] = float(gate_factor)
        all_summary.append(summary)
        per_factor_records[float(gate_factor)] = rows

    write_json(output_dir / "summary.json", {"summaries": all_summary})
    flat_rows = [row for rows in per_factor_records.values() for row in rows]
    write_jsonl(output_dir / "per_sample.jsonl", flat_rows)
    print(f"Wrote results to {output_dir}")


if __name__ == "__main__":
    main()
