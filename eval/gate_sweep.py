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
from toposteer.evaluation import (
    aggregate_scalar_metrics,
    compute_flip_accuracy,
    heatmap_mass_gap,
    normalized_trapz_area,
)
from toposteer.losses.counterfactual import pairwise_mask_scores
from toposteer.models import SteerViTTrainable
from toposteer.utils import ensure_dir, patchify_soft_mask, seed_everything, write_json, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gate sweep diagnostic over a unified manifest.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--base-checkpoint",
        default=None,
        help="Required when --checkpoint is a TopoSteer training checkpoint with model_state_dict.",
    )
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


def bootstrap_flip_accuracy(pos_scores: np.ndarray, neg_scores: np.ndarray, num_samples: int, seed: int = 42) -> dict[str, float]:
    if len(pos_scores) == 0 or num_samples <= 0:
        return {}
    rng = np.random.default_rng(seed)
    idx = np.arange(len(pos_scores))
    values = []
    for _ in range(num_samples):
        sample_idx = rng.choice(idx, size=len(idx), replace=True)
        values.append(float(np.mean(pos_scores[sample_idx] > neg_scores[sample_idx])))
    values = np.asarray(values, dtype=np.float64)
    return {
        "flip_accuracy_ci_low": float(np.quantile(values, 0.025)),
        "flip_accuracy_ci_high": float(np.quantile(values, 0.975)),
    }


def compute_curve_aggregates(summaries: list[dict]) -> dict[str, object]:
    valid = [s for s in summaries if "flip_accuracy" in s]
    if len(valid) < 2:
        return {}

    gates = [float(s["gate_factor"]) for s in valid]
    accs = [float(s["flip_accuracy"]) for s in valid]
    gaps = [float(s["mean_gap"]) for s in valid if "mean_gap" in s]
    out: dict[str, object] = {
        "flip_accuracy_augc": normalized_trapz_area(gates, accs),
    }
    if len(gaps) == len(gates):
        out["mean_gap_augc"] = normalized_trapz_area(gates, gaps)

    families = sorted({family for s in valid for family in s.get("by_family", {}).keys()})
    by_family_augc: dict[str, dict[str, float | int | None]] = {}
    for family in families:
        fam_pairs = []
        for s in valid:
            fam = s.get("by_family", {}).get(family)
            if fam is None:
                continue
            fam_pairs.append((float(s["gate_factor"]), float(fam["flip_accuracy"]), int(fam["n"])))
        if len(fam_pairs) < 2:
            continue
        fam_pairs = sorted(fam_pairs, key=lambda x: x[0])
        by_family_augc[family] = {
            "flip_accuracy_augc": normalized_trapz_area([x[0] for x in fam_pairs], [x[1] for x in fam_pairs]),
            "min_n": min(x[2] for x in fam_pairs),
            "max_n": max(x[2] for x in fam_pairs),
        }
    if by_family_augc:
        out["by_family"] = by_family_augc
    return out


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
        base_checkpoint=args.base_checkpoint,
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
    bootstrap_samples = int(cfg.get("bootstrap_samples", 0))

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
                valid_neg_mask = batch["valid_neg_mask"].to(model.device_name)

                out = model.forward_outputs(images, texts=prompts_pos)
                patch_mask_pos = patchify_soft_mask(masks_pos, model.patch_grid, normalize=False)
                patch_mask_neg = patchify_soft_mask(batch["masks_neg"].to(model.device_name), model.patch_grid, normalize=False)

                pos_scores, neg_scores = pairwise_mask_scores(
                    out.probs,
                    patch_mask_pos=patch_mask_pos,
                    patch_mask_neg=patch_mask_neg,
                    mode=cfg.get("score_mode", "mean"),
                )

                for i in range(images.size(0)):
                    has_neg = bool(valid_neg_mask[i].item())
                    row = {
                        "id": batch["ids"][i],
                        "family": batch["families"][i],
                        "gate_factor": float(gate_factor),
                        "pos_score": float(pos_scores[i].item()),
                        "neg_score": float(neg_scores[i].item()) if has_neg else None,
                        "gap": float((pos_scores[i] - neg_scores[i]).item()) if has_neg else None,
                        "has_neg": has_neg,
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

        summary = aggregate_scalar_metrics(rows)
        valid_rows = [r for r in rows if r["has_neg"]]
        if valid_rows:
            pos_arr = np.asarray([r["pos_score"] for r in valid_rows], dtype=np.float32)
            neg_arr = np.asarray([r["neg_score"] for r in valid_rows], dtype=np.float32)
            pos_scores_t = torch.from_numpy(pos_arr)
            neg_scores_t = torch.from_numpy(neg_arr)
            summary["flip_accuracy"] = compute_flip_accuracy(pos_scores_t, neg_scores_t)
            summary["mean_gap"] = heatmap_mass_gap(pos_scores_t, neg_scores_t)
            summary.update(bootstrap_flip_accuracy(pos_arr, neg_arr, bootstrap_samples, seed=cfg.get("seed", 42)))

            by_family: dict[str, dict[str, float]] = {}
            grouped: dict[str, list[dict]] = defaultdict(list)
            for row in valid_rows:
                grouped[row["family"]].append(row)
            for family, family_rows in grouped.items():
                fp = torch.tensor([r["pos_score"] for r in family_rows], dtype=torch.float32)
                fn = torch.tensor([r["neg_score"] for r in family_rows], dtype=torch.float32)
                by_family[family] = {
                    "n": len(family_rows),
                    "flip_accuracy": compute_flip_accuracy(fp, fn),
                    "mean_gap": heatmap_mass_gap(fp, fn),
                }
            summary["by_family"] = by_family

        summary["gate_factor"] = float(gate_factor)
        all_summary.append(summary)
        per_factor_records[float(gate_factor)] = rows

    curve_aggregates = compute_curve_aggregates(all_summary)
    write_json(output_dir / "summary.json", {"summaries": all_summary, "curve_aggregates": curve_aggregates})
    flat_rows = [row for rows in per_factor_records.values() for row in rows]
    write_jsonl(output_dir / "per_sample.jsonl", flat_rows)
    print(f"Wrote results to {output_dir}")


if __name__ == "__main__":
    main()
