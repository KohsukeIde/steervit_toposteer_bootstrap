#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from toposteer.config import deep_update, load_yaml, parse_override_pairs
from toposteer.evaluation import (
    cosine_similarity_matrix,
    entity_local_far_masks,
    index_region_bank,
    masked_token_pool,
    pca_project_2d,
    resolve_bank_entry,
    record_image_key,
    topk_neighbor_indices,
)
from toposteer.models import SteerViTTrainable
from toposteer.utils import apply_geometric_ops_to_mask, ensure_dir, patchify_soft_mask, read_jsonl, seed_everything, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a LensGraph-style prompt-conditioned entity-topology figure.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", required=True, help="Paired eval manifest, e.g. gold_eval.jsonl")
    parser.add_argument("--region-bank", required=True)
    parser.add_argument("--pair-bank", default=None, help="Optional tighter pair bank JSONL keyed by pair_id")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--base-checkpoint", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--pair-id", default=None)
    parser.add_argument("--pair-index", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--override", nargs="*", default=[])
    return parser.parse_args()


def load_patch_mask(mask_path: str | Path, image_transform, patch_grid: tuple[int, int]) -> tuple[torch.Tensor, torch.Tensor]:
    mask = apply_geometric_ops_to_mask(Image.open(mask_path).convert("L"), image_transform)
    patch_mask = patchify_soft_mask(mask, patch_grid, normalize=False).squeeze(0)
    return mask, patch_mask


def load_region_bank(path: str | Path) -> dict[str, dict[str, dict[str, Any]]]:
    rows = read_jsonl(path)
    return index_region_bank(rows)


def load_pair_bank(path: str | Path) -> dict[str, dict[str, Any]]:
    rows = read_jsonl(path)
    return {str(row.get("pair_id") or row.get("id")): row for row in rows}


def resolve_bank_entry(pair: dict[str, Any], image_bank: dict[str, dict[str, Any]], pair_bank: dict[str, dict[str, Any]] | None = None) -> tuple[str, dict[str, Any] | None]:
    pair_id = str(pair.get("id"))
    if pair_bank is not None and pair_id in pair_bank:
        return f"pair:{pair_id}", pair_bank[pair_id]
    image_key = record_image_key(pair)
    return f"image:{image_key}", image_bank.get(image_key)


def infer_pair(records: list[dict[str, Any]], pair_id: str | None, pair_index: int) -> dict[str, Any]:
    if pair_id is not None:
        matches = [r for r in records if r.get("id") == pair_id]
        if not matches:
            raise ValueError(f"Could not find pair_id={pair_id}")
        return matches[0]
    if pair_index < 0 or pair_index >= len(records):
        raise IndexError(f"pair_index {pair_index} is out of range for {len(records)} records")
    return records[pair_index]


def draw_knn_edges(ax, coords: np.ndarray, sim: torch.Tensor, k: int, color: str = "0.8", alpha: float = 0.35) -> None:
    knn = topk_neighbor_indices(sim, k=k).detach().cpu().numpy()
    for src in range(knn.shape[0]):
        for dst in knn[src].tolist():
            ax.plot(
                [coords[src, 0], coords[dst, 0]],
                [coords[src, 1], coords[dst, 1]],
                color=color,
                alpha=alpha,
                linewidth=0.8,
                zorder=1,
            )


def scatter_entities(ax, coords: np.ndarray, local_mask: np.ndarray, target_idx: int | None, distractor_idx: int | None, title: str) -> None:
    far_mask = ~local_mask
    ax.scatter(coords[far_mask, 0], coords[far_mask, 1], s=26, c="#bdbdbd", alpha=0.75, edgecolors="none", zorder=2)
    ax.scatter(coords[local_mask, 0], coords[local_mask, 1], s=38, c="#f4a259", alpha=0.95, edgecolors="black", linewidths=0.25, zorder=3)
    if target_idx is not None:
        ax.scatter(coords[target_idx, 0], coords[target_idx, 1], s=90, c="#d62728", edgecolors="black", linewidths=0.6, zorder=4, marker="o")
        ax.annotate("target", (coords[target_idx, 0], coords[target_idx, 1]), textcoords="offset points", xytext=(4, 4), fontsize=8)
    if distractor_idx is not None:
        ax.scatter(coords[distractor_idx, 0], coords[distractor_idx, 1], s=90, c="#1f77b4", edgecolors="black", linewidths=0.6, zorder=4, marker="s")
        ax.annotate("distractor", (coords[distractor_idx, 0], coords[distractor_idx, 1]), textcoords="offset points", xytext=(4, -10), fontsize=8)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal", adjustable="datalim")


def overlay_masks(ax, image_tensor: torch.Tensor, mask_pos: torch.Tensor, mask_neg: torch.Tensor, title: str) -> None:
    image = image_tensor.detach().cpu().permute(1, 2, 0).numpy()
    image = (image - image.min()) / max(image.max() - image.min(), 1e-6)
    ax.imshow(image)
    pos = mask_pos.squeeze(0).detach().cpu().numpy()
    neg = mask_neg.squeeze(0).detach().cpu().numpy()
    ax.imshow(np.ma.masked_where(pos <= 0, pos), cmap="Reds", alpha=0.35)
    ax.imshow(np.ma.masked_where(neg <= 0, neg), cmap="Blues", alpha=0.35)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    cfg = deep_update(cfg, parse_override_pairs(args.override))
    if args.device is not None:
        cfg["device"] = args.device

    seed_everything(cfg.get("seed", 42))
    output_dir = ensure_dir(args.output_dir)
    topology_cfg = cfg.get("topology", {})
    gate_factor = float(topology_cfg.get("gate_factor", 1.0))
    local_overlap_threshold = float(topology_cfg.get("local_overlap_threshold", 0.05))
    far_overlap_threshold = float(topology_cfg.get("far_overlap_threshold", 0.0))
    k = int(topology_cfg.get("figure_knn_k", 5))

    model = SteerViTTrainable.from_any_checkpoint(
        checkpoint=args.checkpoint,
        base_checkpoint=args.base_checkpoint,
        device=cfg.get("device", "cuda"),
        trainable_modules=(),
    )
    model.eval()
    model.set_gate_factor(gate_factor)
    image_transform = model.get_transforms()

    pair_records = read_jsonl(args.manifest)
    pair = infer_pair(pair_records, pair_id=args.pair_id, pair_index=args.pair_index)
    bank = load_region_bank(args.region_bank)
    pair_bank = load_pair_bank(args.pair_bank) if args.pair_bank else None
    image_key = record_image_key(pair)
    bank_key, bank_entry = resolve_bank_entry(pair, bank, pair_bank)
    if bank_entry is None:
        raise ValueError(f"No bank entry found for image_key={image_key}")

    bank_regions = bank_entry.get("regions", [])
    id_to_idx = {region.get("id"): idx for idx, region in enumerate(bank_regions)}
    target_idx = id_to_idx.get((pair.get("meta", {}) or {}).get("left_id"))
    distractor_idx = id_to_idx.get((pair.get("meta", {}) or {}).get("right_id"))

    image_path = bank_entry.get("image_path") or pair.get("image_path")
    image_tensor = image_transform(Image.open(image_path).convert("RGB"))
    out_off = model.forward_outputs(image_tensor.unsqueeze(0).to(model.device_name), texts=None)
    out_pos = model.forward_outputs(image_tensor.unsqueeze(0).to(model.device_name), texts=[pair["prompt_pos"]])
    out_neg = model.forward_outputs(image_tensor.unsqueeze(0).to(model.device_name), texts=[pair["prompt_neg"]])

    bank_patch_masks = []
    for region in bank_regions:
        _, patch_mask = load_patch_mask(region["mask_pos_path"], image_transform, model.patch_grid)
        bank_patch_masks.append(patch_mask)
    bank_patch_masks_t = torch.stack(bank_patch_masks, dim=0)

    entity_off = masked_token_pool(out_off.dense_tokens[0].detach().cpu(), bank_patch_masks_t)
    entity_pos = masked_token_pool(out_pos.dense_tokens[0].detach().cpu(), bank_patch_masks_t)
    entity_neg = masked_token_pool(out_neg.dense_tokens[0].detach().cpu(), bank_patch_masks_t)

    mask_pos, patch_pos = load_patch_mask(pair["mask_pos_path"], image_transform, model.patch_grid)
    mask_neg, patch_neg = load_patch_mask(pair["mask_neg_path"], image_transform, model.patch_grid)
    local_mask, _, overlap = entity_local_far_masks(
        bank_patch_masks_t,
        patch_pos,
        patch_neg,
        local_threshold=local_overlap_threshold,
        far_threshold=far_overlap_threshold,
        force_local_indices=[idx for idx in (target_idx, distractor_idx) if idx is not None],
    )

    joint = torch.cat([entity_off, entity_pos, entity_neg], dim=0)
    coords = pca_project_2d(joint)
    n = entity_off.size(0)
    coords_off = coords[:n]
    coords_pos = coords[n : 2 * n]
    coords_neg = coords[2 * n :]

    sim_off = cosine_similarity_matrix(entity_off)
    sim_pos = cosine_similarity_matrix(entity_pos)
    sim_neg = cosine_similarity_matrix(entity_neg)

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    overlay_masks(axes[0], image_tensor, mask_pos, mask_neg, title="image + target/distractor masks")
    draw_knn_edges(axes[1], coords_off, sim_off, k=k)
    scatter_entities(axes[1], coords_off, local_mask, target_idx, distractor_idx, title="prompt off")
    draw_knn_edges(axes[2], coords_pos, sim_pos, k=k)
    scatter_entities(axes[2], coords_pos, local_mask, target_idx, distractor_idx, title=f"prompt: {pair['prompt_pos']}")
    draw_knn_edges(axes[3], coords_neg, sim_neg, k=k)
    scatter_entities(axes[3], coords_neg, local_mask, target_idx, distractor_idx, title=f"prompt: {pair['prompt_neg']}")
    fig.suptitle(pair["id"], y=1.02)
    fig.tight_layout()
    fig.savefig(output_dir / "lensgraph_figure.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    node_table = []
    for idx, region in enumerate(bank_regions):
        node_table.append(
            {
                "node_index": idx,
                "region_id": region.get("id"),
                "prompt": region.get("prompt_pos"),
                "is_local": bool(local_mask[idx]),
                "overlap": float(overlap[idx]),
                "off_x": float(coords_off[idx, 0]),
                "off_y": float(coords_off[idx, 1]),
                "pos_x": float(coords_pos[idx, 0]),
                "pos_y": float(coords_pos[idx, 1]),
                "neg_x": float(coords_neg[idx, 0]),
                "neg_y": float(coords_neg[idx, 1]),
                "is_target": bool(target_idx is not None and idx == target_idx),
                "is_distractor": bool(distractor_idx is not None and idx == distractor_idx),
            }
        )

    write_json(
        output_dir / "figure_metadata.json",
        {
            "pair_id": pair["id"],
            "image_key": image_key,
            "image_path": image_path,
            "prompt_pos": pair["prompt_pos"],
            "prompt_neg": pair["prompt_neg"],
            "gate_factor": gate_factor,
            "figure_knn_k": k,
            "target_idx": target_idx,
            "distractor_idx": distractor_idx,
            "bank_key": bank_key,
            "num_regions": len(bank_regions),
            "bank_scope": bank_scope,
            "node_table": node_table,
            "checkpoint_load_info": getattr(model, "load_info", {}),
        },
    )
    print(f"Saved figure to {output_dir / 'lensgraph_figure.png'}")


if __name__ == "__main__":
    main()
