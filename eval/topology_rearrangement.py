#!/usr/bin/env python
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from toposteer.config import deep_update, load_yaml, parse_override_pairs
from toposteer.evaluation import (
    aggregate_scalar_metrics,
    compute_rearrangement_metrics,
    cosine_similarity_matrix,
    entity_local_far_masks,
    masked_token_pool,
    neighbor_flip_rate_per_node,
    node_rank_to_target,
    patch_local_far_masks,
    record_image_key,
    spearman_rank_correlation_per_node,
)
from toposteer.models import SteerViTTrainable
from toposteer.utils import (
    apply_geometric_ops_to_mask,
    ensure_dir,
    patchify_soft_mask,
    read_jsonl,
    seed_everything,
    write_json,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure prompt-conditioned topology rearrangement on a locked paired benchmark.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", required=True, help="Paired eval manifest, e.g. gold_eval.jsonl")
    parser.add_argument("--region-bank", required=True, help="Per-image region bank JSONL built from the positive PhraseCut manifest")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--base-checkpoint", default=None, help="Public SteerViT checkpoint used when --checkpoint is a TopoSteer wrapper checkpoint")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--pair-id", default=None, help="Optional single pair id to evaluate")
    parser.add_argument("--mode", choices=["entity", "patch", "both"], default="both")
    parser.add_argument("--override", nargs="*", default=[])
    return parser.parse_args()


def load_region_bank(path: str | Path, max_regions_per_image: int | None = None) -> dict[str, dict[str, Any]]:
    rows = read_jsonl(path)
    bank = {}
    for row in rows:
        regions = row.get("regions", [])
        if max_regions_per_image is not None and max_regions_per_image > 0:
            regions = regions[: int(max_regions_per_image)]
        bank[str(row["image_key"])] = {**row, "regions": regions}
    return bank


def load_image_tensor(image_path: str | Path, image_transform) -> torch.Tensor:
    return image_transform(Image.open(image_path).convert("RGB"))


def load_patch_mask(mask_path: str | Path, image_transform, patch_grid: tuple[int, int]) -> tuple[torch.Tensor, torch.Tensor]:
    mask = apply_geometric_ops_to_mask(Image.open(mask_path).convert("L"), image_transform)
    patch_mask = patchify_soft_mask(mask, patch_grid, normalize=False).squeeze(0)
    return mask, patch_mask


def infer_target_distractor_indices(pair_record: dict[str, Any], bank_regions: list[dict[str, Any]]) -> tuple[int | None, int | None]:
    meta = pair_record.get("meta", {}) or {}
    left_id = meta.get("left_id")
    right_id = meta.get("right_id")
    if left_id is None and right_id is None:
        return None, None

    id_to_idx = {region.get("id"): idx for idx, region in enumerate(bank_regions)}
    return id_to_idx.get(left_id), id_to_idx.get(right_id)


def chunked(items: list[str], chunk_size: int) -> list[list[str]]:
    if chunk_size <= 0:
        chunk_size = len(items) or 1
    return [items[i : i + chunk_size] for i in range(0, len(items), chunk_size)]


def cached_dense_tokens(
    model: SteerViTTrainable,
    image_tensor: torch.Tensor,
    prompts: list[str],
    prompt_batch_size: int,
) -> dict[str, torch.Tensor]:
    cache: dict[str, torch.Tensor] = {}
    if not prompts:
        return cache

    for prompt_batch in chunked(prompts, prompt_batch_size):
        images = image_tensor.unsqueeze(0).repeat(len(prompt_batch), 1, 1, 1).to(model.device_name)
        out = model.forward_outputs(images, texts=prompt_batch)
        for i, prompt in enumerate(prompt_batch):
            cache[prompt] = out.dense_tokens[i].detach().cpu()
    return cache


def _metric_view(row: dict[str, Any]) -> dict[str, float]:
    return {
        k: float(v)
        for k, v in row.items()
        if isinstance(v, (int, float)) and not isinstance(v, bool) and not k.endswith("_idx")
    }


def summarize_pair_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    flat_rows = [_metric_view(row) for row in rows]

    summary = aggregate_scalar_metrics(flat_rows)
    summary["num_pairs"] = len(rows)

    grouped = defaultdict(list)
    for row in rows:
        attr_type = row.get("attribute_type", "unknown")
        grouped[attr_type].append(_metric_view(row))

    by_attr_type = {}
    for attr_type, group_rows in grouped.items():
        metrics = aggregate_scalar_metrics(group_rows)
        metrics["num_pairs"] = len(group_rows)
        by_attr_type[attr_type] = metrics
    if by_attr_type:
        summary["by_attribute_type"] = by_attr_type
    return summary


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    cfg = deep_update(cfg, parse_override_pairs(args.override))
    if args.device is not None:
        cfg["device"] = args.device

    seed_everything(cfg.get("seed", 42))
    output_dir = ensure_dir(args.output_dir)
    write_json(output_dir / "resolved_config.json", cfg)
    topology_cfg = cfg.get("topology", {})
    ks = [int(k) for k in topology_cfg.get("ks", [3, 5, 10])]
    prompt_batch_size = int(topology_cfg.get("prompt_batch_size", 16))
    local_overlap_threshold = float(topology_cfg.get("local_overlap_threshold", 0.05))
    far_overlap_threshold = float(topology_cfg.get("far_overlap_threshold", 0.0))
    patch_local_threshold = float(topology_cfg.get("patch_local_threshold", 0.25))
    patch_far_threshold = float(topology_cfg.get("patch_far_threshold", 0.0))
    gate_factor = float(topology_cfg.get("gate_factor", 1.0))
    max_regions_per_image = topology_cfg.get("max_regions_per_image")
    include_rank_correlation = bool(topology_cfg.get("include_rank_correlation", True))

    model = SteerViTTrainable.from_any_checkpoint(
        checkpoint=args.checkpoint,
        base_checkpoint=args.base_checkpoint,
        device=cfg.get("device", "cuda"),
        trainable_modules=(),
    )
    model.eval()
    model.set_gate_factor(gate_factor)
    image_transform = model.get_transforms()
    write_json(output_dir / "checkpoint_load_info.json", getattr(model, "load_info", {}))

    pair_records = read_jsonl(args.manifest)
    if args.pair_id is not None:
        pair_records = [record for record in pair_records if record.get("id") == args.pair_id]
    if args.limit is not None:
        pair_records = pair_records[: args.limit]
    if not pair_records:
        raise ValueError("No paired records found after applying filters.")

    region_bank = load_region_bank(args.region_bank, max_regions_per_image=max_regions_per_image)

    by_image: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in pair_records:
        by_image[record_image_key(record)].append(record)

    rows: list[dict[str, Any]] = []
    counters = Counter()

    for image_key, pairs in tqdm(sorted(by_image.items(), key=lambda x: x[0]), desc="Topology rearrangement"):
        counters["num_image_groups"] += 1
        bank_entry = region_bank.get(image_key)
        if bank_entry is None:
            counters["num_missing_bank"] += len(pairs)
            continue
        bank_regions = list(bank_entry.get("regions", []))
        if len(bank_regions) < 3:
            counters["num_skipped_small_bank"] += len(pairs)
            continue

        image_path = bank_entry.get("image_path") or pairs[0].get("image_path")
        if not image_path or not Path(image_path).exists():
            counters["num_missing_image"] += len(pairs)
            continue

        image_tensor = load_image_tensor(image_path, image_transform)

        unique_prompts = sorted({p for pair in pairs for p in (pair.get("prompt_pos"), pair.get("prompt_neg")) if p})
        dense_cache = cached_dense_tokens(model, image_tensor, unique_prompts, prompt_batch_size=prompt_batch_size)
        off_dense = model.forward_outputs(image_tensor.unsqueeze(0).to(model.device_name), texts=None).dense_tokens[0].detach().cpu()

        bank_patch_masks = []
        mask_cache: dict[str, torch.Tensor] = {}
        patch_cache: dict[str, torch.Tensor] = {}
        for region in bank_regions:
            mask_path = region.get("mask_pos_path")
            if not mask_path or not Path(mask_path).exists():
                bank_patch_masks = []
                break
            _, patch_mask = load_patch_mask(mask_path, image_transform, model.patch_grid)
            mask_cache[mask_path] = _
            patch_cache[mask_path] = patch_mask
            bank_patch_masks.append(patch_mask)
        if not bank_patch_masks:
            counters["num_missing_bank_masks"] += len(pairs)
            continue
        bank_patch_masks_t = torch.stack(bank_patch_masks, dim=0)

        entity_cache: dict[str, torch.Tensor] = {"__off__": masked_token_pool(off_dense, bank_patch_masks_t)}
        for prompt, dense in dense_cache.items():
            entity_cache[prompt] = masked_token_pool(dense, bank_patch_masks_t)

        for pair in pairs:
            prompt_pos = pair.get("prompt_pos")
            prompt_neg = pair.get("prompt_neg")
            mask_pos_path = pair.get("mask_pos_path")
            mask_neg_path = pair.get("mask_neg_path")
            if not prompt_pos or not prompt_neg or not mask_pos_path or not mask_neg_path:
                counters["num_skipped_unpaired"] += 1
                continue
            if prompt_pos not in dense_cache or prompt_neg not in dense_cache:
                counters["num_missing_prompt_cache"] += 1
                continue
            if not Path(mask_pos_path).exists() or not Path(mask_neg_path).exists():
                counters["num_missing_pair_masks"] += 1
                continue

            _, patch_pos = load_patch_mask(mask_pos_path, image_transform, model.patch_grid)
            _, patch_neg = load_patch_mask(mask_neg_path, image_transform, model.patch_grid)
            target_idx, distractor_idx = infer_target_distractor_indices(pair, bank_regions)
            force_local = [idx for idx in (target_idx, distractor_idx) if idx is not None]

            pair_row: dict[str, Any] = {
                "id": pair["id"],
                "image_key": image_key,
                "image_path": image_path,
                "split": pair.get("split", "unknown"),
                "family": pair.get("family", "unknown"),
                "attribute_type": (pair.get("meta", {}) or {}).get("attribute_type", "unknown"),
                "prompt_pos": prompt_pos,
                "prompt_neg": prompt_neg,
                "gate_factor": gate_factor,
                "num_bank_regions": len(bank_regions),
                "target_idx": target_idx,
                "distractor_idx": distractor_idx,
            }

            if args.mode in {"entity", "both"}:
                entity_off = entity_cache["__off__"]
                entity_pos = entity_cache[prompt_pos]
                entity_neg = entity_cache[prompt_neg]
                sim_entity_off = cosine_similarity_matrix(entity_off)
                sim_entity_pos = cosine_similarity_matrix(entity_pos)
                sim_entity_neg = cosine_similarity_matrix(entity_neg)
                entity_local, entity_far, entity_overlap = entity_local_far_masks(
                    bank_patch_masks_t,
                    patch_pos,
                    patch_neg,
                    local_threshold=local_overlap_threshold,
                    far_threshold=far_overlap_threshold,
                    force_local_indices=force_local,
                )
                pair_row["entity_local_overlap_mean"] = float(np.mean(entity_overlap[entity_local])) if entity_local.any() else None
                pair_row["entity_far_overlap_mean"] = float(np.mean(entity_overlap[entity_far])) if entity_far.any() else None
                pair_row.update(
                    compute_rearrangement_metrics(
                        sim_entity_pos,
                        sim_entity_neg,
                        ks=ks,
                        prefix="entity_pos_neg",
                        local_mask=entity_local,
                        far_mask=entity_far,
                        include_rank_correlation=include_rank_correlation,
                    )
                )
                pair_row.update(
                    compute_rearrangement_metrics(
                        sim_entity_off,
                        sim_entity_pos,
                        ks=ks,
                        prefix="entity_off_pos",
                        local_mask=entity_local,
                        far_mask=entity_far,
                        include_rank_correlation=False,
                    )
                )
                pair_row.update(
                    compute_rearrangement_metrics(
                        sim_entity_off,
                        sim_entity_neg,
                        ks=ks,
                        prefix="entity_off_neg",
                        local_mask=entity_local,
                        far_mask=entity_far,
                        include_rank_correlation=False,
                    )
                )
                if target_idx is not None and distractor_idx is not None:
                    for k in ks:
                        flip = neighbor_flip_rate_per_node(sim_entity_pos, sim_entity_neg, int(k))
                        pair_row[f"entity_pos_neg_target_neighbor_flip_rate_k{k}"] = float(flip[target_idx])
                        pair_row[f"entity_pos_neg_distractor_neighbor_flip_rate_k{k}"] = float(flip[distractor_idx])
                    if include_rank_correlation:
                        rho = spearman_rank_correlation_per_node(sim_entity_pos, sim_entity_neg)
                        pair_row["entity_pos_neg_target_rank_spearman"] = float(rho[target_idx])
                        pair_row["entity_pos_neg_distractor_rank_spearman"] = float(rho[distractor_idx])
                    pair_row["entity_pos_target_to_distractor_rank"] = node_rank_to_target(sim_entity_pos, target_idx, distractor_idx)
                    pair_row["entity_neg_target_to_distractor_rank"] = node_rank_to_target(sim_entity_neg, target_idx, distractor_idx)
                    pair_row["entity_off_target_to_distractor_rank"] = node_rank_to_target(sim_entity_off, target_idx, distractor_idx)
                    pair_row["entity_pos_target_distractor_similarity"] = float(sim_entity_pos[target_idx, distractor_idx].item())
                    pair_row["entity_neg_target_distractor_similarity"] = float(sim_entity_neg[target_idx, distractor_idx].item())
                    pair_row["entity_off_target_distractor_similarity"] = float(sim_entity_off[target_idx, distractor_idx].item())

            if args.mode in {"patch", "both"}:
                sim_patch_off = cosine_similarity_matrix(off_dense)
                sim_patch_pos = cosine_similarity_matrix(dense_cache[prompt_pos])
                sim_patch_neg = cosine_similarity_matrix(dense_cache[prompt_neg])
                patch_local, patch_far, patch_overlap = patch_local_far_masks(
                    patch_pos,
                    patch_neg,
                    local_threshold=patch_local_threshold,
                    far_threshold=patch_far_threshold,
                )
                pair_row["patch_local_overlap_mean"] = float(np.mean(patch_overlap[patch_local])) if patch_local.any() else None
                pair_row["patch_far_overlap_mean"] = float(np.mean(patch_overlap[patch_far])) if patch_far.any() else None
                pair_row.update(
                    compute_rearrangement_metrics(
                        sim_patch_pos,
                        sim_patch_neg,
                        ks=ks,
                        prefix="patch_pos_neg",
                        local_mask=patch_local,
                        far_mask=patch_far,
                        include_rank_correlation=False,
                    )
                )
                pair_row.update(
                    compute_rearrangement_metrics(
                        sim_patch_off,
                        sim_patch_pos,
                        ks=ks,
                        prefix="patch_off_pos",
                        local_mask=patch_local,
                        far_mask=patch_far,
                        include_rank_correlation=False,
                    )
                )
                pair_row.update(
                    compute_rearrangement_metrics(
                        sim_patch_off,
                        sim_patch_neg,
                        ks=ks,
                        prefix="patch_off_neg",
                        local_mask=patch_local,
                        far_mask=patch_far,
                        include_rank_correlation=False,
                    )
                )

            rows.append(pair_row)
            counters["num_pairs_evaluated"] += 1

    write_jsonl(output_dir / "per_pair.jsonl", rows)

    summary = summarize_pair_metrics(rows)
    summary["config"] = {
        "manifest": args.manifest,
        "region_bank": args.region_bank,
        "checkpoint": args.checkpoint,
        "base_checkpoint": args.base_checkpoint,
        "mode": args.mode,
        "gate_factor": gate_factor,
        "ks": ks,
        "local_overlap_threshold": local_overlap_threshold,
        "far_overlap_threshold": far_overlap_threshold,
        "patch_local_threshold": patch_local_threshold,
        "patch_far_threshold": patch_far_threshold,
        "max_regions_per_image": max_regions_per_image,
    }
    summary["counters"] = dict(counters)
    write_json(output_dir / "summary.json", summary)
    print(f"Wrote {len(rows)} pair rows to {output_dir / 'per_pair.jsonl'}")
    print(f"Summary written to {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
