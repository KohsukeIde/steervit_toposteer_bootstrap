#!/usr/bin/env python
from __future__ import annotations

import argparse
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from toposteer.datasets import UnifiedRefExpDataset, collate_refexp
from toposteer.models import FrancaAdapter
from toposteer.utils import ensure_dir, patchify_soft_mask, seed_everything, write_json, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare Franca raw vs RASA patch-token spaces on entity masks.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--arch", default="vitb14", choices=["vitb14", "vitl14", "vitg14"])
    parser.add_argument("--weights", default="IN21K")
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--k-values", type=int, nargs="+", default=[1, 5, 10])
    parser.add_argument("--position-sample-patches", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _entity_meta(meta: dict[str, Any], role: str) -> dict[str, Any]:
    if "left_meta" in meta or "right_meta" in meta:
        return meta.get("left_meta" if role == "pos" else "right_meta", {}) or {}
    return meta


def _semantic_label(family: str, meta: dict[str, Any], prompt: str) -> str:
    if family in {"attr", "part_attr"} and meta.get("attribute_name"):
        return f"attr:{meta.get('attribute_type') or 'unk'}:{meta['attribute_name']}"
    if family in {"part", "part_attr"} and meta.get("part_name"):
        return f"part:{meta['part_name']}"
    if meta.get("object_name"):
        return f"object:{meta['object_name']}"
    return f"prompt:{prompt}"


def masked_pool(tokens: torch.Tensor, patch_mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    weights = patch_mask.float()
    weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(eps)
    return torch.einsum("bn,bnd->bd", weights, tokens)


def mask_retrieval_gap(tokens: torch.Tensor, entity: torch.Tensor, pos_mask: torch.Tensor, neg_mask: torch.Tensor) -> torch.Tensor:
    tokens = F.normalize(tokens.float(), dim=-1)
    entity = F.normalize(entity.float(), dim=-1)
    sim = torch.einsum("bnd,bd->bn", tokens, entity)
    pos_score = (sim * pos_mask).sum(dim=1) / pos_mask.sum(dim=1).clamp_min(1e-6)
    neg_score = (sim * neg_mask).sum(dim=1) / neg_mask.sum(dim=1).clamp_min(1e-6)
    return pos_score - neg_score


def position_leakage_corr(tokens: torch.Tensor, patch_grid: tuple[int, int], sample_patches: int) -> float:
    n = tokens.size(0)
    if n < 3:
        return float("nan")

    sample_idx = None
    if sample_patches > 0 and n > sample_patches:
        sample_idx = torch.linspace(0, n - 1, steps=sample_patches, device=tokens.device).long()
        tokens = tokens[sample_idx]
        n = tokens.size(0)

    gh, gw = patch_grid
    yy, xx = torch.meshgrid(
        torch.linspace(0, 1, gh, device=tokens.device),
        torch.linspace(0, 1, gw, device=tokens.device),
        indexing="ij",
    )
    pos = torch.stack([yy.flatten(), xx.flatten()], dim=1)
    if sample_idx is not None:
        pos = pos[sample_idx]
    elif pos.size(0) != tokens.size(0):
        pos = pos[: tokens.size(0)]

    sim = F.normalize(tokens.float(), dim=-1) @ F.normalize(tokens.float(), dim=-1).T
    spatial = -torch.cdist(pos.float(), pos.float())
    tri = torch.triu_indices(n, n, offset=1, device=tokens.device)
    a = sim[tri[0], tri[1]]
    b = spatial[tri[0], tri[1]]
    a = a - a.mean()
    b = b - b.mean()
    denom = torch.sqrt((a * a).sum() * (b * b).sum()).clamp_min(1e-6)
    return float(((a * b).sum() / denom).detach().cpu().item())


def mean_or_nan(values: list[float]) -> float:
    values = [v for v in values if math.isfinite(v)]
    if not values:
        return float("nan")
    return float(sum(values) / len(values))


def summarize_knn(entities: list[dict[str, Any]], stream: str, k_values: list[int]) -> dict[str, float]:
    if len(entities) < 2:
        return {}

    label_counts = Counter(e["label"] for e in entities)
    valid = [i for i, e in enumerate(entities) if label_counts[e["label"]] > 1]
    if not valid:
        return {}

    feats = torch.stack([e[f"{stream}_entity"] for e in entities], dim=0)
    feats = F.normalize(feats.float(), dim=-1)
    sim = feats @ feats.T
    sim.fill_diagonal_(-float("inf"))
    labels = [e["label"] for e in entities]
    families = [e["family"] for e in entities]

    metrics: dict[str, float] = {}
    for k in k_values:
        kk = min(k, len(entities) - 1)
        if kk <= 0:
            continue
        topk = sim.topk(kk, dim=1).indices.cpu().tolist()
        vals = []
        by_family: dict[str, list[float]] = defaultdict(list)
        for i in valid:
            score = sum(labels[j] == labels[i] for j in topk[i]) / kk
            vals.append(score)
            by_family[families[i]].append(score)
        metrics[f"{stream}_knn_purity@{k}"] = mean_or_nan(vals)
        for family, family_vals in by_family.items():
            metrics[f"{stream}_knn_purity@{k}_{family}"] = mean_or_nan(family_vals)
    return metrics


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    output_dir = ensure_dir(args.output_dir)

    model = FrancaAdapter(
        arch=args.arch,
        weights=args.weights,
        device=args.device,
        image_size=args.image_size,
        use_rasa_head=True,
    )
    dataset = UnifiedRefExpDataset(args.manifest, image_transform=model.get_transforms(), limit=args.limit)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_refexp,
    )

    entities: list[dict[str, Any]] = []
    per_record_rows: list[dict[str, Any]] = []
    raw_pos_corrs: list[float] = []
    rasa_pos_corrs: list[float] = []

    for batch in tqdm(loader, desc="Franca raw/RASA diagnostic"):
        images = batch["images"].to(model.device_name)
        masks_pos = batch["masks_pos"].to(model.device_name)
        masks_neg = batch["masks_neg"].to(model.device_name)
        valid_neg_mask = batch["valid_neg_mask"].to(model.device_name)

        out = model.forward_outputs(images)
        patch_pos = patchify_soft_mask(masks_pos, model.patch_grid, normalize=False).to(model.device_name)
        patch_neg = patchify_soft_mask(masks_neg, model.patch_grid, normalize=False).to(model.device_name)

        raw_pos = masked_pool(out.raw_patch_tokens, patch_pos)
        rasa_pos = masked_pool(out.rasa_patch_tokens, patch_pos)
        raw_gap = mask_retrieval_gap(out.raw_patch_tokens, raw_pos, patch_pos, patch_neg)
        rasa_gap = mask_retrieval_gap(out.rasa_patch_tokens, rasa_pos, patch_pos, patch_neg)

        raw_neg = masked_pool(out.raw_patch_tokens, patch_neg)
        rasa_neg = masked_pool(out.rasa_patch_tokens, patch_neg)
        raw_sep = 1.0 - F.cosine_similarity(raw_pos, raw_neg, dim=-1)
        rasa_sep = 1.0 - F.cosine_similarity(rasa_pos, rasa_neg, dim=-1)

        for i in range(images.size(0)):
            has_neg = bool(valid_neg_mask[i].item())
            raw_pos_corrs.append(position_leakage_corr(out.raw_patch_tokens[i], model.patch_grid, args.position_sample_patches))
            rasa_pos_corrs.append(position_leakage_corr(out.rasa_patch_tokens[i], model.patch_grid, args.position_sample_patches))

            family = batch["families"][i]
            meta = batch["meta"][i]
            pos_meta = _entity_meta(meta, "pos")
            pos_label = _semantic_label(family, pos_meta, batch["prompts_pos"][i])
            entities.append(
                {
                    "id": batch["ids"][i],
                    "role": "pos",
                    "family": family,
                    "label": pos_label,
                    "raw_entity": raw_pos[i].detach().cpu(),
                    "rasa_entity": rasa_pos[i].detach().cpu(),
                }
            )

            row = {
                "id": batch["ids"][i],
                "family": family,
                "pos_label": pos_label,
                "has_neg": has_neg,
                "raw_mask_retrieval_gap": float(raw_gap[i].detach().cpu().item()) if has_neg else float("nan"),
                "rasa_mask_retrieval_gap": float(rasa_gap[i].detach().cpu().item()) if has_neg else float("nan"),
                "raw_position_leakage_corr": raw_pos_corrs[-1],
                "rasa_position_leakage_corr": rasa_pos_corrs[-1],
            }

            if has_neg:
                neg_meta = _entity_meta(meta, "neg")
                neg_prompt = batch["prompts_neg"][i] or ""
                neg_label = _semantic_label(family, neg_meta, neg_prompt)
                entities.append(
                    {
                        "id": batch["ids"][i],
                        "role": "neg",
                        "family": family,
                        "label": neg_label,
                        "raw_entity": raw_neg[i].detach().cpu(),
                        "rasa_entity": rasa_neg[i].detach().cpu(),
                    }
                )
                row["neg_label"] = neg_label
                row["raw_pos_neg_separation"] = float(raw_sep[i].detach().cpu().item())
                row["rasa_pos_neg_separation"] = float(rasa_sep[i].detach().cpu().item())

            per_record_rows.append(row)

    summary: dict[str, Any] = {
        "num_records": len(per_record_rows),
        "num_entities": len(entities),
        "arch": args.arch,
        "weights": args.weights,
        "image_size": model.image_size,
        "patch_grid": model.patch_grid,
        "raw_mask_retrieval_gap": mean_or_nan([r["raw_mask_retrieval_gap"] for r in per_record_rows]),
        "rasa_mask_retrieval_gap": mean_or_nan([r["rasa_mask_retrieval_gap"] for r in per_record_rows]),
        "raw_position_leakage_corr": mean_or_nan(raw_pos_corrs),
        "rasa_position_leakage_corr": mean_or_nan(rasa_pos_corrs),
    }
    if any("raw_pos_neg_separation" in r for r in per_record_rows):
        summary["raw_pos_neg_separation"] = mean_or_nan(
            [r["raw_pos_neg_separation"] for r in per_record_rows if "raw_pos_neg_separation" in r]
        )
        summary["rasa_pos_neg_separation"] = mean_or_nan(
            [r["rasa_pos_neg_separation"] for r in per_record_rows if "rasa_pos_neg_separation" in r]
        )

    summary.update(summarize_knn(entities, "raw", args.k_values))
    summary.update(summarize_knn(entities, "rasa", args.k_values))

    write_json(output_dir / "summary.json", summary)
    write_jsonl(output_dir / "per_entity.jsonl", per_record_rows)
    print(f"Wrote Franca diagnostic results to {output_dir}")


if __name__ == "__main__":
    main()
