from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import Dataset

from toposteer.utils.io import read_jsonl
from toposteer.utils.mask_ops import apply_geometric_ops_to_mask


class UnifiedRefExpDataset(Dataset):
    def __init__(
        self,
        manifest_path: str | Path,
        image_transform,
        split: str | None = None,
        family_whitelist: list[str] | None = None,
        limit: int | None = None,
    ) -> None:
        records = read_jsonl(manifest_path)

        if split is not None:
            records = [r for r in records if r.get("split") == split]

        if family_whitelist:
            allowed = set(family_whitelist)
            records = [r for r in records if r.get("family", "plain") in allowed]

        if limit is not None:
            records = records[:limit]

        self.records = records
        self.image_transform = image_transform

    def __len__(self) -> int:
        return len(self.records)

    def _load_image(self, path: str | Path) -> torch.Tensor:
        img = Image.open(path).convert("RGB")
        return self.image_transform(img)

    def _load_mask(self, path: str | Path) -> torch.Tensor:
        mask_pil = Image.open(path).convert("L")
        return apply_geometric_ops_to_mask(mask_pil, self.image_transform)

    def __getitem__(self, index: int) -> dict[str, Any]:
        rec = self.records[index]
        item: dict[str, Any] = {
            "id": rec["id"],
            "source": rec.get("source", "custom"),
            "split": rec.get("split", "custom"),
            "family": rec.get("family", "plain"),
            "image_path": rec["image_path"],
            "image": self._load_image(rec["image_path"]),
            "prompt_pos": rec["prompt_pos"],
            "meta": rec.get("meta", {}),
        }

        item["mask_pos"] = self._load_mask(rec["mask_pos_path"])
        item["mask_pos_path"] = rec["mask_pos_path"]

        if rec.get("mask_neg_path"):
            item["mask_neg"] = self._load_mask(rec["mask_neg_path"])
            item["mask_neg_path"] = rec["mask_neg_path"]
            item["has_neg"] = True
        else:
            item["mask_neg"] = None
            item["mask_neg_path"] = None
            item["has_neg"] = False

        item["prompt_neg"] = rec.get("prompt_neg")
        item["has_prompt_neg"] = item["prompt_neg"] is not None
        return item


def collate_refexp(batch: list[dict[str, Any]]) -> dict[str, Any]:
    images = torch.stack([x["image"] for x in batch], dim=0)
    masks_pos = torch.stack([x["mask_pos"] for x in batch], dim=0)

    masks_neg = torch.stack(
        [x["mask_neg"] if x["mask_neg"] is not None else torch.zeros_like(x["mask_pos"]) for x in batch],
        dim=0,
    )
    valid_neg_mask = torch.tensor([bool(x["has_neg"]) for x in batch], dtype=torch.bool)
    valid_prompt_neg_mask = torch.tensor(
        [bool(x["has_neg"] and x["has_prompt_neg"]) for x in batch],
        dtype=torch.bool,
    )

    return {
        "ids": [x["id"] for x in batch],
        "images": images,
        "masks_pos": masks_pos,
        "masks_neg": masks_neg,
        "valid_neg_mask": valid_neg_mask,
        "valid_prompt_neg_mask": valid_prompt_neg_mask,
        "prompts_pos": [x["prompt_pos"] for x in batch],
        "prompts_neg": [x["prompt_neg"] for x in batch],
        "families": [x["family"] for x in batch],
        "sources": [x["source"] for x in batch],
        "image_paths": [x["image_path"] for x in batch],
        "meta": [x["meta"] for x in batch],
    }
