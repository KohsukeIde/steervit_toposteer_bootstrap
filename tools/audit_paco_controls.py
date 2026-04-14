#!/usr/bin/env python
from __future__ import annotations

import argparse
import itertools
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any

from tqdm import tqdm

from toposteer.utils.io import read_jsonl, write_json, write_jsonl
from toposteer.utils.mask_ops import load_binary_mask, mask_iou


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit whether a PACO manifest actually contains strict controlled pairs.")
    parser.add_argument("--input-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-iou", type=float, default=0.5)
    parser.add_argument("--min-mask-area", type=float, default=10.0)
    parser.add_argument("--max-preview", type=int, default=200)
    return parser.parse_args()


@lru_cache(maxsize=200000)
def _load_mask(path: str):
    return load_binary_mask(path)


@lru_cache(maxsize=200000)
def _mask_area(path: str) -> float:
    return float(_load_mask(path).sum().item())


def _norm(x: Any) -> str | None:
    if x is None:
        return None
    text = str(x).lower().replace("_", " ").strip()
    return text or None


def _group_key(record: dict[str, Any]) -> tuple | None:
    family = record.get("family", "plain")
    meta = record.get("meta", {})
    image_path = record.get("image_path")
    if family == "attr":
        return (
            family,
            image_path,
            _norm(meta.get("object_name")),
            _norm(meta.get("attribute_type")),
        )
    if family == "part":
        return (
            family,
            image_path,
            meta.get("parent_obj_ann_id", _norm(meta.get("object_name"))),
        )
    if family == "part_attr":
        return (
            family,
            image_path,
            meta.get("parent_obj_ann_id", _norm(meta.get("object_name"))),
            _norm(meta.get("part_name")),
            _norm(meta.get("attribute_type")),
        )
    return None


def _pair_ok(a: dict[str, Any], b: dict[str, Any], max_iou: float, min_mask_area: float) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if a["id"] == b["id"]:
        reasons.append("same_id")
    if a.get("image_path") != b.get("image_path"):
        reasons.append("different_image")

    if _mask_area(a["mask_pos_path"]) < min_mask_area or _mask_area(b["mask_pos_path"]) < min_mask_area:
        reasons.append("small_mask")

    iou = mask_iou(_load_mask(a["mask_pos_path"]), _load_mask(b["mask_pos_path"]))
    if iou > max_iou:
        reasons.append("high_iou")

    family = a.get("family", "plain")
    ma = a.get("meta", {})
    mb = b.get("meta", {})

    if family == "attr":
        if _norm(ma.get("object_name")) != _norm(mb.get("object_name")):
            reasons.append("object_name_diff")
        if _norm(ma.get("attribute_type")) != _norm(mb.get("attribute_type")):
            reasons.append("attribute_type_diff")
        if _norm(ma.get("attribute_name")) == _norm(mb.get("attribute_name")):
            reasons.append("same_attribute")
        if ma.get("ann_id") == mb.get("ann_id"):
            reasons.append("same_ann")

    if family == "part":
        if ma.get("parent_obj_ann_id") != mb.get("parent_obj_ann_id"):
            reasons.append("parent_diff")
        if _norm(ma.get("part_name")) == _norm(mb.get("part_name")):
            reasons.append("same_part")
        if ma.get("ann_id") == mb.get("ann_id"):
            reasons.append("same_ann")

    if family == "part_attr":
        if ma.get("parent_obj_ann_id") != mb.get("parent_obj_ann_id"):
            reasons.append("parent_diff")
        if _norm(ma.get("part_name")) != _norm(mb.get("part_name")):
            reasons.append("part_diff")
        if _norm(ma.get("attribute_type")) != _norm(mb.get("attribute_type")):
            reasons.append("attribute_type_diff")
        if _norm(ma.get("attribute_name")) == _norm(mb.get("attribute_name")):
            reasons.append("same_attribute")
        if ma.get("ann_id") == mb.get("ann_id"):
            reasons.append("same_ann")

    return len(reasons) == 0, reasons


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    records = read_jsonl(args.input_manifest)
    family_records = [r for r in records if r.get("family") in {"attr", "part", "part_attr"}]

    groups: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for rec in family_records:
        key = _group_key(rec)
        if key is not None:
            groups[key].append(rec)

    valid_examples = []
    invalid_reason_counts = Counter()
    groups_with_any_valid = Counter()
    candidate_pair_counts = Counter()
    group_size_hist = Counter()

    for key, group in tqdm(sorted(groups.items(), key=lambda x: str(x[0])), desc="Auditing PACO groups"):
        family = key[0]
        group_size_hist[f"{family}:{len(group)}"] += 1
        emitted_for_group = 0
        for a, b in itertools.permutations(group, 2):
            candidate_pair_counts[family] += 1
            ok, reasons = _pair_ok(a, b, max_iou=args.max_iou, min_mask_area=args.min_mask_area)
            if ok:
                groups_with_any_valid[family] += 1
                emitted_for_group += 1
                if len(valid_examples) < args.max_preview:
                    valid_examples.append(
                        {
                            "family": family,
                            "group_key": str(key),
                            "left_id": a["id"],
                            "right_id": b["id"],
                            "left_prompt": a["prompt_pos"],
                            "right_prompt": b["prompt_pos"],
                            "left_meta": a.get("meta", {}),
                            "right_meta": b.get("meta", {}),
                            "mask_iou": float(mask_iou(_load_mask(a["mask_pos_path"]), _load_mask(b["mask_pos_path"]))),
                        }
                    )
            else:
                invalid_reason_counts.update(reasons)
        if emitted_for_group == 0 and len(valid_examples) < args.max_preview:
            pass

    summary = {
        "input_manifest": args.input_manifest,
        "num_records": len(records),
        "num_family_records": len(family_records),
        "num_groups": len(groups),
        "candidate_pair_counts": dict(candidate_pair_counts),
        "groups_with_any_valid": dict(groups_with_any_valid),
        "invalid_reason_counts": dict(invalid_reason_counts.most_common()),
        "group_size_hist": dict(group_size_hist),
        "max_iou": args.max_iou,
        "min_mask_area": args.min_mask_area,
    }

    write_json(out_dir / "summary.json", summary)
    write_jsonl(out_dir / "valid_examples.jsonl", valid_examples)
    print(f"Wrote PACO audit to {out_dir}")


if __name__ == "__main__":
    main()
