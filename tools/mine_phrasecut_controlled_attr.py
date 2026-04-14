#!/usr/bin/env python
from __future__ import annotations

import argparse
import itertools
import re
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any

from tqdm import tqdm

from toposteer.utils.io import read_jsonl, write_json, write_jsonl
from toposteer.utils.mask_ops import load_binary_mask, mask_iou


COLOR_ALIASES = {
    "red": "red",
    "blue": "blue",
    "green": "green",
    "yellow": "yellow",
    "orange": "orange",
    "purple": "purple",
    "pink": "pink",
    "brown": "brown",
    "black": "black",
    "white": "white",
    "gray": "grey",
    "grey": "grey",
    "silver": "silver",
    "gold": "gold",
    "beige": "beige",
}
MATERIAL_ALIASES = {
    "wood": "wood",
    "wooden": "wood",
    "metal": "metal",
    "metallic": "metal",
    "plastic": "plastic",
    "glass": "glass",
    "paper": "paper",
    "cardboard": "cardboard",
    "leather": "leather",
    "fabric": "fabric",
    "cloth": "cloth",
    "rubber": "rubber",
    "stone": "stone",
    "concrete": "concrete",
    "ceramic": "ceramic",
}
SIZE_ALIASES = {
    "small": "small",
    "little": "small",
    "tiny": "small",
    "large": "large",
    "big": "large",
    "huge": "large",
    "tall": "tall",
    "short": "short",
    "long": "long",
    "wide": "wide",
    "narrow": "narrow",
}
STATE_ALIASES = {
    "open": "open",
    "closed": "closed",
    "on": "on",
    "off": "off",
    "empty": "empty",
    "full": "full",
    "broken": "broken",
    "clean": "clean",
    "dirty": "dirty",
    "folded": "folded",
    "unfolded": "unfolded",
    "ripe": "ripe",
    "unripe": "unripe",
    "wet": "wet",
    "dry": "dry",
}
SPATIAL_ALIASES = {
    "left": "left",
    "right": "right",
    "top": "top",
    "bottom": "bottom",
    "front": "front",
    "back": "back",
    "center": "center",
    "middle": "center",
    "upper": "upper",
    "lower": "lower",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mine controlled PhraseCut attribute flip sets.")
    parser.add_argument("--input-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-splits", nargs="*", default=["train"])
    parser.add_argument("--eval-splits", nargs="*", default=["val", "miniv", "test"])
    parser.add_argument("--keep-attr-types", nargs="*", default=["color", "material", "size", "state"])
    parser.add_argument("--max-iou", type=float, default=0.5)
    parser.add_argument("--min-mask-area", type=float, default=10.0)
    parser.add_argument("--max-pairs-per-group", type=int, default=32)
    parser.add_argument("--max-preview", type=int, default=100)
    return parser.parse_args()


@lru_cache(maxsize=200000)
def _load_mask(path: str):
    return load_binary_mask(path)


@lru_cache(maxsize=200000)
def _mask_area(path: str) -> float:
    return float(_load_mask(path).sum().item())


def _normalize_text(text: str | None) -> str:
    if not text:
        return ""
    text = text.lower().strip()
    text = text.replace("-", " ")
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _normalize_object_name(name: str | None) -> str | None:
    norm = _normalize_text(name)
    return norm or None


def _canonicalize_attribute_name(raw: str) -> dict[str, str] | None:
    attr = _normalize_text(raw)
    if not attr:
        return None

    for prefix in ("light ", "dark ", "bright "):
        if attr.startswith(prefix):
            tail = attr[len(prefix) :]
            if tail in COLOR_ALIASES:
                return {"name": COLOR_ALIASES[tail], "type": "color", "raw": attr}

    if attr in COLOR_ALIASES:
        return {"name": COLOR_ALIASES[attr], "type": "color", "raw": attr}
    if attr in MATERIAL_ALIASES:
        return {"name": MATERIAL_ALIASES[attr], "type": "material", "raw": attr}
    if attr in SIZE_ALIASES:
        return {"name": SIZE_ALIASES[attr], "type": "size", "raw": attr}
    if attr in STATE_ALIASES:
        return {"name": STATE_ALIASES[attr], "type": "state", "raw": attr}
    if attr in SPATIAL_ALIASES:
        return {"name": SPATIAL_ALIASES[attr], "type": "spatial", "raw": attr}
    return None


def _extract_canonical_attrs(record: dict[str, Any]) -> list[dict[str, str]]:
    meta = record.get("meta", {})
    attrs = meta.get("attributes") or []
    seen = set()
    out: list[dict[str, str]] = []
    for attr in attrs:
        canon = _canonicalize_attribute_name(attr)
        if canon is None:
            continue
        key = (canon["type"], canon["name"])
        if key in seen:
            continue
        out.append(canon)
        seen.add(key)
    return out


def _has_relations(record: dict[str, Any]) -> bool:
    meta = record.get("meta", {})
    rels = meta.get("relation_descriptions") or []
    return len(rels) > 0


def _ann_signature(record: dict[str, Any]) -> tuple[int, ...]:
    meta = record.get("meta", {})
    ann_ids = meta.get("ann_ids") or []
    try:
        return tuple(sorted(int(x) for x in ann_ids))
    except Exception:
        return tuple()


def _build_candidate(record: dict[str, Any], keep_attr_types: set[str]) -> dict[str, Any] | None:
    if _has_relations(record):
        return None

    object_name = _normalize_object_name(record.get("meta", {}).get("object_name"))
    if object_name is None:
        return None

    attrs = [x for x in _extract_canonical_attrs(record) if x["type"] in keep_attr_types]
    if not attrs:
        return None

    return {
        **record,
        "_object_name_norm": object_name,
        "_canonical_attrs": attrs,
        "_ann_signature": _ann_signature(record),
    }


def _is_gold_candidate(candidate: dict[str, Any]) -> bool:
    attrs = candidate["_canonical_attrs"]
    return len(attrs) == 1


def _is_silver_candidate(candidate: dict[str, Any]) -> bool:
    attrs = candidate["_canonical_attrs"]
    return len(attrs) >= 1


def _candidate_attr(candidate: dict[str, Any]) -> dict[str, str]:
    return candidate["_canonical_attrs"][0]


def _pair_ok(a: dict[str, Any], b: dict[str, Any], max_iou: float, min_mask_area: float) -> bool:
    if a["id"] == b["id"]:
        return False
    if a["image_path"] != b["image_path"]:
        return False
    if a["_object_name_norm"] != b["_object_name_norm"]:
        return False

    attr_a = _candidate_attr(a)
    attr_b = _candidate_attr(b)
    if attr_a["type"] != attr_b["type"]:
        return False
    if attr_a["name"] == attr_b["name"]:
        return False

    if _mask_area(a["mask_pos_path"]) < min_mask_area or _mask_area(b["mask_pos_path"]) < min_mask_area:
        return False
    if mask_iou(_load_mask(a["mask_pos_path"]), _load_mask(b["mask_pos_path"])) > max_iou:
        return False

    ann_a = set(a["_ann_signature"])
    ann_b = set(b["_ann_signature"])
    if ann_a and ann_b and ann_a.intersection(ann_b):
        return False

    return True


def _pair_record(a: dict[str, Any], b: dict[str, Any], tier: str) -> dict[str, Any]:
    attr_a = _candidate_attr(a)
    attr_b = _candidate_attr(b)
    iou = mask_iou(_load_mask(a["mask_pos_path"]), _load_mask(b["mask_pos_path"]))
    control_family = f"attr_same_object_diff_attr_{tier}"
    return {
        "id": f"phrasecut_{tier}_flip_{a['id']}__vs__{b['id']}",
        "source": "phrasecut+flip",
        "split": a.get("split", "custom"),
        "family": "attr",
        "image_path": a["image_path"],
        "mask_pos_path": a["mask_pos_path"],
        "mask_neg_path": b["mask_pos_path"],
        "prompt_pos": a["prompt_pos"],
        "prompt_neg": b["prompt_pos"],
        "meta": {
            "pair_origin": tier,
            "control_family": control_family,
            "same_object_name": True,
            "same_parent": None,
            "same_part": None,
            "changed_fields": ["attribute_name"],
            "num_changed_fields": 1,
            "mask_iou": float(iou),
            "attribute_type": attr_a["type"],
            "attribute_name_pos": attr_a["name"],
            "attribute_name_neg": attr_b["name"],
            "object_name": a["_object_name_norm"],
            "left_id": a["id"],
            "right_id": b["id"],
            "left_ann_signature": list(a["_ann_signature"]),
            "right_ann_signature": list(b["_ann_signature"]),
            "left_meta": a.get("meta", {}),
            "right_meta": b.get("meta", {}),
        },
    }


def _emit_pairs(
    groups: dict[tuple, list[dict[str, Any]]],
    tier: str,
    max_pairs_per_group: int,
    max_iou: float,
    min_mask_area: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    pair_records: list[dict[str, Any]] = []
    previews: list[dict[str, Any]] = []
    for key, group in tqdm(sorted(groups.items(), key=lambda x: str(x[0])), desc=f"Mining {tier}"):
        emitted = 0
        group = sorted(group, key=lambda x: x["id"])
        for a, b in itertools.permutations(group, 2):
            if emitted >= max_pairs_per_group:
                break
            if not _pair_ok(a, b, max_iou=max_iou, min_mask_area=min_mask_area):
                continue
            record = _pair_record(a, b, tier=tier)
            pair_records.append(record)
            previews.append(
                {
                    "key": str(key),
                    "left_id": a["id"],
                    "right_id": b["id"],
                    "object_name": a["_object_name_norm"],
                    "attribute_type": _candidate_attr(a)["type"],
                    "attribute_name_pos": _candidate_attr(a)["name"],
                    "attribute_name_neg": _candidate_attr(b)["name"],
                    "mask_iou": record["meta"]["mask_iou"],
                    "split": a.get("split", "unknown"),
                }
            )
            emitted += 1
    return pair_records, previews


def _group_candidates(records: list[dict[str, Any]]) -> dict[tuple, list[dict[str, Any]]]:
    groups: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for rec in records:
        attr = _candidate_attr(rec)
        key = (
            rec.get("split", "unknown"),
            rec["image_path"],
            rec["_object_name_norm"],
            attr["type"],
        )
        groups[key].append(rec)
    return groups


def _counter_by(records: list[dict[str, Any]], field: str) -> dict[str, int]:
    counter = Counter()
    for rec in records:
        value = rec.get(field)
        if value is not None:
            counter[str(value)] += 1
    return dict(counter)


def main() -> None:
    args = parse_args()
    keep_attr_types = set(args.keep_attr_types)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    records = read_jsonl(args.input_manifest)
    candidates = []
    for rec in records:
        candidate = _build_candidate(rec, keep_attr_types=keep_attr_types)
        if candidate is not None:
            candidates.append(candidate)

    train_splits = set(args.train_splits)
    eval_splits = set(args.eval_splits)

    gold_train = [c for c in candidates if c.get("split") in train_splits and _is_gold_candidate(c)]
    gold_eval = [c for c in candidates if c.get("split") in eval_splits and _is_gold_candidate(c)]
    silver_train = [c for c in candidates if c.get("split") in train_splits and _is_silver_candidate(c)]
    silver_eval = [c for c in candidates if c.get("split") in eval_splits and _is_silver_candidate(c)]

    gold_train_pairs, gold_train_previews = _emit_pairs(
        _group_candidates(gold_train),
        tier="gold_train",
        max_pairs_per_group=args.max_pairs_per_group,
        max_iou=args.max_iou,
        min_mask_area=args.min_mask_area,
    )
    gold_eval_pairs, gold_eval_previews = _emit_pairs(
        _group_candidates(gold_eval),
        tier="gold_eval",
        max_pairs_per_group=args.max_pairs_per_group,
        max_iou=args.max_iou,
        min_mask_area=args.min_mask_area,
    )
    silver_train_pairs, silver_train_previews = _emit_pairs(
        _group_candidates(silver_train),
        tier="silver_train",
        max_pairs_per_group=args.max_pairs_per_group,
        max_iou=args.max_iou,
        min_mask_area=args.min_mask_area,
    )
    silver_eval_pairs, silver_eval_previews = _emit_pairs(
        _group_candidates(silver_eval),
        tier="silver_eval",
        max_pairs_per_group=args.max_pairs_per_group,
        max_iou=args.max_iou,
        min_mask_area=args.min_mask_area,
    )

    write_jsonl(out_dir / "gold_train.jsonl", gold_train_pairs)
    write_jsonl(out_dir / "gold_eval.jsonl", gold_eval_pairs)
    write_jsonl(out_dir / "silver_train.jsonl", silver_train_pairs)
    write_jsonl(out_dir / "silver_eval.jsonl", silver_eval_pairs)
    write_jsonl(out_dir / "gold_train_preview.jsonl", gold_train_previews[: args.max_preview])
    write_jsonl(out_dir / "gold_eval_preview.jsonl", gold_eval_previews[: args.max_preview])
    write_jsonl(out_dir / "silver_train_preview.jsonl", silver_train_previews[: args.max_preview])
    write_jsonl(out_dir / "silver_eval_preview.jsonl", silver_eval_previews[: args.max_preview])

    summary = {
        "input_manifest": args.input_manifest,
        "keep_attr_types": sorted(keep_attr_types),
        "train_splits": sorted(train_splits),
        "eval_splits": sorted(eval_splits),
        "max_iou": args.max_iou,
        "min_mask_area": args.min_mask_area,
        "candidate_counts": {
            "all_candidates": len(candidates),
            "gold_train_candidates": len(gold_train),
            "gold_eval_candidates": len(gold_eval),
            "silver_train_candidates": len(silver_train),
            "silver_eval_candidates": len(silver_eval),
        },
        "pair_counts": {
            "gold_train": len(gold_train_pairs),
            "gold_eval": len(gold_eval_pairs),
            "silver_train": len(silver_train_pairs),
            "silver_eval": len(silver_eval_pairs),
        },
        "by_split": _counter_by(candidates, "split"),
        "gold_eval_attr_types": dict(Counter(rec["meta"]["attribute_type"] for rec in gold_eval_pairs)),
        "silver_train_attr_types": dict(Counter(rec["meta"]["attribute_type"] for rec in silver_train_pairs)),
    }
    write_json(out_dir / "summary.json", summary)
    print(f"Wrote PhraseCut controlled attr sets to {out_dir}")


if __name__ == "__main__":
    main()
