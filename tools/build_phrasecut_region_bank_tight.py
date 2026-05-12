#!/usr/bin/env python
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from tqdm import tqdm

from toposteer.evaluation import record_image_key
from toposteer.utils import (
    candidate_attr_types,
    ensure_dir,
    has_relations,
    normalize_object_name,
    object_name_of,
    read_jsonl,
    write_json,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a tighter, pair-specific PhraseCut region bank.")
    parser.add_argument("--input-manifest", required=True, help="Positive PhraseCut-style manifest")
    parser.add_argument("--pair-manifest", required=True, help="Controlled paired benchmark manifest")
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--require-existing-image", action="store_true")
    parser.add_argument("--require-existing-mask", action="store_true")
    parser.add_argument("--require-non-relational", action="store_true")
    parser.add_argument("--object-scope", choices=["same_object", "image"], default="same_object")
    parser.add_argument("--require-attribute-type-match", action="store_true")
    parser.add_argument("--allow-image-attr-context", action="store_true", help="Allow same-image attr-matched regions even when object name differs")
    parser.add_argument("--context-budget", type=int, default=16, help="Max number of non-endpoint context regions per pair")
    parser.add_argument("--max-regions-per-pair", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def _region_stub(record: dict[str, Any], selection_reason: str) -> dict[str, Any]:
    meta = record.get("meta", {}) or {}
    return {
        "id": record["id"],
        "family": record.get("family", "plain"),
        "prompt_pos": record.get("prompt_pos"),
        "mask_pos_path": record.get("mask_pos_path"),
        "split": record.get("split", "unknown"),
        "selection_reason": selection_reason,
        "meta": {
            "image_id": meta.get("image_id"),
            "task_id": meta.get("task_id"),
            "object_name": meta.get("object_name"),
            "attributes": meta.get("attributes", []),
            "relation_descriptions": meta.get("relation_descriptions", []),
            "ann_ids": meta.get("ann_ids", []),
            "phrase_type": meta.get("phrase_type"),
        },
    }


def _record_ok(record: dict[str, Any], require_existing_image: bool, require_existing_mask: bool) -> bool:
    image_path = record.get("image_path")
    mask_path = record.get("mask_pos_path")
    if require_existing_image and (not image_path or not Path(image_path).exists()):
        return False
    if require_existing_mask and (not mask_path or not Path(mask_path).exists()):
        return False
    return True


def _pair_object_name(pair: dict[str, Any]) -> str | None:
    meta = pair.get("meta", {}) or {}
    name = meta.get("object_name")
    if not name:
        name = (meta.get("left_meta", {}) or {}).get("object_name")
    if not name:
        name = (meta.get("right_meta", {}) or {}).get("object_name")
    return normalize_object_name(name)


def _selection_reason(
    record: dict[str, Any],
    *,
    pair_object_name: str | None,
    pair_attr_type: str | None,
    endpoint_ids: set[str],
    object_scope: str,
    require_non_relational: bool,
    require_attribute_type_match: bool,
) -> str | None:
    rec_id = str(record.get("id"))
    if rec_id in endpoint_ids:
        return "pair_endpoint"

    if require_non_relational and has_relations(record):
        return None

    rec_object = object_name_of(record)
    rec_attr_types = candidate_attr_types(record)
    attr_match = pair_attr_type is None or pair_attr_type in rec_attr_types
    same_object = bool(pair_object_name and rec_object == pair_object_name)

    if object_scope == "same_object":
        if not same_object:
            return None
        if attr_match:
            return "same_object_attr"
        return "same_object_context"

    if attr_match:
        return "image_attr_context"
    return "image_context"


_REASON_PRIORITY = {
    "pair_endpoint": 0,
    "same_object_attr": 1,
    "same_object_context": 2,
    "attr_context": 3,
    "image_attr_context": 4,
    "image_context": 5,
}


def main() -> None:
    args = parse_args()
    source_records = read_jsonl(args.input_manifest)
    _selection_reason.allow_image_attr_context = bool(args.allow_image_attr_context)
    pair_records = read_jsonl(args.pair_manifest)
    if args.limit is not None:
        pair_records = pair_records[: args.limit]

    image_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in source_records:
        image_groups[record_image_key(record)].append(record)
        image_path = record.get("image_path")
        if image_path:
            alias_key = f"path:{image_path}"
            if alias_key != record_image_key(record):
                image_groups[alias_key].append(record)

    rows: list[dict[str, Any]] = []
    stats = Counter()
    reasons = Counter()

    for pair in tqdm(pair_records, desc="Building tight region banks"):
        stats["num_pairs_seen"] += 1
        image_key = record_image_key(pair)
        image_records = image_groups.get(image_key, [])
        if not image_records:
            stats["num_pairs_missing_image_group"] += 1
            continue

        pair_meta = pair.get("meta", {}) or {}
        pair_attr_type = pair_meta.get("attribute_type")
        pair_object_name = _pair_object_name(pair)
        endpoint_ids = {str(x) for x in [pair_meta.get("left_id"), pair_meta.get("right_id")] if x is not None}

        buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in image_records:
            if not _record_ok(record, args.require_existing_image, args.require_existing_mask):
                stats["num_candidate_missing_paths"] += 1
                continue
            reason = _selection_reason(
                record,
                pair_object_name=pair_object_name,
                pair_attr_type=pair_attr_type,
                endpoint_ids=endpoint_ids,
                object_scope=args.object_scope,
                require_non_relational=args.require_non_relational,
                require_attribute_type_match=args.require_attribute_type_match,
            )
            if reason is None:
                continue
            buckets[reason].append(record)

        ordered: list[tuple[str, dict[str, Any]]] = []
        for reason in sorted(buckets.keys(), key=lambda x: _REASON_PRIORITY.get(x, 999)):
            for rec in sorted(buckets[reason], key=lambda r: str(r.get("id"))):
                ordered.append((reason, rec))

        if args.context_budget is not None and args.context_budget >= 0:
            kept: list[tuple[str, dict[str, Any]]] = []
            num_context = 0
            for reason, rec in ordered:
                if reason == "pair_endpoint":
                    kept.append((reason, rec))
                    continue
                if num_context >= int(args.context_budget):
                    continue
                kept.append((reason, rec))
                num_context += 1
            ordered = kept

        if args.max_regions_per_pair is not None and args.max_regions_per_pair > 0:
            ordered = ordered[: int(args.max_regions_per_pair)]

        if len(ordered) < 2:
            stats["num_pairs_too_small"] += 1
            continue

        region_rows = []
        for reason, rec in ordered:
            region_rows.append(_region_stub(rec, reason))
            reasons[reason] += 1

        rows.append(
            {
                "pair_id": pair["id"],
                "image_key": image_key,
                "image_path": pair.get("image_path"),
                "split": pair.get("split", "unknown"),
                "source": pair.get("source", "unknown"),
                "bank_scope": "pair_tight",
                "meta": {
                    "image_id": pair_meta.get("image_id"),
                    "object_name": pair_object_name,
                    "attribute_type": pair_attr_type,
                    "num_records": len(region_rows),
                    "selection_reasons": dict(Counter(r["selection_reason"] for r in region_rows)),
                },
                "regions": region_rows,
            }
        )
        stats["num_pairs_kept"] += 1

    ensure_dir(Path(args.output_jsonl).parent)
    write_jsonl(args.output_jsonl, rows)
    write_json(
        args.summary_json,
        {
            "input_manifest": args.input_manifest,
            "pair_manifest": args.pair_manifest,
            "output_jsonl": args.output_jsonl,
            "num_pair_banks": len(rows),
            "num_regions": int(sum(len(r["regions"]) for r in rows)),
            "selection_reasons": dict(reasons),
            "mean_regions_per_pair": (sum(len(r["regions"]) for r in rows) / len(rows)) if rows else 0.0,
            **dict(stats),
        },
    )
    print(f"Wrote {len(rows)} pair-specific region banks to {args.output_jsonl}")


if __name__ == "__main__":
    main()
