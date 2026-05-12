#!/usr/bin/env python
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from tqdm import tqdm

from toposteer.evaluation import record_image_key
from toposteer.utils import read_jsonl, write_json, write_jsonl
from toposteer.utils.phrasecut_filters import candidate_attr_types, has_relations, normalize_object_name, object_name_of


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a tighter PhraseCut region bank keyed by pair_id for topology falsification tests.")
    parser.add_argument("--pair-manifest", required=True, help="Paired eval manifest, e.g. gold_eval.jsonl")
    parser.add_argument("--source-manifest", required=True, help="Positive PhraseCut manifest used to source candidate regions")
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--source", default="phrasecut", help="Only keep records whose source matches this value. Use '*' to disable.")
    parser.add_argument("--require-existing-image", action="store_true")
    parser.add_argument("--require-existing-mask", action="store_true")
    parser.add_argument("--require-non-relational", action="store_true", help="Drop regions with relation descriptions")
    parser.add_argument("--same-object-name", action="store_true", help="Keep only regions whose normalized object_name matches the pair object")
    parser.add_argument("--match-pair-attribute-type", action="store_true", help="Keep only regions that expose the same canonical attribute type as the pair")
    parser.add_argument("--keep-attr-types", nargs="*", default=None, help="Optional canonical attribute-type whitelist")
    parser.add_argument("--require-color-bearing", action="store_true", help="Keep only regions that expose a canonical color attribute")
    parser.add_argument("--keep-target-distractor-always", action="store_true", help="Always include left/right pair members when available")
    parser.add_argument("--max-regions-per-pair", type=int, default=None)
    parser.add_argument("--min-regions-per-pair", type=int, default=3)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def _region_stub(record: dict[str, Any]) -> dict[str, Any]:
    meta = record.get("meta", {}) or {}
    return {
        "id": record["id"],
        "family": record.get("family", "plain"),
        "prompt_pos": record.get("prompt_pos"),
        "mask_pos_path": record.get("mask_pos_path"),
        "split": record.get("split", "unknown"),
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


def _pair_object_name(pair: dict[str, Any]) -> str | None:
    meta = pair.get("meta", {}) or {}
    names = [
        meta.get("object_name"),
        (meta.get("left_meta", {}) or {}).get("object_name"),
        (meta.get("right_meta", {}) or {}).get("object_name"),
    ]
    for name in names:
        norm = normalize_object_name(name)
        if norm:
            return norm
    return None


def _pair_attr_type(pair: dict[str, Any]) -> str | None:
    meta = pair.get("meta", {}) or {}
    value = meta.get("attribute_type")
    return str(value) if value else None


def main() -> None:
    args = parse_args()
    pair_records = read_jsonl(args.pair_manifest)
    if args.limit is not None:
        pair_records = pair_records[: args.limit]
    source_records = read_jsonl(args.source_manifest)

    allowed_source = None if args.source == "*" else str(args.source)
    keep_attr_types = set(args.keep_attr_types) if args.keep_attr_types else None

    source_by_image: dict[str, list[dict[str, Any]]] = defaultdict(list)
    source_by_id: dict[str, dict[str, Any]] = {}
    source_stats = Counter()
    for record in source_records:
        if allowed_source is not None and record.get("source") != allowed_source:
            continue
        image_path = record.get("image_path")
        mask_path = record.get("mask_pos_path")
        if args.require_existing_image and (not image_path or not Path(image_path).exists()):
            source_stats["num_missing_image"] += 1
            continue
        if args.require_existing_mask and (not mask_path or not Path(mask_path).exists()):
            source_stats["num_missing_mask"] += 1
            continue
        key = record_image_key(record)
        source_by_image[key].append(record)
        image_path = record.get("image_path")
        if image_path:
            alias_key = f"path:{image_path}"
            if alias_key != key:
                source_by_image[alias_key].append(record)
        source_by_id[str(record.get("id"))] = record
        source_stats["num_source_records_kept"] += 1

    rows = []
    stats = Counter()
    regions_per_pair = []
    regions_by_attr_type = Counter()

    for pair in tqdm(pair_records, desc="Building tighter pair bank"):
        stats["num_pairs_seen"] += 1
        image_key = record_image_key(pair)
        pair_id = str(pair.get("id"))
        object_name = _pair_object_name(pair)
        pair_attr_type = _pair_attr_type(pair)
        left_id = str((pair.get("meta", {}) or {}).get("left_id")) if (pair.get("meta", {}) or {}).get("left_id") is not None else None
        right_id = str((pair.get("meta", {}) or {}).get("right_id")) if (pair.get("meta", {}) or {}).get("right_id") is not None else None

        candidates = []
        local_stats = Counter()
        for record in source_by_image.get(image_key, []):
            record_id = str(record.get("id"))
            keep = True
            forced = False
            if args.keep_target_distractor_always and record_id in {left_id, right_id}:
                keep = True
                forced = True
            if keep and not forced and args.require_non_relational and has_relations(record):
                keep = False
                local_stats["filtered_relational"] += 1
            record_object = object_name_of(record)
            if keep and not forced and args.same_object_name and object_name and record_object != object_name:
                keep = False
                local_stats["filtered_object_name"] += 1
            record_attr_types = candidate_attr_types(record)
            if keep and not forced and keep_attr_types is not None and not (record_attr_types & keep_attr_types):
                keep = False
                local_stats["filtered_attr_whitelist"] += 1
            if keep and not forced and args.match_pair_attribute_type and pair_attr_type and pair_attr_type not in record_attr_types:
                keep = False
                local_stats["filtered_attr_type"] += 1
            if keep and not forced and args.require_color_bearing and "color" not in record_attr_types:
                keep = False
                local_stats["filtered_not_color_bearing"] += 1
            if keep:
                stub = _region_stub(record)
                stub["debug"] = {
                    "forced_keep": bool(forced),
                    "normalized_object_name": record_object,
                    "attribute_types": sorted(record_attr_types),
                    "has_relations": bool(has_relations(record)),
                }
                candidates.append(stub)
                local_stats["num_regions_kept"] += 1

        deduped: dict[str, dict[str, Any]] = {}
        for region in candidates:
            deduped[str(region["id"])] = region
        candidates = list(deduped.values())

        def _sort_key(region: dict[str, Any]):
            rid = str(region.get("id"))
            forced_rank = 0 if rid in {left_id, right_id} else 1
            debug = region.get("debug", {}) or {}
            attr_match = 0 if (pair_attr_type and pair_attr_type in (debug.get("attribute_types") or [])) else 1
            rel_rank = 1 if debug.get("has_relations") else 0
            return (forced_rank, attr_match, rel_rank, rid)

        candidates = sorted(candidates, key=_sort_key)
        if args.max_regions_per_pair is not None and args.max_regions_per_pair > 0:
            candidates = candidates[: int(args.max_regions_per_pair)]

        rows.append(
            {
                "pair_id": pair_id,
                "image_key": image_key,
                "image_path": pair.get("image_path"),
                "source": pair.get("source", "unknown"),
                "split": pair.get("split", "unknown"),
                "meta": {
                    "object_name": object_name,
                    "attribute_type": pair_attr_type,
                    "left_id": left_id,
                    "right_id": right_id,
                    "num_source_image_records": len(source_by_image.get(image_key, [])),
                    "num_kept_regions": len(candidates),
                    "filters": {
                        "require_non_relational": bool(args.require_non_relational),
                        "same_object_name": bool(args.same_object_name),
                        "match_pair_attribute_type": bool(args.match_pair_attribute_type),
                        "keep_attr_types": sorted(keep_attr_types) if keep_attr_types else None,
                        "require_color_bearing": bool(args.require_color_bearing),
                        "keep_target_distractor_always": bool(args.keep_target_distractor_always),
                    },
                    "filter_counters": dict(local_stats),
                },
                "regions": candidates,
            }
        )
        regions_per_pair.append(len(candidates))
        if len(candidates) >= int(args.min_regions_per_pair):
            stats["num_pairs_meeting_min_regions"] += 1
        if left_id and any(str(r.get("id")) == left_id for r in candidates):
            stats["num_pairs_with_left"] += 1
        if right_id and any(str(r.get("id")) == right_id for r in candidates):
            stats["num_pairs_with_right"] += 1
        if pair_attr_type:
            regions_by_attr_type[pair_attr_type] += len(candidates)

    Path(args.output_jsonl).parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_jsonl, rows)
    write_json(
        args.summary_json,
        {
            "pair_manifest": args.pair_manifest,
            "source_manifest": args.source_manifest,
            "num_pairs": len(rows),
            "num_source_images": len(source_by_image),
            **dict(stats),
            **{f"source_{k}": v for k, v in source_stats.items()},
            "regions_per_pair": {
                "min": min(regions_per_pair) if regions_per_pair else 0,
                "max": max(regions_per_pair) if regions_per_pair else 0,
                "mean": (sum(regions_per_pair) / len(regions_per_pair)) if regions_per_pair else 0.0,
            },
            "regions_by_pair_attribute_type": dict(regions_by_attr_type),
            "settings": {
                "require_non_relational": bool(args.require_non_relational),
                "same_object_name": bool(args.same_object_name),
                "match_pair_attribute_type": bool(args.match_pair_attribute_type),
                "keep_attr_types": sorted(keep_attr_types) if keep_attr_types else None,
                "require_color_bearing": bool(args.require_color_bearing),
                "keep_target_distractor_always": bool(args.keep_target_distractor_always),
                "max_regions_per_pair": args.max_regions_per_pair,
                "min_regions_per_pair": args.min_regions_per_pair,
            },
        },
    )
    print(f"Wrote {len(rows)} pair-bank entries to {args.output_jsonl}")


if __name__ == "__main__":
    main()
