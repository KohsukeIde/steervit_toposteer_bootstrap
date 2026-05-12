#!/usr/bin/env python
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from tqdm import tqdm

from toposteer.evaluation import record_image_key
from toposteer.utils.io import read_jsonl, write_json, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Group PhraseCut-style records into a per-image region bank.")
    parser.add_argument("--input-manifest", required=True, help="Unified JSONL manifest, typically phrasecut_full/manifest.jsonl")
    parser.add_argument("--output-jsonl", required=True, help="Where to write the per-image region bank JSONL")
    parser.add_argument("--summary-json", required=True, help="Where to write the summary JSON")
    parser.add_argument("--source", default="phrasecut", help="Only keep records whose `source` matches this value. Use '*' to disable.")
    parser.add_argument("--splits", nargs="*", default=None, help="Optional split filter, e.g. val miniv test")
    parser.add_argument("--family-whitelist", nargs="*", default=None)
    parser.add_argument("--require-existing-image", action="store_true")
    parser.add_argument("--require-existing-mask", action="store_true")
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


def main() -> None:
    args = parse_args()
    records = read_jsonl(args.input_manifest)
    allowed_source = None if args.source == "*" else str(args.source)
    allowed_splits = set(args.splits) if args.splits else None
    allowed_families = set(args.family_whitelist) if args.family_whitelist else None

    banks: dict[str, dict[str, Any]] = {}
    stats = Counter()
    by_split = Counter()
    by_family = Counter()

    for idx, record in enumerate(tqdm(records, desc="Building region bank")):
        if args.limit is not None and idx >= args.limit:
            break
        stats["num_seen"] += 1

        if allowed_source is not None and record.get("source") != allowed_source:
            stats["num_filtered_source"] += 1
            continue
        if allowed_splits is not None and record.get("split") not in allowed_splits:
            stats["num_filtered_split"] += 1
            continue
        if allowed_families is not None and record.get("family") not in allowed_families:
            stats["num_filtered_family"] += 1
            continue

        image_path = record.get("image_path")
        mask_path = record.get("mask_pos_path")
        if args.require_existing_image and (not image_path or not Path(image_path).exists()):
            stats["num_missing_image"] += 1
            continue
        if args.require_existing_mask and (not mask_path or not Path(mask_path).exists()):
            stats["num_missing_mask"] += 1
            continue

        key = record_image_key(record)
        meta = record.get("meta", {}) or {}
        entry = banks.setdefault(
            key,
            {
                "image_key": key,
                "image_path": record.get("image_path"),
                "split": record.get("split", "unknown"),
                "source": record.get("source", "unknown"),
                "meta": {
                    "image_id": meta.get("image_id"),
                    "num_records": 0,
                },
                "regions": [],
            },
        )
        entry["regions"].append(_region_stub(record))
        entry["meta"]["num_records"] = len(entry["regions"])

        stats["num_kept_records"] += 1
        by_split[record.get("split", "unknown")] += 1
        by_family[record.get("family", "plain")] += 1

    rows = []
    for key in sorted(banks.keys()):
        row = banks[key]
        row["regions"] = sorted(row["regions"], key=lambda x: x["id"])
        rows.append(row)

    Path(args.output_jsonl).parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_jsonl, rows)
    write_json(
        args.summary_json,
        {
            "input_manifest": args.input_manifest,
            "output_jsonl": args.output_jsonl,
            "num_images": len(rows),
            "num_regions": int(sum(len(r["regions"]) for r in rows)),
            **dict(stats),
            "regions_by_split": dict(by_split),
            "regions_by_family": dict(by_family),
            "max_regions_per_image": max((len(r["regions"]) for r in rows), default=0),
            "min_regions_per_image": min((len(r["regions"]) for r in rows), default=0),
            "mean_regions_per_image": (sum(len(r["regions"]) for r in rows) / len(rows)) if rows else 0.0,
        },
    )
    print(f"Wrote {len(rows)} image-level banks to {args.output_jsonl}")


if __name__ == "__main__":
    main()
