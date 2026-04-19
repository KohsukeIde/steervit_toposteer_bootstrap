#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from toposteer.utils import ensure_dir, read_jsonl, write_json, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Split a controlled attribute flip manifest into locked dev/test manifests without "
            "leaking reverse pairs or equivalent attribute flips across splits."
        )
    )
    parser.add_argument("--input-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--attr-type", default="color")
    parser.add_argument("--dev-pairs", type=int, default=150)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prefix", default="gold_color")
    return parser.parse_args()


def _meta(record: dict[str, Any]) -> dict[str, Any]:
    meta = record.get("meta")
    return meta if isinstance(meta, dict) else {}


def _attr_type(record: dict[str, Any]) -> str:
    return str(_meta(record).get("attribute_type", "unknown"))


def _group_key(record: dict[str, Any]) -> tuple[str, str, str, tuple[str, str]]:
    meta = _meta(record)
    attr_pos = str(meta.get("attribute_name_pos", "")).strip()
    attr_neg = str(meta.get("attribute_name_neg", "")).strip()
    unordered_attrs = tuple(sorted((attr_pos, attr_neg)))
    return (
        str(record.get("image_path", "")),
        str(meta.get("object_name", "")),
        str(meta.get("attribute_type", "unknown")),
        unordered_attrs,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_and_hash(path: Path, records: list[dict[str, Any]]) -> dict[str, Any]:
    write_jsonl(path, records)
    return {"path": str(path), "num_records": len(records), "sha256": _sha256(path)}


def _count_by_attr(records: list[dict[str, Any]]) -> dict[str, int]:
    return dict(Counter(_attr_type(record) for record in records))


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)
    records = read_jsonl(args.input_manifest)

    target = [record for record in records if _attr_type(record) == args.attr_type]
    reference = [record for record in records if _attr_type(record) != args.attr_type]

    groups: dict[tuple[str, str, str, tuple[str, str]], list[dict[str, Any]]] = defaultdict(list)
    for record in target:
        groups[_group_key(record)].append(record)

    group_items = sorted(groups.items(), key=lambda item: repr(item[0]))
    rng = random.Random(args.seed)
    rng.shuffle(group_items)

    dev_groups = []
    test_groups = []
    dev_count = 0
    for key, group in group_items:
        if dev_count < args.dev_pairs:
            dev_groups.append((key, group))
            dev_count += len(group)
        else:
            test_groups.append((key, group))

    dev_records = [record for _, group in dev_groups for record in group]
    test_records = [record for _, group in test_groups for record in group]

    dev_keys = {key for key, _ in dev_groups}
    test_keys = {key for key, _ in test_groups}
    overlap = dev_keys & test_keys
    if overlap:
        raise RuntimeError(f"Internal split leakage: {len(overlap)} group keys appear in both dev and test.")

    all_path = output_dir / f"{args.prefix}_all.jsonl"
    dev_path = output_dir / f"{args.prefix}_dev.jsonl"
    test_path = output_dir / f"{args.prefix}_test.jsonl"
    reference_path = output_dir / "gold_non_color_reference.jsonl"

    manifests = {
        "all": _write_and_hash(all_path, target),
        "dev": _write_and_hash(dev_path, dev_records),
        "test": _write_and_hash(test_path, test_records),
        "non_color_reference": _write_and_hash(reference_path, reference),
    }

    summary = {
        "input_manifest": args.input_manifest,
        "attr_type": args.attr_type,
        "seed": args.seed,
        "requested_dev_pairs": args.dev_pairs,
        "num_input_records": len(records),
        "num_target_records": len(target),
        "num_reference_records": len(reference),
        "num_target_groups": len(groups),
        "num_dev_groups": len(dev_groups),
        "num_test_groups": len(test_groups),
        "num_dev_records": len(dev_records),
        "num_test_records": len(test_records),
        "attr_type_counts_input": _count_by_attr(records),
        "attr_type_counts_target": _count_by_attr(target),
        "attr_type_counts_reference": _count_by_attr(reference),
        "group_split_leakage_count": len(overlap),
        "manifests": manifests,
        "split_rule": {
            "unit": "image_path + object_name + attribute_type + unordered(attribute_name_pos, attribute_name_neg)",
            "rationale": "Keeps reverse pairs and equivalent color flips in the same locked split.",
        },
    }
    write_json(output_dir / "summary.json", summary)

    print(
        f"Wrote {len(dev_records)} dev and {len(test_records)} test {args.attr_type} pairs "
        f"from {len(groups)} groups to {output_dir}"
    )


if __name__ == "__main__":
    main()
