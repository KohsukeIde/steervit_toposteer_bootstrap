#!/usr/bin/env python
from __future__ import annotations

import argparse
import itertools
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

from tqdm import tqdm

from toposteer.utils.io import read_jsonl, write_json, write_jsonl
from toposteer.utils.mask_ops import load_binary_mask, mask_iou


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a paired hard flip-set manifest from one or more base manifests.")
    parser.add_argument("--input-manifests", nargs="+", required=True)
    parser.add_argument("--output-manifest", required=True)
    parser.add_argument("--max-pairs-per-group", type=int, default=32)
    parser.add_argument("--max-iou", type=float, default=0.5)
    parser.add_argument("--min-mask-area", type=float, default=10.0)
    parser.add_argument("--keep-families", nargs="+", default=["attr", "part", "part_attr", "plain"])
    parser.add_argument("--fallback-image-pairs", action="store_true", help="Also pair same-image records when strict metadata groups are too sparse.")
    parser.add_argument("--max-fallback-pairs", type=int, default=2000)
    parser.add_argument("--max-fallback-pairs-per-group", type=int, default=16)
    parser.add_argument("--skip-strict-pairing", action="store_true", help="Skip metadata grouping and build only same-image fallback pairs.")
    parser.add_argument("--max-pairs-per-family", type=int, default=None, help="Optional cap for each family in --keep-families.")
    parser.add_argument("--interleave-family-output", action="store_true", help="Round-robin output rows by family so --limit reads stay balanced.")
    parser.add_argument(
        "--sort-fallback-groups-by-size",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Process larger same-image groups first when fallback pairing.",
    )
    return parser.parse_args()


@lru_cache(maxsize=200000)
def _load_mask(path: str):
    return load_binary_mask(path)


@lru_cache(maxsize=200000)
def mask_area(path: str) -> float:
    return float(_load_mask(path).sum().item())


def family_key(record: dict) -> tuple:
    family = record.get("family", "plain")
    meta = record.get("meta", {})
    image_path = record["image_path"]

    if family == "attr":
        return (
            family,
            image_path,
            meta.get("object_name"),
            meta.get("attribute_type"),
        )

    if family == "part":
        return (
            family,
            image_path,
            meta.get("parent_obj_ann_id", meta.get("object_name")),
        )

    if family == "part_attr":
        return (
            family,
            image_path,
            meta.get("parent_obj_ann_id", meta.get("object_name")),
            meta.get("attribute_type"),
        )

    return (family, image_path, meta.get("object_name"))


def pair_is_valid(a: dict, b: dict, max_iou: float, min_mask_area: float, strict_metadata: bool = True) -> bool:
    if a["id"] == b["id"]:
        return False
    if a["image_path"] != b["image_path"]:
        return False
    if a.get("prompt_pos") == b.get("prompt_pos"):
        return False
    if mask_area(a["mask_pos_path"]) < min_mask_area or mask_area(b["mask_pos_path"]) < min_mask_area:
        return False

    iou = mask_iou(_load_mask(a["mask_pos_path"]), _load_mask(b["mask_pos_path"]))
    if iou > max_iou:
        return False

    family = a.get("family", "plain")
    meta_a = a.get("meta", {})
    meta_b = b.get("meta", {})

    if family == "attr":
        if meta_a.get("attribute_name") == meta_b.get("attribute_name"):
            return False
        if strict_metadata and meta_a.get("object_name") != meta_b.get("object_name"):
            return False

    if family == "part":
        if meta_a.get("part_name") == meta_b.get("part_name"):
            return False

    if family == "part_attr":
        if (
            meta_a.get("attribute_name") == meta_b.get("attribute_name")
            and meta_a.get("part_name") == meta_b.get("part_name")
        ):
            return False

    return True


def make_pair_record(a: dict, b: dict) -> dict:
    return {
        "id": f"flip_{a['id']}__vs__{b['id']}",
        "source": f"{a.get('source', 'custom')}+flip",
        "split": a.get("split", "custom"),
        "family": a.get("family", "plain"),
        "image_path": a["image_path"],
        "mask_pos_path": a["mask_pos_path"],
        "mask_neg_path": b["mask_pos_path"],
        "prompt_pos": a["prompt_pos"],
        "prompt_neg": b["prompt_pos"],
        "meta": {
            "left_id": a["id"],
            "right_id": b["id"],
            "left_meta": a.get("meta", {}),
            "right_meta": b.get("meta", {}),
        },
    }


def family_cap_reached(family_counts: Counter, family: str, cap: int | None) -> bool:
    return cap is not None and family_counts[family] >= cap


def all_family_caps_reached(family_counts: Counter, families: list[str], cap: int | None) -> bool:
    if cap is None:
        return False
    return all(family_counts[family] >= cap for family in families)


def interleave_by_family(records: list[dict], family_order: list[str]) -> list[dict]:
    buckets = defaultdict(list)
    for record in records:
        buckets[record.get("family", "plain")].append(record)

    ordered = []
    max_len = max((len(bucket) for bucket in buckets.values()), default=0)
    for i in range(max_len):
        for family in family_order:
            if i < len(buckets[family]):
                ordered.append(buckets[family][i])
    return ordered


def main() -> None:
    args = parse_args()
    records = []
    for manifest in args.input_manifests:
        records.extend(read_jsonl(manifest))

    records = [r for r in records if r.get("family", "plain") in set(args.keep_families)]
    groups = defaultdict(list)
    for rec in records:
        groups[family_key(rec)].append(rec)

    pair_records = []
    summary_rows = []
    family_counts = Counter()

    if not args.skip_strict_pairing:
        for key, group in tqdm(groups.items(), desc="Pairing groups"):
            family = key[0]
            if family_cap_reached(family_counts, family, args.max_pairs_per_family):
                continue

            emitted = 0
            for a, b in itertools.permutations(group, 2):
                if emitted >= args.max_pairs_per_group or family_cap_reached(
                    family_counts, family, args.max_pairs_per_family
                ):
                    break
                if pair_is_valid(a, b, max_iou=args.max_iou, min_mask_area=args.min_mask_area):
                    pair_records.append(make_pair_record(a, b))
                    family_counts[family] += 1
                    emitted += 1
            summary_rows.append({"key": str(key), "group_size": len(group), "emitted_pairs": emitted})

    if (
        args.fallback_image_pairs
        and len(pair_records) < args.max_fallback_pairs
        and not all_family_caps_reached(family_counts, args.keep_families, args.max_pairs_per_family)
    ):
        seen_ids = {r["id"] for r in pair_records}
        image_groups = defaultdict(list)
        for rec in records:
            image_groups[(rec.get("family", "plain"), rec["image_path"])].append(rec)

        groups_by_family = defaultdict(list)
        for key, group in image_groups.items():
            groups_by_family[key[0]].append((key, group))

        for family in args.keep_families:
            family_groups = groups_by_family.get(family, [])
            if args.sort_fallback_groups_by_size:
                family_groups = sorted(family_groups, key=lambda item: len(item[1]), reverse=True)

            for key, group in tqdm(family_groups, desc=f"Fallback same-image pairing ({family})"):
                if len(pair_records) >= args.max_fallback_pairs or family_cap_reached(
                    family_counts, family, args.max_pairs_per_family
                ):
                    break

                emitted = 0
                for a, b in itertools.permutations(group, 2):
                    if (
                        emitted >= args.max_fallback_pairs_per_group
                        or len(pair_records) >= args.max_fallback_pairs
                        or family_cap_reached(family_counts, family, args.max_pairs_per_family)
                    ):
                        break
                    pair_id = f"flip_{a['id']}__vs__{b['id']}"
                    if pair_id in seen_ids:
                        continue
                    if pair_is_valid(
                        a,
                        b,
                        max_iou=args.max_iou,
                        min_mask_area=args.min_mask_area,
                        strict_metadata=False,
                    ):
                        pair_records.append(make_pair_record(a, b))
                        seen_ids.add(pair_id)
                        family_counts[family] += 1
                        emitted += 1
                if emitted:
                    summary_rows.append({"key": f"fallback:{key}", "group_size": len(group), "emitted_pairs": emitted})

    if args.interleave_family_output:
        pair_records = interleave_by_family(pair_records, args.keep_families)

    write_jsonl(args.output_manifest, pair_records)
    write_json(
        str(Path(args.output_manifest).with_suffix(".summary.json")),
        {
            "num_pairs": len(pair_records),
            "pairs_by_family": dict(Counter(r.get("family", "plain") for r in pair_records)),
            "groups": summary_rows[:5000],
            "input_manifests": args.input_manifests,
        },
    )
    print(f"Wrote {len(pair_records)} paired records to {args.output_manifest}")


if __name__ == "__main__":
    main()
