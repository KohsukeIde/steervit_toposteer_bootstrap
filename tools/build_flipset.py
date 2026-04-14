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
    parser.add_argument("--max-pairs-per-control-family", type=int, default=None, help="Optional cap for each controlled pair subtype.")
    parser.add_argument("--keep-control-families", nargs="+", default=None, help="Only keep pairs with these controlled pair subtypes.")
    parser.add_argument("--use-control-family-as-family", action="store_true", help="Write controlled subtype into the manifest family field.")
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


def _meta_value(meta: dict, key: str):
    value = meta.get(key)
    return value if value not in {"", None} else None


def _attribute_value(meta: dict):
    attr_name = _meta_value(meta, "attribute_name")
    if attr_name is not None:
        return attr_name
    attrs = meta.get("attributes") or []
    attrs = tuple(sorted(x for x in attrs if x))
    return attrs or None


def _same(meta_a: dict, meta_b: dict, key: str) -> bool:
    a_val = _meta_value(meta_a, key)
    b_val = _meta_value(meta_b, key)
    return a_val is not None and b_val is not None and a_val == b_val


def pair_profile(a: dict, b: dict, pair_origin: str, iou: float | None = None) -> dict:
    meta_a = a.get("meta", {})
    meta_b = b.get("meta", {})
    fields = {
        "object_name": "same_object_name",
        "parent_obj_ann_id": "same_parent",
        "part_name": "same_part",
        "attribute_type": "same_attribute_type",
    }

    changed_fields = []
    same_flags = {}
    for field, same_key in fields.items():
        a_val = _meta_value(meta_a, field)
        b_val = _meta_value(meta_b, field)
        same_flags[same_key] = a_val is not None and b_val is not None and a_val == b_val
        if (a_val is not None or b_val is not None) and a_val != b_val:
            changed_fields.append(field)

    attr_a = _attribute_value(meta_a)
    attr_b = _attribute_value(meta_b)
    same_flags["same_attribute_name"] = attr_a is not None and attr_b is not None and attr_a == attr_b
    if (attr_a is not None or attr_b is not None) and attr_a != attr_b:
        changed_fields.append("attribute_name")

    attr_type_a = _meta_value(meta_a, "attribute_type")
    attr_type_b = _meta_value(meta_b, "attribute_type")
    attribute_type_compatible = same_flags["same_attribute_type"] or (attr_type_a is None and attr_type_b is None)

    family = a.get("family", "plain")
    control_family = f"{family}_uncontrolled"
    if family in {"attr", "plain", "relation"}:
        if (
            same_flags["same_object_name"]
            and attribute_type_compatible
            and "attribute_name" in changed_fields
            and "part_name" not in changed_fields
        ):
            control_family = "attr_same_object_diff_attr"

    elif family == "part":
        if same_flags["same_parent"] and "part_name" in changed_fields:
            control_family = "part_same_parent_diff_part"
        elif same_flags["same_object_name"] and "part_name" in changed_fields:
            control_family = "part_same_object_diff_part"

    elif family == "part_attr":
        if (
            same_flags["same_object_name"]
            and same_flags["same_part"]
            and attribute_type_compatible
            and "attribute_name" in changed_fields
        ):
            control_family = "part_attr_same_part_diff_attr"
        elif (
            same_flags["same_object_name"]
            and same_flags["same_attribute_name"]
            and attribute_type_compatible
            and "part_name" in changed_fields
        ):
            control_family = "part_attr_same_attr_diff_part"
        elif (
            same_flags["same_object_name"]
            and "part_name" in changed_fields
            and "attribute_name" in changed_fields
        ):
            control_family = "part_attr_diff_part_diff_attr"

    return {
        "pair_origin": pair_origin,
        "control_family": control_family,
        **same_flags,
        "attribute_type_compatible": attribute_type_compatible,
        "changed_fields": changed_fields,
        "num_changed_fields": len(changed_fields),
        "mask_iou": iou,
    }


def pair_is_valid(
    a: dict,
    b: dict,
    max_iou: float,
    min_mask_area: float,
    strict_metadata: bool = True,
    keep_control_families: set[str] | None = None,
) -> tuple[bool, float | None]:
    if a["id"] == b["id"]:
        return False, None
    if a["image_path"] != b["image_path"]:
        return False, None
    if a.get("prompt_pos") == b.get("prompt_pos"):
        return False, None
    if mask_area(a["mask_pos_path"]) < min_mask_area or mask_area(b["mask_pos_path"]) < min_mask_area:
        return False, None

    iou = mask_iou(_load_mask(a["mask_pos_path"]), _load_mask(b["mask_pos_path"]))
    if iou > max_iou:
        return False, iou

    family = a.get("family", "plain")
    meta_a = a.get("meta", {})
    meta_b = b.get("meta", {})

    if family == "attr":
        if _attribute_value(meta_a) == _attribute_value(meta_b):
            return False, iou
        if strict_metadata and meta_a.get("object_name") != meta_b.get("object_name"):
            return False, iou

    if family == "part":
        if meta_a.get("part_name") == meta_b.get("part_name"):
            return False, iou

    if family == "part_attr":
        if (
            _attribute_value(meta_a) == _attribute_value(meta_b)
            and meta_a.get("part_name") == meta_b.get("part_name")
        ):
            return False, iou

    if keep_control_families is not None:
        profile = pair_profile(a, b, pair_origin="candidate", iou=iou)
        if profile["control_family"] not in keep_control_families:
            return False, iou

    return True, iou


def make_pair_record(a: dict, b: dict, pair_origin: str, iou: float | None = None, use_control_family_as_family: bool = False) -> dict:
    profile = pair_profile(a, b, pair_origin=pair_origin, iou=iou)
    family = profile["control_family"] if use_control_family_as_family else a.get("family", "plain")
    return {
        "id": f"flip_{a['id']}__vs__{b['id']}",
        "source": f"{a.get('source', 'custom')}+flip",
        "split": a.get("split", "custom"),
        "family": family,
        "base_family": a.get("family", "plain"),
        **profile,
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


def pair_caps_reached(family_counts: Counter, control_counts: Counter, pair_record: dict, args: argparse.Namespace) -> bool:
    if family_cap_reached(family_counts, pair_record.get("base_family", pair_record.get("family", "plain")), args.max_pairs_per_family):
        return True
    if args.max_pairs_per_control_family is not None:
        return control_counts[pair_record["control_family"]] >= args.max_pairs_per_control_family
    return False


def count_pair(family_counts: Counter, control_counts: Counter, pair_record: dict) -> None:
    family_counts[pair_record.get("base_family", pair_record.get("family", "plain"))] += 1
    control_counts[pair_record["control_family"]] += 1


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
    control_counts = Counter()
    keep_control_families = set(args.keep_control_families) if args.keep_control_families else None

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
                valid, iou = pair_is_valid(
                    a,
                    b,
                    max_iou=args.max_iou,
                    min_mask_area=args.min_mask_area,
                    keep_control_families=keep_control_families,
                )
                if valid:
                    pair_record = make_pair_record(
                        a,
                        b,
                        pair_origin="strict",
                        iou=iou,
                        use_control_family_as_family=args.use_control_family_as_family,
                    )
                    if pair_caps_reached(family_counts, control_counts, pair_record, args):
                        continue
                    pair_records.append(pair_record)
                    count_pair(family_counts, control_counts, pair_record)
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
                    valid, iou = pair_is_valid(
                        a,
                        b,
                        max_iou=args.max_iou,
                        min_mask_area=args.min_mask_area,
                        strict_metadata=False,
                        keep_control_families=keep_control_families,
                    )
                    if valid:
                        pair_record = make_pair_record(
                            a,
                            b,
                            pair_origin="fallback",
                            iou=iou,
                            use_control_family_as_family=args.use_control_family_as_family,
                        )
                        if pair_caps_reached(family_counts, control_counts, pair_record, args):
                            continue
                        pair_records.append(pair_record)
                        seen_ids.add(pair_id)
                        count_pair(family_counts, control_counts, pair_record)
                        emitted += 1
                if emitted:
                    summary_rows.append({"key": f"fallback:{key}", "group_size": len(group), "emitted_pairs": emitted})

    if args.interleave_family_output:
        order = args.keep_families
        if args.use_control_family_as_family and args.keep_control_families:
            order = args.keep_control_families
        pair_records = interleave_by_family(pair_records, order)

    write_jsonl(args.output_manifest, pair_records)
    write_json(
        str(Path(args.output_manifest).with_suffix(".summary.json")),
        {
            "num_pairs": len(pair_records),
            "pairs_by_family": dict(Counter(r.get("family", "plain") for r in pair_records)),
            "pairs_by_base_family": dict(Counter(r.get("base_family", r.get("family", "plain")) for r in pair_records)),
            "pairs_by_pair_origin": dict(Counter(r.get("pair_origin", "unknown") for r in pair_records)),
            "pairs_by_control_family": dict(Counter(r.get("control_family", "unknown") for r in pair_records)),
            "pairs_by_num_changed_fields": {
                str(k): v for k, v in Counter(r.get("num_changed_fields", -1) for r in pair_records).items()
            },
            "pairs_by_changed_fields": dict(Counter(",".join(r.get("changed_fields", [])) for r in pair_records)),
            "groups": summary_rows[:5000],
            "input_manifests": args.input_manifests,
        },
    )
    print(f"Wrote {len(pair_records)} paired records to {args.output_manifest}")


if __name__ == "__main__":
    main()
