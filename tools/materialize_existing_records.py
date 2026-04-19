#!/usr/bin/env python
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from toposteer.utils import ensure_dir, read_jsonl, write_json, write_jsonl

DEFAULT_REQUIRED_FIELDS = ["image_path", "mask_pos_path"]
OPTIONAL_NEG_FIELDS = ["mask_neg_path"]
PROMPT_FIELDS_FOR_NEG = ["prompt_neg"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Filter a JSONL manifest down to records whose required local files already exist. "
            "Useful for warm_refseg sanity runs when full image download is incomplete."
        )
    )
    parser.add_argument("--input-manifest", required=True)
    parser.add_argument("--output-manifest", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument(
        "--required-path-fields",
        nargs="*",
        default=list(DEFAULT_REQUIRED_FIELDS),
        help="Record fields that must exist on disk for the sample to be kept.",
    )
    parser.add_argument(
        "--strip-missing-neg",
        action="store_true",
        help=(
            "If a negative mask path is present but missing, keep the record and clear the paired negative fields. "
            "Useful when converting a mixed manifest into positive-only warm_refseg data."
        ),
    )
    parser.add_argument(
        "--drop-if-missing-neg",
        action="store_true",
        help="Drop records whose negative mask path is present but missing.",
    )
    parser.add_argument(
        "--keep-splits",
        nargs="*",
        default=None,
        help="Optional split allowlist. Example: --keep-splits train",
    )
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def _exists(path_value: Any) -> bool:
    if not path_value:
        return False
    return Path(path_value).exists()


def _split_name(record: dict[str, Any]) -> str:
    return str(record.get("split", "unknown"))


def _family_name(record: dict[str, Any]) -> str:
    return str(record.get("family", "plain"))


def _clone_and_strip_neg(record: dict[str, Any]) -> dict[str, Any]:
    out = dict(record)
    for key in OPTIONAL_NEG_FIELDS + PROMPT_FIELDS_FOR_NEG:
        if key in out:
            out[key] = None
    return out


def main() -> None:
    args = parse_args()
    records = read_jsonl(args.input_manifest)
    if args.limit is not None:
        records = records[: args.limit]

    required_fields = list(args.required_path_fields)
    kept: list[dict[str, Any]] = []
    missing_reason_counter: Counter[str] = Counter()
    by_split = defaultdict(lambda: {"kept": 0, "dropped": 0})
    by_family = defaultdict(lambda: {"kept": 0, "dropped": 0})
    preview_missing: list[dict[str, Any]] = []
    keep_splits = set(args.keep_splits) if args.keep_splits else None

    for record in records:
        split = _split_name(record)
        family = _family_name(record)

        if keep_splits is not None and split not in keep_splits:
            by_split[split]["dropped"] += 1
            by_family[family]["dropped"] += 1
            missing_reason_counter["filtered_split"] += 1
            continue

        missing_required = [field for field in required_fields if not _exists(record.get(field))]
        if missing_required:
            by_split[split]["dropped"] += 1
            by_family[family]["dropped"] += 1
            for field in missing_required:
                missing_reason_counter[f"missing_required::{field}"] += 1
            if len(preview_missing) < 50:
                preview_missing.append(
                    {
                        "id": record.get("id"),
                        "split": split,
                        "family": family,
                        "reason": "missing_required",
                        "fields": missing_required,
                    }
                )
            continue

        neg_missing = record.get("mask_neg_path") and not _exists(record.get("mask_neg_path"))
        if neg_missing and args.drop_if_missing_neg:
            by_split[split]["dropped"] += 1
            by_family[family]["dropped"] += 1
            missing_reason_counter["missing_optional::mask_neg_path"] += 1
            if len(preview_missing) < 50:
                preview_missing.append(
                    {
                        "id": record.get("id"),
                        "split": split,
                        "family": family,
                        "reason": "missing_optional_neg_drop",
                        "fields": ["mask_neg_path"],
                    }
                )
            continue

        if neg_missing and args.strip_missing_neg:
            record = _clone_and_strip_neg(record)
            missing_reason_counter["stripped_optional::mask_neg_path"] += 1

        kept.append(record)
        by_split[split]["kept"] += 1
        by_family[family]["kept"] += 1

    ensure_dir(Path(args.output_manifest).parent)
    write_jsonl(args.output_manifest, kept)

    summary = {
        "input_manifest": args.input_manifest,
        "output_manifest": args.output_manifest,
        "summary_json": args.summary_json,
        "num_total": len(records),
        "num_kept": len(kept),
        "num_dropped": len(records) - len(kept),
        "required_path_fields": required_fields,
        "keep_splits": sorted(keep_splits) if keep_splits is not None else None,
        "strip_missing_neg": bool(args.strip_missing_neg),
        "drop_if_missing_neg": bool(args.drop_if_missing_neg),
        "missing_reasons": dict(missing_reason_counter),
        "by_split": {k: dict(v) for k, v in sorted(by_split.items())},
        "by_family": {k: dict(v) for k, v in sorted(by_family.items())},
        "preview_missing": preview_missing,
    }
    write_json(args.summary_json, summary)
    print(f"Kept {summary['num_kept']} / {summary['num_total']} records -> {args.output_manifest}")


if __name__ == "__main__":
    main()
