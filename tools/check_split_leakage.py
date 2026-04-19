#!/usr/bin/env python
from __future__ import annotations

import argparse
from collections import Counter
from typing import Any

from toposteer.utils import read_jsonl, write_json

DEFAULT_KEY_PRIORITY = ["meta.image_id", "image_id", "image_path", "id"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Check train/eval leakage between two manifests. By default the script tries meta.image_id, "
            "then image_id, then image_path, then id."
        )
    )
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--eval-manifest", required=True)
    parser.add_argument("--summary-json", default=None)
    parser.add_argument("--key-priority", nargs="*", default=list(DEFAULT_KEY_PRIORITY))
    parser.add_argument(
        "--ignore-overlap",
        action="store_true",
        help="Report overlap in JSON/stdout but exit successfully instead of raising an error.",
    )
    parser.add_argument("--max-preview", type=int, default=20)
    return parser.parse_args()


def _nested_get(record: dict[str, Any], dotted_key: str) -> Any:
    cursor: Any = record
    for part in dotted_key.split("."):
        if not isinstance(cursor, dict) or part not in cursor:
            return None
        cursor = cursor[part]
    return cursor


def _resolve_key(record: dict[str, Any], key_priority: list[str]) -> str | None:
    for key in key_priority:
        value = _nested_get(record, key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return f"{key}::{text}"
    return None


def _collect_keys(records: list[dict[str, Any]], key_priority: list[str]) -> tuple[dict[str, list[str]], int]:
    bucket: dict[str, list[str]] = {}
    missing = 0
    for record in records:
        resolved = _resolve_key(record, key_priority)
        if resolved is None:
            missing += 1
            continue
        bucket.setdefault(resolved, []).append(str(record.get("id", "<no-id>")))
    return bucket, missing


def main() -> None:
    args = parse_args()
    train_records = read_jsonl(args.train_manifest)
    eval_records = read_jsonl(args.eval_manifest)

    train_keys, train_missing = _collect_keys(train_records, args.key_priority)
    eval_keys, eval_missing = _collect_keys(eval_records, args.key_priority)

    overlap_keys = sorted(set(train_keys.keys()) & set(eval_keys.keys()))
    overlap_preview = []
    overlap_by_prefix = Counter()
    for key in overlap_keys[: args.max_preview]:
        prefix = key.split("::", 1)[0]
        overlap_by_prefix[prefix] += 1
        overlap_preview.append(
            {
                "key": key,
                "train_ids": train_keys[key][: args.max_preview],
                "eval_ids": eval_keys[key][: args.max_preview],
            }
        )
    for key in overlap_keys[args.max_preview :]:
        prefix = key.split("::", 1)[0]
        overlap_by_prefix[prefix] += 1

    summary = {
        "train_manifest": args.train_manifest,
        "eval_manifest": args.eval_manifest,
        "key_priority": args.key_priority,
        "train_num_records": len(train_records),
        "eval_num_records": len(eval_records),
        "train_missing_keys": train_missing,
        "eval_missing_keys": eval_missing,
        "train_num_unique_keys": len(train_keys),
        "eval_num_unique_keys": len(eval_keys),
        "overlap_count": len(overlap_keys),
        "overlap_by_key_prefix": dict(overlap_by_prefix),
        "overlap_preview": overlap_preview,
    }

    if args.summary_json is not None:
        write_json(args.summary_json, summary)

    print(
        f"train={summary['train_num_unique_keys']} eval={summary['eval_num_unique_keys']} "
        f"overlap={summary['overlap_count']}"
    )

    if overlap_keys and not args.ignore_overlap:
        raise SystemExit(
            "Leakage detected between train and eval manifests. "
            f"See {args.summary_json or 'stdout'} for overlap details."
        )


if __name__ == "__main__":
    main()
