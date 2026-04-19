#!/usr/bin/env python
from __future__ import annotations

import argparse
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import requests
from tqdm import tqdm

from toposteer.utils import ensure_dir, read_json, read_jsonl, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download only the PhraseCut images referenced by one or more manifests.")
    parser.add_argument("--manifests", nargs="+", required=True)
    parser.add_argument("--image-meta-json", required=True)
    parser.add_argument("--images-dir", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--limit-images", type=int, default=None)
    parser.add_argument("--timeout", type=float, default=120.0)
    return parser.parse_args()


def _nested_get(record: dict[str, Any], dotted_key: str) -> Any:
    cursor: Any = record
    for part in dotted_key.split("."):
        if not isinstance(cursor, dict) or part not in cursor:
            return None
        cursor = cursor[part]
    return cursor


def _image_id_from_record(record: dict[str, Any]) -> str | None:
    for key in ("meta.image_id", "image_id", "meta.left_meta.image_id", "meta.right_meta.image_id"):
        value = _nested_get(record, key)
        if value is not None:
            return str(value)
    image_path = record.get("image_path")
    if image_path:
        stem = Path(str(image_path)).stem
        if re.fullmatch(r"\d+", stem):
            return stem
    return None


def _download_one(item: dict[str, Any], images_dir: Path, skip_existing: bool, timeout: float) -> tuple[str, str | None]:
    image_id = str(item["image_id"])
    out_path = images_dir / f"{image_id}.jpg"
    if skip_existing and out_path.exists() and out_path.stat().st_size > 0:
        return image_id, None

    url = item.get("url")
    if not url:
        return image_id, "missing_url"

    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    out_path.write_bytes(response.content)
    return image_id, None


def main() -> None:
    args = parse_args()
    images_dir = ensure_dir(args.images_dir)
    image_meta = read_json(args.image_meta_json)
    image_index = {str(item["image_id"]): item for item in image_meta}

    requested_ids: list[str] = []
    manifest_counts: dict[str, int] = {}
    missing_key_records = 0
    for manifest in args.manifests:
        records = read_jsonl(manifest)
        manifest_counts[manifest] = len(records)
        for record in records:
            image_id = _image_id_from_record(record)
            if image_id is None:
                missing_key_records += 1
                continue
            requested_ids.append(image_id)

    unique_ids = sorted(set(requested_ids), key=lambda x: int(x) if x.isdigit() else x)
    if args.limit_images is not None:
        unique_ids = unique_ids[: args.limit_images]

    missing_meta = [image_id for image_id in unique_ids if image_id not in image_index]
    items = [image_index[image_id] for image_id in unique_ids if image_id in image_index]

    failures: list[dict[str, str]] = []
    with ThreadPoolExecutor(max_workers=max(args.workers, 1)) as pool:
        futures = [pool.submit(_download_one, item, images_dir, args.skip_existing, args.timeout) for item in items]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Downloading manifest images"):
            try:
                image_id, error = future.result()
            except Exception as exc:
                failures.append({"image_id": "<unknown>", "error": str(exc)})
                continue
            if error:
                failures.append({"image_id": image_id, "error": error})

    existing_after = sum(1 for image_id in unique_ids if (images_dir / f"{image_id}.jpg").exists())
    split_counts = Counter(str(item.get("split", "unknown")) for item in items)
    summary = {
        "manifests": args.manifests,
        "manifest_counts": manifest_counts,
        "image_meta_json": args.image_meta_json,
        "images_dir": str(images_dir),
        "num_requested_refs": len(requested_ids),
        "num_unique_requested_images": len(unique_ids),
        "num_missing_record_image_keys": missing_key_records,
        "num_missing_image_meta": len(missing_meta),
        "missing_image_meta_preview": missing_meta[:50],
        "split_counts": dict(split_counts),
        "num_download_items": len(items),
        "num_existing_after": existing_after,
        "num_failures": len(failures),
        "failures_preview": failures[:100],
    }
    write_json(args.summary_json, summary)

    print(
        f"Images available: {existing_after}/{len(unique_ids)} "
        f"for {len(args.manifests)} manifests. Failures: {len(failures)}"
    )
    if failures:
        raise SystemExit(f"Failed to download {len(failures)} images. See {args.summary_json}")


if __name__ == "__main__":
    main()
