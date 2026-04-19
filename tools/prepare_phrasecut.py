#!/usr/bin/env python
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from tqdm import tqdm

from toposteer.utils.io import ensure_dir, read_json, write_json, write_jsonl
from toposteer.utils.mask_ops import polygons_to_mask, save_binary_mask


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert PhraseCut annotations to unified manifest format.")
    parser.add_argument(
        "--refer-json",
        type=str,
        nargs="+",
        required=True,
        help="One or more refer_train.json / refer_val.json / refer_test.json / refer_miniv.json files.",
    )
    parser.add_argument("--image-meta-json", type=str, required=True, help="Path to image_data_split3000.json")
    parser.add_argument("--images-dir", type=str, required=True, help="Directory containing Visual Genome images")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--allow-missing-images", action="store_true")
    return parser.parse_args()


def infer_family(task: dict) -> str:
    phrase_structure = task.get("phrase_structure", {})
    family = phrase_structure.get("type")
    if family in {"name", "attribute", "relation", "verbose"}:
        return {
            "name": "plain",
            "attribute": "attr",
            "relation": "relation",
            "verbose": "plain",
        }[family]
    if phrase_structure.get("attributes"):
        return "attr"
    if phrase_structure.get("relation_descriptions"):
        return "relation"
    return "plain"


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)
    mask_dir = ensure_dir(output_dir / "masks")

    refer_json_paths = [Path(path) for path in args.refer_json]
    tasks = []
    task_counts_by_file = {}
    for refer_json_path in refer_json_paths:
        file_tasks = read_json(refer_json_path)
        task_counts_by_file[str(refer_json_path)] = len(file_tasks)
        tasks.extend(file_tasks)

    image_meta = read_json(args.image_meta_json)
    image_index = {x["image_id"]: x for x in image_meta}
    images_dir = Path(args.images_dir)

    records = []
    split_counts = Counter()
    family_counts = Counter()
    missing_images = 0
    for idx, task in enumerate(tqdm(tasks, desc="Preparing PhraseCut")):
        if args.limit is not None and idx >= args.limit:
            break

        image_id = task["image_id"]
        info = image_index[image_id]
        width = info["width"]
        height = info["height"]
        image_path = images_dir / f"{image_id}.jpg"
        if not image_path.exists():
            # some VG distributions store files under VG_100K_2 or use png; keep it explicit
            fallback_png = images_dir / f"{image_id}.png"
            if fallback_png.exists():
                image_path = fallback_png
            elif args.allow_missing_images:
                missing_images += 1
            else:
                raise FileNotFoundError(f"Could not find image for PhraseCut image_id={image_id} at {image_path}")

        polygons = task.get("Polygons") or task.get("polygons")
        mask = polygons_to_mask(polygons, height=height, width=width)
        mask_path = mask_dir / f"{task['task_id']}.png"
        save_binary_mask(mask, mask_path)

        phrase_structure = task.get("phrase_structure", {})
        family = infer_family(task)
        split = info.get("split", "unknown")
        record = {
            "id": f"phrasecut_{task['task_id']}",
            "source": "phrasecut",
            "split": split,
            "family": family,
            "image_path": str(image_path),
            "mask_pos_path": str(mask_path),
            "prompt_pos": task["phrase"],
            "meta": {
                "image_id": image_id,
                "task_id": task["task_id"],
                "ann_ids": task.get("ann_ids", []),
                "object_name": phrase_structure.get("name"),
                "attributes": phrase_structure.get("attributes", []),
                "relation_descriptions": phrase_structure.get("relation_descriptions", []),
                "phrase_type": phrase_structure.get("type"),
            },
        }
        records.append(record)
        split_counts[split] += 1
        family_counts[family] += 1

    manifest_path = output_dir / "manifest.jsonl"
    write_jsonl(manifest_path, records)
    write_json(
        output_dir / "summary.json",
        {
            "num_records": len(records),
            "manifest_path": str(manifest_path),
            "refer_json": [str(path) for path in refer_json_paths],
            "task_counts_by_file": task_counts_by_file,
            "split_counts": dict(split_counts),
            "family_counts": dict(family_counts),
            "missing_images": missing_images,
            "allow_missing_images": bool(args.allow_missing_images),
            "image_meta_json": str(args.image_meta_json),
            "images_dir": str(images_dir),
        },
    )
    print(f"Wrote {len(records)} records to {manifest_path}")


if __name__ == "__main__":
    main()
