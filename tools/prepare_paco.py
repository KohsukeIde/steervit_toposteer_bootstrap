#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from tqdm import tqdm

from toposteer.utils.io import ensure_dir, read_json, write_json, write_jsonl
from toposteer.utils.mask_ops import decode_coco_segmentation, save_binary_mask


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert PACO-LVIS style COCO JSON into unified manifest format.")
    parser.add_argument("--annotations-json", type=str, required=True)
    parser.add_argument("--images-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--include-part-attributes", action="store_true")
    return parser.parse_args()


def normalize_name(name: str) -> str:
    return name.replace("_", " ").replace("(automobile)", "").replace("(computer_equipment)", "").replace("  ", " ").strip()


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)
    mask_dir = ensure_dir(output_dir / "masks")
    split_name = "val" if "val" in Path(args.annotations_json).stem else "train"

    data = read_json(args.annotations_json)
    images_dir = Path(args.images_dir)

    image_by_id = {x["id"]: x for x in data["images"]}
    obj_cat_by_id = {x["id"]: x for x in data.get("categories", [])}
    part_cat_by_id = {x["id"]: x for x in data.get("part_categories", [])}
    attr_by_id = {x["id"]: x for x in data.get("attributes", [])}
    ann_by_id = {x["id"]: x for x in data.get("annotations", [])}
    attr_type_map = {k: set(v) for k, v in data.get("attr_type_to_attr_idxs", {}).items()}

    records = []

    for idx, ann in enumerate(tqdm(data.get("annotations", []), desc="Preparing PACO")):
        if args.limit is not None and idx >= args.limit:
            break

        image_info = image_by_id[ann["image_id"]]
        image_path = images_dir / image_info["file_name"]
        if not image_path.exists():
            raise FileNotFoundError(f"Missing image file: {image_path}")

        mask_path = mask_dir / f"ann_{ann['id']}.png"
        if not mask_path.exists():
            mask = decode_coco_segmentation(ann["segmentation"], height=image_info["height"], width=image_info["width"])
            save_binary_mask(mask, mask_path)

        category_id = ann["category_id"]
        is_part = category_id in part_cat_by_id
        cat_name = normalize_name(
            (part_cat_by_id.get(category_id) or obj_cat_by_id.get(category_id) or {"name": "unknown"})["name"]
        )

        parent_obj_name = None
        parent_obj_ann_id = ann.get("obj_ann_id")
        if is_part and parent_obj_ann_id in ann_by_id:
            parent_obj = ann_by_id[parent_obj_ann_id]
            parent_obj_name = normalize_name(obj_cat_by_id.get(parent_obj["category_id"], {"name": "object"})["name"])

        # plain part records
        if is_part and parent_obj_name is not None:
            records.append(
                {
                    "id": f"paco_part_{ann['id']}",
                    "source": "paco",
                    "split": split_name,
                    "family": "part",
                    "image_path": str(image_path),
                    "mask_pos_path": str(mask_path),
                    "prompt_pos": f"{parent_obj_name} {cat_name}",
                    "meta": {
                        "image_id": ann["image_id"],
                        "ann_id": ann["id"],
                        "parent_obj_ann_id": parent_obj_ann_id,
                        "object_name": parent_obj_name,
                        "part_name": cat_name,
                        "is_part": True,
                    },
                }
            )

        # object attribute records
        attribute_ids = ann.get("attribute_ids", []) or []
        if attribute_ids:
            object_name = parent_obj_name if is_part else normalize_name(obj_cat_by_id.get(category_id, {"name": "object"})["name"])
            for attr_id in attribute_ids:
                attr_name = normalize_name(attr_by_id[attr_id]["name"])
                attr_type = None
                for k, attr_ids in attr_type_map.items():
                    if attr_id in attr_ids:
                        attr_type = k
                        break

                if is_part and not args.include_part_attributes:
                    continue

                if is_part and parent_obj_name is not None:
                    prompt = f"{attr_name} {parent_obj_name} {cat_name}"
                    family = "part_attr"
                else:
                    prompt = f"{attr_name} {object_name}"
                    family = "attr"

                records.append(
                    {
                        "id": f"paco_attr_{ann['id']}_{attr_id}",
                        "source": "paco",
                        "split": split_name,
                        "family": family,
                        "image_path": str(image_path),
                        "mask_pos_path": str(mask_path),
                        "prompt_pos": prompt,
                        "meta": {
                            "image_id": ann["image_id"],
                            "ann_id": ann["id"],
                            "object_name": object_name,
                            "part_name": cat_name if is_part else None,
                            "attribute_name": attr_name,
                            "attribute_type": attr_type,
                            "is_part": is_part,
                            "parent_obj_ann_id": parent_obj_ann_id,
                        },
                    }
                )

    manifest_path = output_dir / "manifest.jsonl"
    write_jsonl(manifest_path, records)
    write_json(
        output_dir / "summary.json",
        {
            "num_records": len(records),
            "manifest_path": str(manifest_path),
            "annotations_json": str(args.annotations_json),
            "images_dir": str(images_dir),
        },
    )
    print(f"Wrote {len(records)} records to {manifest_path}")


if __name__ == "__main__":
    main()
