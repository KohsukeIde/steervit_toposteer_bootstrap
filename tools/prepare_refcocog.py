#!/usr/bin/env python
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

from pycocotools.coco import COCO
from tqdm import tqdm

from toposteer.utils.io import ensure_dir, write_json, write_jsonl
from toposteer.utils.mask_ops import decode_coco_segmentation, save_binary_mask


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert Refer-style RefCOCOg annotations to unified manifest format.")
    parser.add_argument("--refs-pkl", type=str, required=True, help="Path to refs(umd).p or refs(google).p")
    parser.add_argument("--instances-json", type=str, required=True, help="COCO instances annotation json with masks")
    parser.add_argument("--images-dir", type=str, required=True, help="COCO image directory")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory for masks and manifest")
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)
    mask_dir = ensure_dir(output_dir / "masks")

    with open(args.refs_pkl, "rb") as f:
        refs = pickle.load(f, encoding="latin1")

    coco = COCO(args.instances_json)
    images_dir = Path(args.images_dir)

    records = []
    for idx, ref in enumerate(tqdm(refs, desc="Preparing RefCOCOg")):
        if args.limit is not None and idx >= args.limit:
            break

        ann_id = ref["ann_id"]
        ann = coco.anns[ann_id]
        img_info = coco.loadImgs([ref["image_id"]])[0]
        image_path = str(images_dir / img_info["file_name"])

        mask = decode_coco_segmentation(ann["segmentation"], height=img_info["height"], width=img_info["width"])
        mask_path = mask_dir / f"ann_{ann_id}.png"
        if not mask_path.exists():
            save_binary_mask(mask, mask_path)

        for sent in ref.get("sentences", []):
            sent_id = sent["sent_id"]
            prompt = sent["sent"]
            record = {
                "id": f"refcocog_{ann_id}_{sent_id}",
                "source": "refcocog",
                "split": ref.get("split", "unknown"),
                "family": "plain",
                "image_path": image_path,
                "mask_pos_path": str(mask_path),
                "prompt_pos": prompt,
                "meta": {
                    "image_id": ref["image_id"],
                    "ann_id": ann_id,
                    "sent_id": sent_id,
                    "ref_id": ref.get("ref_id"),
                    "category_id": ann.get("category_id"),
                },
            }
            records.append(record)

    manifest_path = output_dir / "manifest.jsonl"
    write_jsonl(manifest_path, records)
    write_json(
        output_dir / "summary.json",
        {
            "num_records": len(records),
            "manifest_path": str(manifest_path),
            "images_dir": str(images_dir),
            "refs_pkl": str(args.refs_pkl),
            "instances_json": str(args.instances_json),
        },
    )
    print(f"Wrote {len(records)} records to {manifest_path}")


if __name__ == "__main__":
    main()
