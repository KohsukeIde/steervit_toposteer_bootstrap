#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.request import urlretrieve

import gdown
import requests
from tqdm import tqdm

from toposteer.utils.io import ensure_dir, write_json


PHRASECUT_DRIVE_IDS = {
    "image_data_split": "1xB9eExJo35K3DQu8PnBWZ1q_OMoMl8-i",
    "refer_miniv": "1oLcTQ1blTIQyu5ZMelQN2HSniuQaxM4E",
    "refer_input_miniv": "1QPZ35eSLRRczM4OMjlkiI88m7JGS5rsN",
}
PACO_LVIS_URL = "https://dl.fbaipublicfiles.com/paco/annotations/paco_lvis_v1.zip"
PACO_LVIS_SHA256 = "02ac4edb22c251e07853e6231d69aec3fad0a180f03de2f8c880650322debc80"
COCO_VAL2017_URL = "http://images.cocodataset.org/zips/val2017.zip"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download small raw datasets for TopoSteer diagnostics.")
    parser.add_argument("--output-root", default="data/raw")
    parser.add_argument("--skip-existing", action="store_true", default=True)
    parser.add_argument("--phrasecut-miniv", action="store_true", default=True)
    parser.add_argument("--paco-lvis-val", action="store_true", default=True)
    parser.add_argument("--coco-val2017", action="store_true", default=True)
    parser.add_argument("--download-paco-val-images", action="store_true", default=True)
    parser.add_argument("--image-workers", type=int, default=16)
    return parser.parse_args()


def sha256sum(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def download_url(url: str, path: Path, skip_existing: bool = True) -> None:
    ensure_dir(path.parent)
    if skip_existing and path.exists() and path.stat().st_size > 0:
        print(f"Already exists: {path}")
        return
    urlretrieve(url, path)


def unzip_file(path: Path, output_dir: Path, marker: Path | None = None) -> None:
    if marker is not None and marker.exists():
        print(f"Already extracted: {marker}")
        return
    ensure_dir(output_dir)
    with zipfile.ZipFile(path) as zf:
        zf.extractall(output_dir)


def gdown_file(file_id: str, path: Path, skip_existing: bool = True) -> None:
    ensure_dir(path.parent)
    if skip_existing and path.exists() and path.stat().st_size > 0:
        print(f"Already exists: {path}")
        return
    gdown.download(id=file_id, output=str(path), quiet=False)


def download_phrasecut_miniv(root: Path, skip_existing: bool) -> dict:
    phrasecut_root = ensure_dir(root / "phrasecut" / "VGPhraseCut_v0")
    images_dir = ensure_dir(phrasecut_root / "images")
    image_meta_path = phrasecut_root / "image_data_split.json"
    refer_path = phrasecut_root / "refer_miniv.json"

    gdown_file(PHRASECUT_DRIVE_IDS["image_data_split"], image_meta_path, skip_existing=skip_existing)
    gdown_file(PHRASECUT_DRIVE_IDS["refer_miniv"], refer_path, skip_existing=skip_existing)
    gdown_file(PHRASECUT_DRIVE_IDS["refer_input_miniv"], phrasecut_root / "refer_input_miniv.json", skip_existing=skip_existing)

    with image_meta_path.open("r", encoding="utf-8") as f:
        image_meta = json.load(f)
    miniv_images = [x for x in image_meta if x.get("split") == "miniv"]

    for item in tqdm(miniv_images, desc="Downloading PhraseCut miniv images"):
        image_id = item["image_id"]
        out_path = images_dir / f"{image_id}.jpg"
        if skip_existing and out_path.exists() and out_path.stat().st_size > 0:
            continue
        response = requests.get(item["url"], timeout=60)
        response.raise_for_status()
        out_path.write_bytes(response.content)

    return {
        "refer_json": str(refer_path),
        "image_meta_json": str(image_meta_path),
        "images_dir": str(images_dir),
        "num_images": len(miniv_images),
    }


def download_paco_lvis(root: Path, skip_existing: bool) -> dict:
    paco_root = ensure_dir(root / "paco")
    zip_path = paco_root / "paco_lvis_v1.zip"
    download_url(PACO_LVIS_URL, zip_path, skip_existing=skip_existing)
    digest = sha256sum(zip_path)
    if digest != PACO_LVIS_SHA256:
        raise RuntimeError(f"PACO checksum mismatch for {zip_path}: {digest} != {PACO_LVIS_SHA256}")
    unzip_file(zip_path, paco_root, marker=paco_root / "paco_lvis_v1_val.json")
    return {
        "annotations_dir": str(paco_root),
        "val_annotations_json": str(paco_root / "paco_lvis_v1_val.json"),
        "zip_path": str(zip_path),
    }


def _download_image(item: dict, images_dir: Path, skip_existing: bool) -> str:
    rel_path = item["file_name"]
    out_path = images_dir / rel_path
    if skip_existing and out_path.exists() and out_path.stat().st_size > 0:
        return str(out_path)
    ensure_dir(out_path.parent)
    response = requests.get(item["coco_url"], timeout=120)
    response.raise_for_status()
    out_path.write_bytes(response.content)
    return str(out_path)


def download_paco_val_images(root: Path, annotations_json: str | Path, skip_existing: bool, workers: int) -> dict:
    images_dir = ensure_dir(root / "coco" / "paco_lvis_val_images")
    with Path(annotations_json).open("r", encoding="utf-8") as f:
        data = json.load(f)

    images = [x for x in data["images"] if x.get("coco_url") and x.get("file_name")]
    failures = []
    with ThreadPoolExecutor(max_workers=max(workers, 1)) as pool:
        futures = [pool.submit(_download_image, item, images_dir, skip_existing) for item in images]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Downloading PACO-LVIS val images"):
            try:
                future.result()
            except Exception as exc:  # keep going so we can report all flaky URLs
                failures.append(str(exc))

    if failures:
        raise RuntimeError(f"Failed to download {len(failures)} PACO images. First failure: {failures[0]}")

    return {
        "images_dir": str(images_dir),
        "num_images": len(images),
    }


def download_coco_val2017(root: Path, skip_existing: bool) -> dict:
    coco_root = ensure_dir(root / "coco")
    zip_path = coco_root / "val2017.zip"
    download_url(COCO_VAL2017_URL, zip_path, skip_existing=skip_existing)
    unzip_file(zip_path, coco_root, marker=coco_root / "val2017")
    return {
        "val2017_dir": str(coco_root / "val2017"),
        "zip_path": str(zip_path),
    }


def main() -> None:
    args = parse_args()
    root = ensure_dir(args.output_root)
    summary = {}
    if args.phrasecut_miniv:
        summary["phrasecut_miniv"] = download_phrasecut_miniv(root, skip_existing=args.skip_existing)
    if args.paco_lvis_val:
        summary["paco_lvis"] = download_paco_lvis(root, skip_existing=args.skip_existing)
        if args.download_paco_val_images:
            summary["paco_lvis_val_images"] = download_paco_val_images(
                root,
                summary["paco_lvis"]["val_annotations_json"],
                skip_existing=args.skip_existing,
                workers=args.image_workers,
            )
    if args.coco_val2017:
        summary["coco_val2017"] = download_coco_val2017(root, skip_existing=args.skip_existing)
    write_json(root / "diagnostic_data_summary.json", summary)
    print(f"Wrote diagnostic data summary to {root / 'diagnostic_data_summary.json'}")


if __name__ == "__main__":
    main()
