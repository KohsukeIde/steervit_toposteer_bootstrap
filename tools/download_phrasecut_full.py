#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import gdown
import requests
from tqdm import tqdm

from toposteer.utils.io import ensure_dir, write_json


PHRASECUT_DRIVE_IDS = {
    "image_data_split": "1xB9eExJo35K3DQu8PnBWZ1q_OMoMl8-i",
    "refer_miniv": "1oLcTQ1blTIQyu5ZMelQN2HSniuQaxM4E",
    "refer_test": "1jrzXm1gcq6f5hNDeamZd0UmyyHUv61IZ",
    "refer_train": "1qx-0q6r9r0YUGpoyT0B8HJKmUFWQDSu7",
    "refer_val": "1UyojArOFPlsSeNbA9fHWjCjOOU-OCohG",
    "refer_input_miniv": "1QPZ35eSLRRczM4OMjlkiI88m7JGS5rsN",
    "refer_input_test": "1xfr3MKPSSPUfIf3i_LQA3JAEdbCRyMiD",
    "refer_input_train": "1udrL3DM6Ksml7jXGRY8PSPSCk5sGIsGd",
    "refer_input_val": "1DjJRoTdJGpee8k4QKfjhV97XQLjuiLk0",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download PhraseCut full annotations and images.")
    parser.add_argument("--output-root", default="data/raw/phrasecut/VGPhraseCut_v0")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test", "miniv"], choices=["train", "val", "test", "miniv"])
    parser.add_argument("--image-workers", type=int, default=16)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--download-input-refer", action="store_true", default=True)
    parser.add_argument("--no-download-input-refer", dest="download_input_refer", action="store_false")
    return parser.parse_args()


def gdown_file(file_id: str, path: Path, skip_existing: bool) -> bool:
    ensure_dir(path.parent)
    if skip_existing and path.exists() and path.stat().st_size > 0:
        print(f"Already exists: {path}")
        return False
    gdown.download(id=file_id, output=str(path), quiet=False)
    if not path.exists() or path.stat().st_size == 0:
        raise RuntimeError(f"Failed to download Google Drive file to {path}")
    return True


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def download_annotations(root: Path, splits: list[str], skip_existing: bool, download_input_refer: bool) -> dict[str, Any]:
    downloads: dict[str, str] = {}
    image_meta_path = root / "image_data_split.json"
    gdown_file(PHRASECUT_DRIVE_IDS["image_data_split"], image_meta_path, skip_existing=skip_existing)
    downloads["image_data_split"] = str(image_meta_path)

    for split in splits:
        refer_path = root / f"refer_{split}.json"
        gdown_file(PHRASECUT_DRIVE_IDS[f"refer_{split}"], refer_path, skip_existing=skip_existing)
        downloads[f"refer_{split}"] = str(refer_path)

        if download_input_refer:
            refer_input_path = root / f"refer_input_{split}.json"
            gdown_file(PHRASECUT_DRIVE_IDS[f"refer_input_{split}"], refer_input_path, skip_existing=skip_existing)
            downloads[f"refer_input_{split}"] = str(refer_input_path)

    annotation_counts = {}
    for split in splits:
        annotation_counts[split] = len(load_json(root / f"refer_{split}.json"))

    return {
        "downloads": downloads,
        "annotation_counts": annotation_counts,
    }


def _download_image(item: dict[str, Any], images_dir: Path, skip_existing: bool) -> tuple[str, str | None]:
    image_id = item["image_id"]
    url = item.get("url")
    if not url:
        return str(image_id), "missing_url"

    out_path = images_dir / f"{image_id}.jpg"
    if skip_existing and out_path.exists() and out_path.stat().st_size > 0:
        return str(out_path), None

    response = requests.get(url, timeout=120)
    response.raise_for_status()
    out_path.write_bytes(response.content)
    return str(out_path), None


def download_images(root: Path, splits: list[str], skip_existing: bool, workers: int) -> dict[str, Any]:
    image_meta = load_json(root / "image_data_split.json")
    split_set = set(splits)
    images = [item for item in image_meta if item.get("split") in split_set]
    images_dir = ensure_dir(root / "images")

    failures: list[dict[str, str]] = []
    with ThreadPoolExecutor(max_workers=max(workers, 1)) as pool:
        futures = [pool.submit(_download_image, item, images_dir, skip_existing) for item in images]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Downloading PhraseCut images"):
            try:
                _, error = future.result()
                if error:
                    failures.append({"error": error})
            except Exception as exc:
                failures.append({"error": str(exc)})

    if failures:
        write_json(root / "download_phrasecut_full_failures.json", failures[:1000])
        raise RuntimeError(f"Failed to download {len(failures)} PhraseCut images. See download_phrasecut_full_failures.json")

    expected_by_split = Counter(item.get("split", "unknown") for item in images)
    existing_by_split = Counter()
    for item in images:
        if (images_dir / f"{item['image_id']}.jpg").exists():
            existing_by_split[item.get("split", "unknown")] += 1

    return {
        "images_dir": str(images_dir),
        "expected_image_counts": dict(expected_by_split),
        "existing_image_counts": dict(existing_by_split),
    }


def main() -> None:
    args = parse_args()
    root = ensure_dir(args.output_root)
    splits = list(dict.fromkeys(args.splits))

    annotation_summary = download_annotations(
        root,
        splits=splits,
        skip_existing=args.skip_existing,
        download_input_refer=args.download_input_refer,
    )
    image_summary = download_images(
        root,
        splits=splits,
        skip_existing=args.skip_existing,
        workers=args.image_workers,
    )

    summary = {
        "output_root": str(root),
        "splits": splits,
        **annotation_summary,
        **image_summary,
    }
    write_json(root / "summary.json", summary)
    write_json(root / "download_phrasecut_full_summary.json", summary)
    print(f"Wrote PhraseCut full download summary to {root / 'download_phrasecut_full_summary.json'}")


if __name__ == "__main__":
    main()
