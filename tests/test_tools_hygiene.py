import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _run_script(script_relpath: str, args: list[str], cwd: Path) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(SRC_ROOT) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    return subprocess.run(
        [sys.executable, str(REPO_ROOT / script_relpath), *args],
        cwd=str(cwd),
        env=env,
        capture_output=True,
        text=True,
    )


def test_materialize_existing_records_filters_missing_required(tmp_path: Path):
    existing_img = tmp_path / "a.jpg"
    existing_mask = tmp_path / "a_mask.png"
    existing_img.write_text("img")
    existing_mask.write_text("mask")

    manifest = tmp_path / "manifest.jsonl"
    out_manifest = tmp_path / "out.jsonl"
    summary_json = tmp_path / "summary.json"
    _write_jsonl(
        manifest,
        [
            {
                "id": "keep",
                "split": "train",
                "family": "plain",
                "image_path": str(existing_img),
                "mask_pos_path": str(existing_mask),
                "prompt_pos": "chair",
            },
            {
                "id": "drop",
                "split": "train",
                "family": "plain",
                "image_path": str(tmp_path / "missing.jpg"),
                "mask_pos_path": str(existing_mask),
                "prompt_pos": "table",
            },
        ],
    )

    proc = _run_script(
        "tools/materialize_existing_records.py",
        [
            "--input-manifest",
            str(manifest),
            "--output-manifest",
            str(out_manifest),
            "--summary-json",
            str(summary_json),
        ],
        cwd=tmp_path,
    )
    assert proc.returncode == 0, proc.stderr
    lines = out_manifest.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    kept = json.loads(lines[0])
    assert kept["id"] == "keep"
    summary = json.loads(summary_json.read_text(encoding="utf-8"))
    assert summary["num_kept"] == 1
    assert summary["num_dropped"] == 1


def test_check_split_leakage_fails_on_overlap(tmp_path: Path):
    train_manifest = tmp_path / "train.jsonl"
    eval_manifest = tmp_path / "eval.jsonl"
    summary_json = tmp_path / "summary.json"
    _write_jsonl(train_manifest, [{"id": "t1", "meta": {"image_id": 123}}])
    _write_jsonl(eval_manifest, [{"id": "e1", "meta": {"image_id": 123}}])

    proc = _run_script(
        "tools/check_split_leakage.py",
        [
            "--train-manifest",
            str(train_manifest),
            "--eval-manifest",
            str(eval_manifest),
            "--summary-json",
            str(summary_json),
        ],
        cwd=tmp_path,
    )
    assert proc.returncode != 0
    summary = json.loads(summary_json.read_text(encoding="utf-8"))
    assert summary["overlap_count"] == 1
