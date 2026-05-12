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


def test_build_phrasecut_region_bank_groups_by_image(tmp_path: Path):
    image_a = tmp_path / "1.jpg"
    mask_a = tmp_path / "1_mask.png"
    image_b = tmp_path / "2.jpg"
    mask_b = tmp_path / "2_mask.png"
    for path in (image_a, mask_a, image_b, mask_b):
        path.write_text("x")

    manifest = tmp_path / "manifest.jsonl"
    output_jsonl = tmp_path / "region_bank.jsonl"
    summary_json = tmp_path / "summary.json"
    _write_jsonl(
        manifest,
        [
            {
                "id": "a1",
                "source": "phrasecut",
                "split": "val",
                "family": "attr",
                "image_path": str(image_a),
                "mask_pos_path": str(mask_a),
                "prompt_pos": "red car",
                "meta": {"image_id": 1},
            },
            {
                "id": "a2",
                "source": "phrasecut",
                "split": "val",
                "family": "plain",
                "image_path": str(image_a),
                "mask_pos_path": str(mask_a),
                "prompt_pos": "car",
                "meta": {"image_id": 1},
            },
            {
                "id": "b1",
                "source": "phrasecut",
                "split": "test",
                "family": "attr",
                "image_path": str(image_b),
                "mask_pos_path": str(mask_b),
                "prompt_pos": "blue chair",
                "meta": {"image_id": 2},
            },
        ],
    )

    proc = _run_script(
        "tools/build_phrasecut_region_bank.py",
        [
            "--input-manifest",
            str(manifest),
            "--output-jsonl",
            str(output_jsonl),
            "--summary-json",
            str(summary_json),
            "--require-existing-image",
            "--require-existing-mask",
        ],
        cwd=tmp_path,
    )
    assert proc.returncode == 0, proc.stderr
    rows = [json.loads(line) for line in output_jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) == 2
    assert rows[0]["meta"]["num_records"] >= 1
    summary = json.loads(summary_json.read_text(encoding="utf-8"))
    assert summary["num_images"] == 2
    assert summary["num_regions"] == 3
