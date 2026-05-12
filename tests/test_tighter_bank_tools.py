import json
import os
import subprocess
import sys
from pathlib import Path

from PIL import Image

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


def _save_mask(path: Path, on: bool = True) -> None:
    Image.new("L", (8, 8), 255 if on else 0).save(path)


def test_build_phrasecut_pair_bank_keeps_same_object_color_regions(tmp_path: Path):
    image = tmp_path / "img.jpg"
    Image.new("RGB", (8, 8), (0, 0, 0)).save(image)
    mask_a = tmp_path / "a.png"
    mask_b = tmp_path / "b.png"
    mask_c = tmp_path / "c.png"
    _save_mask(mask_a)
    _save_mask(mask_b)
    _save_mask(mask_c)

    source_manifest = tmp_path / "source.jsonl"
    pair_manifest = tmp_path / "pairs.jsonl"
    output_jsonl = tmp_path / "pair_bank.jsonl"
    summary_json = tmp_path / "summary.json"

    _write_jsonl(
        source_manifest,
        [
            {
                "id": "left",
                "source": "phrasecut",
                "split": "val",
                "family": "attr",
                "image_path": str(image),
                "mask_pos_path": str(mask_a),
                "prompt_pos": "red car",
                "meta": {"image_id": 1, "object_name": "car", "attributes": ["red"]},
            },
            {
                "id": "right",
                "source": "phrasecut",
                "split": "val",
                "family": "attr",
                "image_path": str(image),
                "mask_pos_path": str(mask_b),
                "prompt_pos": "blue car",
                "meta": {"image_id": 1, "object_name": "car", "attributes": ["blue"]},
            },
            {
                "id": "other",
                "source": "phrasecut",
                "split": "val",
                "family": "attr",
                "image_path": str(image),
                "mask_pos_path": str(mask_c),
                "prompt_pos": "red person",
                "meta": {"image_id": 1, "object_name": "person", "attributes": ["red"]},
            },
        ],
    )
    _write_jsonl(
        pair_manifest,
        [
            {
                "id": "pair_1",
                "source": "phrasecut+flip",
                "split": "val",
                "image_path": str(image),
                "mask_pos_path": str(mask_a),
                "mask_neg_path": str(mask_b),
                "prompt_pos": "red car",
                "prompt_neg": "blue car",
                "meta": {"object_name": "car", "attribute_type": "color", "left_id": "left", "right_id": "right"},
            }
        ],
    )

    proc = _run_script(
        "tools/build_phrasecut_pair_bank.py",
        [
            "--pair-manifest",
            str(pair_manifest),
            "--source-manifest",
            str(source_manifest),
            "--output-jsonl",
            str(output_jsonl),
            "--summary-json",
            str(summary_json),
            "--require-existing-image",
            "--require-existing-mask",
            "--same-object-name",
            "--match-pair-attribute-type",
            "--keep-attr-types",
            "color",
            "--require-color-bearing",
            "--keep-target-distractor-always",
        ],
        cwd=tmp_path,
    )
    assert proc.returncode == 0, proc.stderr
    rows = [json.loads(line) for line in output_jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) == 1
    region_ids = {region["id"] for region in rows[0]["regions"]}
    assert region_ids == {"left", "right"}


def test_check_topology_success_flags_failure_when_entity_signal_absent(tmp_path: Path):
    released = tmp_path / "released.jsonl"
    candidate = tmp_path / "candidate.jsonl"
    out = tmp_path / "result.json"
    _write_jsonl(
        released,
        [
            {"id": "a", "entity_pos_neg_localized_edit_diff_k5": 0.02, "entity_pos_neg_neighbor_flip_rate_k5_local": 0.12, "entity_pos_neg_neighbor_flip_rate_k5_far": 0.04},
            {"id": "b", "entity_pos_neg_localized_edit_diff_k5": 0.01, "entity_pos_neg_neighbor_flip_rate_k5_local": 0.10, "entity_pos_neg_neighbor_flip_rate_k5_far": 0.05},
        ],
    )
    _write_jsonl(
        candidate,
        [
            {"id": "a", "entity_pos_neg_localized_edit_diff_k5": -0.01, "entity_pos_neg_neighbor_flip_rate_k5_local": 0.08, "entity_pos_neg_neighbor_flip_rate_k5_far": 0.10},
            {"id": "b", "entity_pos_neg_localized_edit_diff_k5": -0.02, "entity_pos_neg_neighbor_flip_rate_k5_local": 0.07, "entity_pos_neg_neighbor_flip_rate_k5_far": 0.11},
        ],
    )
    proc = _run_script(
        "tools/check_topology_success.py",
        [
            "--released-per-pair",
            str(released),
            "--candidate-per-pair",
            str(candidate),
            "--output-json",
            str(out),
            "--bootstrap-samples",
            "200",
            "--strict-reject-on-fail",
        ],
        cwd=tmp_path,
    )
    assert proc.returncode == 0, proc.stderr
    result = json.loads(out.read_text(encoding="utf-8"))
    assert result["entity_lensgraph_go"] is False
    assert result["recommended_wedge"] == "patch_phenomenon_negative_result"
