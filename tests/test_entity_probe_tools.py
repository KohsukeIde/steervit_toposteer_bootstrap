import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _run(script_relpath: str, args: list[str], cwd: Path) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(SRC_ROOT) + (os.pathsep + env.get("PYTHONPATH", "") if env.get("PYTHONPATH") else "")
    return subprocess.run(
        [sys.executable, str(REPO_ROOT / script_relpath), *args],
        cwd=str(cwd),
        env=env,
        capture_output=True,
        text=True,
    )


def _save_mask(path: Path, coords: tuple[slice, slice]) -> None:
    arr = np.zeros((8, 8), dtype=np.uint8)
    arr[coords] = 255
    Image.fromarray(arr).save(path)


def test_build_phrasecut_region_bank_tight_pair_scoped(tmp_path: Path):
    image = tmp_path / "img.png"
    Image.fromarray(np.full((8, 8, 3), 128, dtype=np.uint8)).save(image)

    m1 = tmp_path / "m1.png"
    m2 = tmp_path / "m2.png"
    m3 = tmp_path / "m3.png"
    m4 = tmp_path / "m4.png"
    _save_mask(m1, (slice(0, 4), slice(0, 4)))
    _save_mask(m2, (slice(0, 4), slice(4, 8)))
    _save_mask(m3, (slice(4, 8), slice(0, 4)))
    _save_mask(m4, (slice(4, 8), slice(4, 8)))

    source_manifest = tmp_path / "source.jsonl"
    pair_manifest = tmp_path / "pairs.jsonl"
    out_jsonl = tmp_path / "tight_bank.jsonl"
    summary_json = tmp_path / "summary.json"

    _write_jsonl(
        source_manifest,
        [
            {"id": "r1", "image_path": str(image), "mask_pos_path": str(m1), "prompt_pos": "blue shorts", "source": "phrasecut", "split": "val", "meta": {"image_id": 1, "object_name": "shorts", "attributes": ["blue"], "relation_descriptions": []}},
            {"id": "r2", "image_path": str(image), "mask_pos_path": str(m2), "prompt_pos": "red shorts", "source": "phrasecut", "split": "val", "meta": {"image_id": 1, "object_name": "shorts", "attributes": ["red"], "relation_descriptions": []}},
            {"id": "r3", "image_path": str(image), "mask_pos_path": str(m3), "prompt_pos": "green shorts", "source": "phrasecut", "split": "val", "meta": {"image_id": 1, "object_name": "shorts", "attributes": ["green"], "relation_descriptions": []}},
            {"id": "r4", "image_path": str(image), "mask_pos_path": str(m4), "prompt_pos": "blue shirt", "source": "phrasecut", "split": "val", "meta": {"image_id": 1, "object_name": "shirt", "attributes": ["blue"], "relation_descriptions": []}},
        ],
    )
    _write_jsonl(
        pair_manifest,
        [
            {
                "id": "pair1",
                "image_path": str(image),
                "mask_pos_path": str(m1),
                "mask_neg_path": str(m2),
                "prompt_pos": "blue shorts",
                "prompt_neg": "red shorts",
                "split": "val",
                "source": "phrasecut+flip",
                "meta": {"image_id": 1, "left_id": "r1", "right_id": "r2", "object_name": "shorts", "attribute_type": "color"},
            }
        ],
    )

    proc = _run(
        "tools/build_phrasecut_region_bank_tight.py",
        [
            "--input-manifest", str(source_manifest),
            "--pair-manifest", str(pair_manifest),
            "--output-jsonl", str(out_jsonl),
            "--summary-json", str(summary_json),
            "--require-existing-image",
            "--require-existing-mask",
            "--require-non-relational",
            "--object-scope", "same_object",
            "--require-attribute-type-match",
            "--context-budget", "8",
        ],
        cwd=tmp_path,
    )
    assert proc.returncode == 0, proc.stderr
    rows = [json.loads(line) for line in out_jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) == 1
    row = rows[0]
    assert row["pair_id"] == "pair1"
    region_ids = [r["id"] for r in row["regions"]]
    assert region_ids == ["r1", "r2", "r3"]
    reasons = [r["selection_reason"] for r in row["regions"]]
    assert reasons.count("pair_endpoint") == 2
    assert "same_object_attr" in reasons


def test_check_topology_success_pass(tmp_path: Path):
    candidate = tmp_path / "candidate.json"
    baseline = tmp_path / "baseline.json"
    out = tmp_path / "report.json"
    candidate.write_text(json.dumps({
        "entity_pos_neg_localized_edit_diff_k5": 0.02,
        "entity_pos_neg_neighbor_flip_rate_k5_far": 0.10,
        "entity_pos_target_to_distractor_rank_hit_at_5": 0.40,
        "patch_pos_neg_localized_edit_diff_k5": 0.05,
    }), encoding="utf-8")
    baseline.write_text(json.dumps({
        "entity_pos_neg_neighbor_flip_rate_k5_far": 0.12,
        "entity_pos_target_to_distractor_rank_hit_at_5": 0.35,
    }), encoding="utf-8")

    proc = _run(
        "tools/check_topology_success.py",
        [
            "--criteria", str(REPO_ROOT / "configs/topology_success_entity.yaml"),
            "--candidate-summary", str(candidate),
            "--baseline-summary", str(baseline),
            "--output-json", str(out),
        ],
        cwd=tmp_path,
    )
    assert proc.returncode == 0, proc.stderr
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["pass"] is True
    assert len(report["checks"]) >= 4


def test_render_topology_failure_browser_smoke(tmp_path: Path):
    image = tmp_path / "img.png"
    Image.fromarray(np.full((8, 8, 3), 180, dtype=np.uint8)).save(image)
    m1 = tmp_path / "m1.png"
    m2 = tmp_path / "m2.png"
    m3 = tmp_path / "m3.png"
    _save_mask(m1, (slice(0, 4), slice(0, 4)))
    _save_mask(m2, (slice(0, 4), slice(4, 8)))
    _save_mask(m3, (slice(4, 8), slice(0, 4)))

    pair_manifest = tmp_path / "pairs.jsonl"
    per_pair = tmp_path / "per_pair.jsonl"
    bank = tmp_path / "bank.jsonl"
    out_dir = tmp_path / "browser"

    _write_jsonl(
        pair_manifest,
        [{
            "id": "pair1",
            "image_path": str(image),
            "mask_pos_path": str(m1),
            "mask_neg_path": str(m2),
            "prompt_pos": "blue shorts",
            "prompt_neg": "red shorts",
            "split": "val",
            "meta": {"image_id": 1, "left_id": "r1", "right_id": "r2"},
        }],
    )
    _write_jsonl(
        per_pair,
        [{
            "id": "pair1",
            "entity_pos_neg_localized_edit_diff_k5": -0.02,
            "entity_pos_neg_neighbor_flip_rate_k5_local": 0.3,
            "entity_pos_neg_neighbor_flip_rate_k5_far": 0.4,
            "patch_pos_neg_localized_edit_diff_k5": 0.05,
            "patch_pos_neg_neighbor_flip_rate_k5_local": 0.5,
            "patch_pos_neg_neighbor_flip_rate_k5_far": 0.1,
            "num_bank_regions": 3,
            "target_idx": 0,
            "distractor_idx": 1,
            "bank_scope": "pair",
        }],
    )
    _write_jsonl(
        bank,
        [{
            "pair_id": "pair1",
            "image_key": "image_id:1",
            "image_path": str(image),
            "regions": [
                {"id": "r1", "mask_pos_path": str(m1), "prompt_pos": "blue shorts", "selection_reason": "pair_endpoint", "meta": {"object_name": "shorts", "attributes": ["blue"], "relation_descriptions": []}},
                {"id": "r2", "mask_pos_path": str(m2), "prompt_pos": "red shorts", "selection_reason": "pair_endpoint", "meta": {"object_name": "shorts", "attributes": ["red"], "relation_descriptions": []}},
                {"id": "r3", "mask_pos_path": str(m3), "prompt_pos": "green shorts", "selection_reason": "same_object_attr", "meta": {"object_name": "shorts", "attributes": ["green"], "relation_descriptions": []}},
            ],
        }],
    )

    proc = _run(
        "tools/render_topology_failure_browser.py",
        [
            "--pair-manifest", str(pair_manifest),
            "--per-pair-jsonl", str(per_pair),
            "--output-dir", str(out_dir),
            "--region-bank", str(bank),
            "--ascending",
            "--limit", "10",
        ],
        cwd=tmp_path,
    )
    assert proc.returncode == 0, proc.stderr
    assert (out_dir / "index.html").exists()
    assert (out_dir / "annotation_template.jsonl").exists()
    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["num_rendered"] == 1
