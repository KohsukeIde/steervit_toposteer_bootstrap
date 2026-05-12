#!/usr/bin/env python
from __future__ import annotations

import argparse
import html
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from tqdm import tqdm

from toposteer.evaluation import index_region_bank, record_image_key, resolve_bank_entry
from toposteer.utils import ensure_dir, load_binary_mask, read_jsonl, write_json, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render an HTML failure browser for topology probe results.")
    parser.add_argument("--pair-manifest", "--manifest", dest="pair_manifest", required=True)
    parser.add_argument("--per-pair-jsonl", "--per-pair", dest="per_pair_jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--region-bank", default=None, help="Optional image or pair-specific region bank")
    parser.add_argument("--pair-bank", default=None, help="Optional additional pair-specific bank JSONL")
    parser.add_argument("--sort-key", default="entity_pos_neg_localized_edit_diff_k5")
    parser.add_argument("--ascending", action="store_true", help="Sort ascending (useful for worst negative localized diff)")
    parser.add_argument("--descending", action="store_true", help="Sort descending (legacy alias)")
    parser.add_argument("--limit", "--top-n", dest="limit", type=int, default=100)
    parser.add_argument("--top-overlap", "--bank-overlap-preview", dest="top_overlap", type=int, default=5)
    parser.add_argument("--min-bank-regions", type=int, default=0)
    return parser.parse_args()


def _load_bank(path: str | None) -> dict[str, dict[str, dict[str, Any]]]:
    if not path:
        return {"by_image_key": {}, "by_pair_id": {}}
    return index_region_bank(read_jsonl(path))


def _merge_banks(a: dict[str, dict[str, dict[str, Any]]], b: dict[str, dict[str, dict[str, Any]]]) -> dict[str, dict[str, dict[str, Any]]]:
    return {
        "by_image_key": {**a.get("by_image_key", {}), **b.get("by_image_key", {})},
        "by_pair_id": {**a.get("by_pair_id", {}), **b.get("by_pair_id", {})},
    }


def _safe_metric(row: dict[str, Any], key: str, descending: bool) -> float:
    value = row.get(key)
    if value is None:
        return -float("inf") if descending else float("inf")
    try:
        value = float(value)
    except Exception:
        return -float("inf") if descending else float("inf")
    return -value if descending else value


def _blend_overlay(image_path: str, mask_pos_path: str, mask_neg_path: str, out_path: Path) -> None:
    image = np.array(Image.open(image_path).convert("RGB"), dtype=np.float32)
    pos = load_binary_mask(mask_pos_path).squeeze(0).numpy().astype(bool)
    neg = load_binary_mask(mask_neg_path).squeeze(0).numpy().astype(bool)
    overlay = image.copy()
    overlay[pos] = 0.55 * overlay[pos] + 0.45 * np.array([220, 40, 40], dtype=np.float32)
    overlay[neg] = 0.55 * overlay[neg] + 0.45 * np.array([40, 80, 220], dtype=np.float32)
    Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8)).save(out_path)


def _region_overlap(region_mask_path: str, query_mask: np.ndarray) -> float:
    region = load_binary_mask(region_mask_path).squeeze(0).numpy().astype(bool)
    denom = float(region.sum())
    if denom <= 0:
        return 0.0
    inter = float(np.logical_and(region, query_mask).sum())
    return inter / denom


def _top_overlap_table(pair: dict[str, Any], bank_entry: dict[str, Any] | None, top_k: int) -> dict[str, list[dict[str, Any]]]:
    if bank_entry is None:
        return {"positive": [], "negative": []}
    pos = load_binary_mask(pair["mask_pos_path"]).squeeze(0).numpy().astype(bool)
    neg = load_binary_mask(pair["mask_neg_path"]).squeeze(0).numpy().astype(bool)
    pos_rows = []
    neg_rows = []
    for region in bank_entry.get("regions", []):
        mask_path = region.get("mask_pos_path")
        if not mask_path or not Path(mask_path).exists():
            continue
        meta = region.get("meta", {}) or {}
        base = {
            "region_id": region.get("id"),
            "selection_reason": region.get("selection_reason"),
            "prompt": region.get("prompt_pos"),
            "object_name": meta.get("object_name"),
            "attributes": meta.get("attributes", []),
            "relations": meta.get("relation_descriptions", []),
        }
        pos_rows.append({**base, "overlap": _region_overlap(mask_path, pos)})
        neg_rows.append({**base, "overlap": _region_overlap(mask_path, neg)})
    pos_rows.sort(key=lambda x: (-x["overlap"], str(x.get("region_id"))))
    neg_rows.sort(key=lambda x: (-x["overlap"], str(x.get("region_id"))))
    return {"positive": pos_rows[:top_k], "negative": neg_rows[:top_k]}


def _fmt(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.4f}"
    return html.escape(str(v))


def main() -> None:
    args = parse_args()
    descending = bool(args.descending and not args.ascending)
    out_dir = ensure_dir(args.output_dir)
    assets_dir = ensure_dir(out_dir / "assets")

    pair_rows = {row["id"]: row for row in read_jsonl(args.pair_manifest)}
    per_pair = read_jsonl(args.per_pair_jsonl)
    bank_index = _merge_banks(_load_bank(args.region_bank), _load_bank(args.pair_bank))

    filtered = []
    for row in per_pair:
        if int(row.get("num_bank_regions", 0)) < int(args.min_bank_regions):
            continue
        filtered.append(row)
    filtered.sort(key=lambda row: (_safe_metric(row, args.sort_key, descending), row.get("id", "")))
    selected = filtered[: args.limit]

    html_rows: list[str] = []
    annotation_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []

    for rank, row in enumerate(tqdm(selected, desc="Rendering failure browser"), start=1):
        pair = pair_rows.get(row["id"])
        if pair is None:
            continue
        image_key = record_image_key(pair)
        bank_scope, bank_entry = resolve_bank_entry(pair, bank_index)
        overlay_path = assets_dir / f"{rank:03d}_{pair['id']}.png"
        _blend_overlay(pair["image_path"], pair["mask_pos_path"], pair["mask_neg_path"], overlay_path)
        overlap_tables = _top_overlap_table(pair, bank_entry, top_k=args.top_overlap)

        metrics = {
            "sort_metric": row.get(args.sort_key),
            "entity_locdiff_k5": row.get("entity_pos_neg_localized_edit_diff_k5"),
            "entity_local_flip_k5": row.get("entity_pos_neg_neighbor_flip_rate_k5_local"),
            "entity_far_flip_k5": row.get("entity_pos_neg_neighbor_flip_rate_k5_far"),
            "patch_locdiff_k5": row.get("patch_pos_neg_localized_edit_diff_k5"),
            "patch_local_flip_k5": row.get("patch_pos_neg_neighbor_flip_rate_k5_local"),
            "patch_far_flip_k5": row.get("patch_pos_neg_neighbor_flip_rate_k5_far"),
            "num_bank_regions": row.get("num_bank_regions"),
            "target_idx": row.get("target_idx"),
            "distractor_idx": row.get("distractor_idx"),
            "bank_scope": row.get("bank_scope", bank_scope),
        }

        def table_html(rows_: list[dict[str, Any]]) -> str:
            if not rows_:
                return "<em>none</em>"
            items = []
            for rr in rows_:
                items.append(
                    "<li>"
                    f"id={html.escape(str(rr.get('region_id')))} | obj={html.escape(str(rr.get('object_name')))} | "
                    f"overlap={rr.get('overlap', 0.0):.3f} | reason={html.escape(str(rr.get('selection_reason')))} | "
                    f"attrs={html.escape(', '.join(map(str, rr.get('attributes', []))))}"
                    "</li>"
                )
            return "<ul>" + "".join(items) + "</ul>"

        metric_table = "".join(f"<tr><th>{html.escape(k)}</th><td>{_fmt(v)}</td></tr>" for k, v in metrics.items())
        html_rows.append(
            f"""
            <div class='case'>
              <h2>#{rank} {html.escape(pair['id'])}</h2>
              <p><strong>prompt_pos</strong>: {html.escape(pair['prompt_pos'])}<br>
                 <strong>prompt_neg</strong>: {html.escape(pair['prompt_neg'])}<br>
                 <strong>image_key</strong>: {html.escape(image_key)}
              </p>
              <div class='grid'>
                <div><img src='assets/{overlay_path.name}' alt='overlay'><p>target=red, distractor=blue</p></div>
                <div>
                  <table>{metric_table}</table>
                  <p><strong>Top positive-overlap bank regions</strong></p>
                  {table_html(overlap_tables['positive'])}
                  <p><strong>Top negative-overlap bank regions</strong></p>
                  {table_html(overlap_tables['negative'])}
                </div>
              </div>
            </div>
            """
        )
        annotation_rows.append({
            "pair_id": pair["id"],
            "rank": rank,
            "sort_key": args.sort_key,
            "sort_value": row.get(args.sort_key),
            "primary_failure": None,
            "bank_issue": None,
            "ambiguity_issue": None,
            "multiple_instances": None,
            "mask_issue": None,
            "notes": "",
        })
        manifest_rows.append({
            "pair_id": pair["id"],
            "sort_value": row.get(args.sort_key),
            "overlay_path": f"assets/{overlay_path.name}",
            "bank_scope": row.get("bank_scope", bank_scope),
            "num_bank_regions": row.get("num_bank_regions"),
        })

    html_doc = f"""
    <html>
    <head>
      <meta charset='utf-8'>
      <title>Topology Failure Browser</title>
      <style>
        body {{ font-family: Arial, sans-serif; margin: 24px; }}
        .case {{ border-top: 1px solid #ccc; padding-top: 18px; margin-top: 18px; }}
        .grid {{ display: grid; grid-template-columns: 420px 1fr; gap: 18px; align-items: start; }}
        img {{ max-width: 400px; border: 1px solid #bbb; }}
        table {{ border-collapse: collapse; }}
        th, td {{ border: 1px solid #ccc; padding: 4px 8px; text-align: left; }}
        ul {{ margin-top: 0.25rem; }}
      </style>
    </head>
    <body>
      <h1>Topology Failure Browser</h1>
      <p>Sorted by <code>{html.escape(args.sort_key)}</code> ({'descending' if descending else 'ascending'}), showing {len(html_rows)} cases.</p>
      {''.join(html_rows)}
    </body>
    </html>
    """
    (out_dir / "index.html").write_text(html_doc, encoding="utf-8")
    write_jsonl(out_dir / "annotation_template.jsonl", annotation_rows)
    write_jsonl(out_dir / "manifest.jsonl", manifest_rows)
    write_json(
        out_dir / "summary.json",
        {
            "pair_manifest": args.pair_manifest,
            "per_pair_jsonl": args.per_pair_jsonl,
            "region_bank": args.region_bank,
            "pair_bank": args.pair_bank,
            "sort_key": args.sort_key,
            "descending": descending,
            "num_rendered": len(html_rows),
            "min_bank_regions": int(args.min_bank_regions),
        },
    )
    print(f"Wrote failure browser to {out_dir / 'index.html'}")


if __name__ == "__main__":
    main()
