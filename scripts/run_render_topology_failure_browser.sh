#!/usr/bin/env bash
set -euo pipefail

python tools/render_topology_failure_browser.py \
  --manifest data/processed/phrasecut_controlled_attr_full/gold_eval.jsonl \
  --per-pair runs/topology_rearrangement_warm_refseg/per_pair.jsonl \
  --region-bank data/processed/phrasecut_full/region_bank.jsonl \
  --output-dir runs/failure_browser_entity \
  --sort-key entity_pos_neg_localized_edit_diff_k5 \
  --top-n 80
