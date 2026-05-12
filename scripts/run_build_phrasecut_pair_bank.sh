#!/usr/bin/env bash
set -euo pipefail

python tools/build_phrasecut_pair_bank.py \
  --pair-manifest data/processed/phrasecut_controlled_attr_full/gold_eval.jsonl \
  --source-manifest data/processed/phrasecut_full/manifest_existing.jsonl \
  --output-jsonl data/processed/phrasecut_tighter_pair_bank_color/gold_eval_pair_bank.jsonl \
  --summary-json data/processed/phrasecut_tighter_pair_bank_color/gold_eval_pair_bank_summary.json \
  --require-existing-image \
  --require-existing-mask \
  --require-non-relational \
  --same-object-name \
  --match-pair-attribute-type \
  --keep-attr-types color \
  --require-color-bearing \
  --keep-target-distractor-always \
  --max-regions-per-pair 24
