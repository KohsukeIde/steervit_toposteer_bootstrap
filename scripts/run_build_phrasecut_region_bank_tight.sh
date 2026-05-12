#!/usr/bin/env bash
set -euo pipefail

python tools/build_phrasecut_region_bank_tight.py \
  --input-manifest "$1" \
  --pair-manifest "$2" \
  --output-jsonl "$3" \
  --summary-json "$4" \
  --require-non-relational \
  --object-scope same_object \
  --require-attribute-type-match \
  --context-budget 16
