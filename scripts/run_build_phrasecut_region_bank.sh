#!/usr/bin/env bash
set -euo pipefail

python tools/build_phrasecut_region_bank.py \
  --input-manifest "$1" \
  --output-jsonl "$2" \
  --summary-json "$3" \
  --source phrasecut \
  --require-existing-image \
  --require-existing-mask
