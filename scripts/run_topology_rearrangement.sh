#!/usr/bin/env bash
set -euo pipefail

MANIFEST="$1"
REGION_BANK="$2"
CHECKPOINT="$3"
OUTPUT_DIR="$4"
BASE_CHECKPOINT="${5:-}"

CMD=(python eval/topology_rearrangement.py \
  --config configs/topology_rearrangement.yaml \
  --manifest "$MANIFEST" \
  --region-bank "$REGION_BANK" \
  --checkpoint "$CHECKPOINT" \
  --output-dir "$OUTPUT_DIR")

if [[ -n "$BASE_CHECKPOINT" ]]; then
  CMD+=(--base-checkpoint "$BASE_CHECKPOINT")
fi

"${CMD[@]}"
