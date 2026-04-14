#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 <manifest.jsonl> <checkpoint> <output_dir>"
  exit 1
fi

python eval/gate_sweep.py \
  --config configs/smoke_zero_shot.yaml \
  --manifest "$1" \
  --checkpoint "$2" \
  --output-dir "$3"
