#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 ]]; then
  echo "Usage: $0 <train_manifest.jsonl> <val_manifest.jsonl> <checkpoint> <output_dir>"
  exit 1
fi

python train/train_refseg.py \
  --config configs/warm_cf.yaml \
  --train-manifest "$1" \
  --val-manifest "$2" \
  --checkpoint "$3" \
  --output-dir "$4"
