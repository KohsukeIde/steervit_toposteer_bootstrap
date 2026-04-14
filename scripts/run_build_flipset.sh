#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <output_manifest.jsonl> <input_manifest1.jsonl> [<input_manifest2.jsonl> ...]"
  exit 1
fi

OUT="$1"
shift

python tools/build_flipset.py \
  --output-manifest "$OUT" \
  --input-manifests "$@"
