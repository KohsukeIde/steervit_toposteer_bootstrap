#!/usr/bin/env bash
set -euo pipefail

python tools/check_topology_success.py \
  --released-per-pair runs/topology_rearrangement_released/per_pair.jsonl \
  --candidate-per-pair runs/topology_rearrangement_warm_refseg_tighter_bank/per_pair.jsonl \
  --output-json runs/topology_rearrangement_warm_refseg_tighter_bank/success_check.json \
  --bootstrap-samples 4000 \
  --k 5 \
  --max-far-drift-increase 0.01 \
  --strict-reject-on-fail
