#!/usr/bin/env bash
set -euo pipefail

python eval/topology_rearrangement.py \
  --config configs/topology_rearrangement_tighter_bank.yaml \
  --manifest data/processed/phrasecut_controlled_attr_full/gold_eval.jsonl \
  --region-bank data/processed/phrasecut_full/region_bank.jsonl \
  --pair-bank data/processed/phrasecut_tighter_pair_bank_color/gold_eval_pair_bank.jsonl \
  --checkpoint runs/warm_refseg_phrasecut_locked_color_5k/checkpoint_final.pt \
  --base-checkpoint steervit_dinov2_base.pth \
  --output-dir runs/topology_rearrangement_warm_refseg_tighter_bank
