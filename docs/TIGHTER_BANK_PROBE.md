# Tighter-bank topology probe

This document adds the three pieces needed for the next probe cycle:

1. `tools/render_topology_failure_browser.py`
2. `tools/build_phrasecut_pair_bank.py`
3. `tools/check_topology_success.py`

## Recommended sequence

### 1) Inspect failure modes first

```bash
python tools/render_topology_failure_browser.py \
  --manifest data/processed/phrasecut_controlled_attr_full/gold_eval.jsonl \
  --per-pair runs/topology_rearrangement_warm_refseg/per_pair.jsonl \
  --region-bank data/processed/phrasecut_full/region_bank.jsonl \
  --output-dir runs/failure_browser_entity \
  --sort-key entity_pos_neg_localized_edit_diff_k5 \
  --top-n 80
```

### 2) Build a tighter pair-conditioned bank

```bash
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
```

### 3) Re-run the topology probe with `--pair-bank`

```bash
python eval/topology_rearrangement.py \
  --config configs/topology_rearrangement_tighter_bank.yaml \
  --manifest data/processed/phrasecut_controlled_attr_full/gold_eval.jsonl \
  --region-bank data/processed/phrasecut_full/region_bank.jsonl \
  --pair-bank data/processed/phrasecut_tighter_pair_bank_color/gold_eval_pair_bank.jsonl \
  --checkpoint runs/warm_refseg_phrasecut_locked_color_5k/checkpoint_final.pt \
  --base-checkpoint steervit_dinov2_base.pth \
  --output-dir runs/topology_rearrangement_warm_refseg_tighter_bank
```

### 4) Run the pre-registered decision rule

```bash
python tools/check_topology_success.py \
  --released-per-pair runs/topology_rearrangement_released/per_pair.jsonl \
  --candidate-per-pair runs/topology_rearrangement_warm_refseg_tighter_bank/per_pair.jsonl \
  --output-json runs/topology_rearrangement_warm_refseg_tighter_bank/success_check.json \
  --bootstrap-samples 4000 \
  --k 5 \
  --max-far-drift-increase 0.01 \
  --strict-reject-on-fail
```

If the check fails after the tighter-bank probe, the default recommendation is to pivot to the patch-level phenomenon framing.
