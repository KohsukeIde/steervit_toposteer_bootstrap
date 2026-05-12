# Topology probe: what to run next

This document adds the **phenomenon-first** probing layer on top of the existing clean color-attribute pipeline.

The goal is not to train a new model first. The goal is to answer a narrower question:

> When the prompt changes from `A` to `B`, does the neighborhood graph over entity / region tokens actually rearrange?

The code added here supports three steps.

## 1. Build a per-image region bank

Use the full positive PhraseCut manifest to group all region annotations that belong to the same image.

```bash
python tools/build_phrasecut_region_bank.py \
  --input-manifest data/processed/phrasecut_full/manifest_existing.jsonl \
  --output-jsonl data/processed/phrasecut_full/region_bank.jsonl \
  --summary-json data/processed/phrasecut_full/region_bank_summary.json \
  --source phrasecut \
  --require-existing-image \
  --require-existing-mask
```

This writes one JSONL row per image. Each row contains the image path and the list of region records on that image.

## 2. Measure prompt-conditioned kNN rearrangement

Run the topology probe on the locked paired benchmark.

### Released SteerViT

```bash
python eval/topology_rearrangement.py \
  --config configs/topology_rearrangement.yaml \
  --manifest data/processed/phrasecut_controlled_attr_full/gold_eval.jsonl \
  --region-bank data/processed/phrasecut_full/region_bank.jsonl \
  --checkpoint steervit_dinov2_base.pth \
  --output-dir runs/topology_rearrangement_released
```

### Warm-refseg checkpoint

```bash
python eval/topology_rearrangement.py \
  --config configs/topology_rearrangement.yaml \
  --manifest data/processed/phrasecut_controlled_attr_full/gold_eval.jsonl \
  --region-bank data/processed/phrasecut_full/region_bank.jsonl \
  --checkpoint runs/warm_refseg_phrasecut_locked_color_5k/checkpoint_final.pt \
  --base-checkpoint steervit_dinov2_base.pth \
  --output-dir runs/topology_rearrangement_warm_refseg
```

Important metrics in `summary.json`:

- `entity_pos_neg_neighbor_flip_rate_k*_local`
- `entity_pos_neg_neighbor_flip_rate_k*_far`
- `entity_pos_neg_localized_edit_ratio_k*`
- `entity_pos_neg_rank_spearman_*`
- patch-level counterparts under the `patch_...` prefix

Interpretation:

- `local flip > far flip` means the graph changes mostly around target / distractor regions.
- `localized_edit_ratio > 1` is the main qualitative success signal.
- if entity-level localization is cleaner than patch-level localization, the case for region/entity pooling becomes much stronger.

## 3. Render a figure for one pair

```bash
python tools/render_lensgraph_figure.py \
  --config configs/topology_rearrangement.yaml \
  --manifest data/processed/phrasecut_controlled_attr_full/gold_eval.jsonl \
  --region-bank data/processed/phrasecut_full/region_bank.jsonl \
  --checkpoint runs/warm_refseg_phrasecut_locked_color_5k/checkpoint_final.pt \
  --base-checkpoint steervit_dinov2_base.pth \
  --pair-index 0 \
  --output-dir runs/lensgraph_figure_example
```

This writes:

- `lensgraph_figure.png`
- `figure_metadata.json`

The figure shows:

- image with target / distractor masks
- entity graph with prompt off
- entity graph with prompt A
- entity graph with prompt B

## Recommended order

1. released SteerViT, gate 1.0
2. warm-refseg final, gate 1.0
3. compare entity vs patch metrics
4. only then decide whether topology-specific training is worth adding
