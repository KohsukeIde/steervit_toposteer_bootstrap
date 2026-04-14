# Experiment log

This file is the running record for diagnostics, training runs, results, and go/no-go decisions.

When adding a new run, record:

- date,
- exact manifest and config,
- exact command or a reproducible command sketch,
- output directory,
- headline metrics,
- decision and next action.

## 2026-04-14: Stage 0 diagnostic bootstrap

### Goal

Run the first real-data diagnostics before any training:

- SteerViT gate sweep on paired hard examples.
- Franca raw vs RASA feature-space diagnostic.
- Decide whether Franca should become the mainline backbone now, or remain an ablation while the SteerViT baseline moves forward.

### Environment

Project root:

```text
/home/cvrt/Desktop/dev/topo_steer/steervit_toposteer_bootstrap
```

Third-party source checkouts:

```text
third_party/Franca
third_party/SteerViT
third_party/PhraseCutDataset
```

Python environment:

```text
.venv
Python 3.12.3
torch 2.11.0+cu128
torchvision 0.26.0+cu128
```

Notes:

- Franca and SteerViT were installed editable with `--no-deps` to avoid dependency pin conflicts.
- Franca diagnostics were run with `XFORMERS_DISABLED=1`.
- PACO-LVIS `val` references images outside COCO `val2017`, so `tools/download_diagnostic_data.py` also downloads PACO-referenced images from each annotation's `coco_url`.

### Code changes

Added:

- `src/toposteer/models/franca_adapter.py`
- `eval/franca_space_diagnostic.py`
- `tools/download_diagnostic_data.py`

Updated:

- `src/toposteer/models/__init__.py`
- `tools/prepare_paco.py`
- `tools/build_flipset.py`
- `eval/gate_sweep.py`

Important behavior:

- `FrancaAdapter` returns raw `x_norm_patchtokens` and RASA `patch_token_rasa`.
- `eval/franca_space_diagnostic.py` writes `summary.json` and `per_entity.jsonl`.
- `tools/build_flipset.py` now supports fallback same-image pairing, per-family caps, fallback-only mode, and interleaved family output.
- `eval/gate_sweep.py` now includes per-family metrics in `summary.json`.

### Data

Raw data summary:

```text
data/raw/diagnostic_data_summary.json
```

Processed manifests:

| Manifest | Rows | Notes |
| --- | ---: | --- |
| `data/processed/phrasecut_miniv/manifest.jsonl` | 821 | PhraseCut miniv |
| `data/processed/paco_lvis_val/manifest.jsonl` | 49,258 | PACO-LVIS val |
| `data/processed/flipset_diag.jsonl` | 1,200 | initial hard set; attr-heavy |
| `data/processed/flipset_balanced_attr_part.jsonl` | 768 | balanced hard set |

Balanced hard set family counts:

| Family | Pairs |
| --- | ---: |
| `attr` | 256 |
| `part` | 256 |
| `part_attr` | 256 |

Balanced hard set command:

```bash
.venv/bin/python tools/build_flipset.py \
  --input-manifests data/processed/paco_lvis_val/manifest.jsonl data/processed/phrasecut_miniv/manifest.jsonl \
  --output-manifest data/processed/flipset_balanced_attr_part.jsonl \
  --keep-families attr part part_attr \
  --skip-strict-pairing \
  --fallback-image-pairs \
  --max-fallback-pairs 768 \
  --max-fallback-pairs-per-group 16 \
  --max-pairs-per-family 256 \
  --max-iou 0.5 \
  --min-mask-area 10 \
  --interleave-family-output
```

### Run A: initial SteerViT gate sweep

Output:

```text
runs/gate_sweep_diag/summary.json
```

Command sketch:

```bash
.venv/bin/python eval/gate_sweep.py \
  --config configs/smoke_zero_shot.yaml \
  --manifest data/processed/flipset_diag.jsonl \
  --checkpoint steervit_dinov2_base.pth \
  --output-dir runs/gate_sweep_diag \
  --limit 64 \
  --device cuda \
  --override batch_size=2 num_workers=2 num_overlay_samples=8
```

Headline metrics:

| Gate | Flip acc | Mean gap |
| ---: | ---: | ---: |
| 0.00 | 0.5938 | 0.00160 |
| 0.25 | 0.6719 | 0.01300 |
| 0.50 | 0.6875 | 0.01535 |
| 0.75 | 0.7188 | 0.01271 |
| 1.00 | 0.6875 | 0.01106 |

Interpretation:

The released SteerViT checkpoint shows a clear gate-conditioned signal. Gate `0.75` gives the best flip accuracy on this small initial subset.

### Run B: initial Franca raw vs RASA diagnostic

Output:

```text
runs/franca_space_diag/summary.json
```

Command:

```bash
XFORMERS_DISABLED=1 .venv/bin/python eval/franca_space_diagnostic.py \
  --manifest data/processed/flipset_diag.jsonl \
  --output-dir runs/franca_space_diag \
  --limit 512 \
  --batch-size 4 \
  --num-workers 2 \
  --device cuda \
  --arch vitb14 \
  --weights IN21K \
  --position-sample-patches 256
```

Headline metrics:

| Metric | Raw | RASA |
| --- | ---: | ---: |
| mask retrieval gap | 0.28710 | 0.27977 |
| position leakage corr | 0.42495 | 0.33166 |
| pos-neg separation | 0.37769 | 0.37064 |
| kNN purity@1 | 0.47105 | 0.47105 |
| kNN purity@5 | 0.45397 | 0.45103 |
| kNN purity@10 | 0.33503 | 0.33533 |

Interpretation:

RASA reduces position leakage, but does not clearly beat raw tokens on kNN purity or mask retrieval in this attr-heavy initial set.

### Run C: balanced Franca raw vs RASA diagnostic

Output:

```text
runs/franca_space_balanced_attr_part/summary.json
```

Command:

```bash
XFORMERS_DISABLED=1 .venv/bin/python eval/franca_space_diagnostic.py \
  --manifest data/processed/flipset_balanced_attr_part.jsonl \
  --output-dir runs/franca_space_balanced_attr_part \
  --limit 768 \
  --batch-size 4 \
  --num-workers 2 \
  --device cuda \
  --arch vitb14 \
  --weights IN21K \
  --position-sample-patches 256
```

Headline metrics:

| Metric | Raw | RASA |
| --- | ---: | ---: |
| mask retrieval gap | 0.47181 | 0.45212 |
| position leakage corr | 0.42900 | 0.32941 |
| pos-neg separation | 0.57915 | 0.55923 |
| kNN purity@1 | 0.59608 | 0.60000 |
| kNN purity@5 | 0.58078 | 0.58013 |
| kNN purity@10 | 0.56647 | 0.56725 |

Family-specific kNN purity:

| Family | k | Raw | RASA |
| --- | ---: | ---: | ---: |
| `attr` | 1 | 0.50491 | 0.50491 |
| `attr` | 5 | 0.52063 | 0.51866 |
| `attr` | 10 | 0.52908 | 0.52908 |
| `part` | 1 | 0.98633 | 0.98828 |
| `part` | 5 | 0.92852 | 0.93125 |
| `part` | 10 | 0.89180 | 0.89531 |
| `part_attr` | 1 | 0.29470 | 0.30452 |
| `part_attr` | 5 | 0.29116 | 0.28841 |
| `part_attr` | 10 | 0.27662 | 0.27544 |

Interpretation:

The balanced set confirms the robust part of the Franca story: RASA substantially reduces position leakage. However, RASA does not yet provide a clear semantic-neighborhood advantage over raw tokens. Raw remains better on mask retrieval gap and pos-neg separation, while kNN purity is mostly tied.

### Run D: balanced SteerViT gate sweep

Output:

```text
runs/gate_sweep_balanced_attr_part/summary.json
```

Command:

```bash
.venv/bin/python eval/gate_sweep.py \
  --config configs/smoke_zero_shot.yaml \
  --manifest data/processed/flipset_balanced_attr_part.jsonl \
  --checkpoint steervit_dinov2_base.pth \
  --output-dir runs/gate_sweep_balanced_attr_part \
  --limit 256 \
  --device cuda \
  --override batch_size=4 num_workers=2 save_overlays=False
```

Overall metrics:

| Gate | Flip acc | Mean gap |
| ---: | ---: | ---: |
| 0.00 | 0.3789 | -0.00026 |
| 0.25 | 0.6133 | 0.00289 |
| 0.50 | 0.7148 | 0.00559 |
| 0.75 | 0.7578 | 0.00586 |
| 1.00 | 0.7734 | 0.00600 |

Family-specific flip accuracy:

| Gate | `attr` | `part` | `part_attr` |
| ---: | ---: | ---: | ---: |
| 0.00 | 0.5465 | 0.2824 | 0.3059 |
| 0.25 | 0.7442 | 0.3529 | 0.7412 |
| 0.50 | 0.7791 | 0.4353 | 0.9294 |
| 0.75 | 0.7907 | 0.5529 | 0.9294 |
| 1.00 | 0.8605 | 0.5882 | 0.8706 |

Interpretation:

SteerViT shows a strong prompt-conditioning signal on the balanced hard set. The gate sweep improves overall flip accuracy from `0.3789` to `0.7734`, with especially strong gains for `attr` and `part_attr`. `part` is weaker but still improves monotonically.

### Decision

Do not move the mainline fully to Franca yet.

Current evidence supports:

- Keep Franca as an ablation / appendix path for now.
- Preserve the dual-space idea as a hypothesis, but do not make it the main contribution until RASA shows a clearer semantic-neighborhood advantage.
- Move the mainline back to SteerViT baseline diagnostics and continue to `warm_refseg` then `warm_cf`.

Next experiment:

```text
SteerViT released checkpoint
vs SteerViT + warm_refseg
vs SteerViT + warm_cf
on data/processed/flipset_balanced_attr_part.jsonl
```

### Verification

Commands:

```bash
.venv/bin/python -m compileall src tools eval train tests
.venv/bin/python -m pytest tests
```

Result:

```text
5 passed
```

## 2026-04-14: Feedback-based next policy

### Decision

Keep the mainline on SteerViT, and keep Franca as an ablation / appendix path for now.

Do not start `warm_refseg` or `warm_cf` from the current balanced flip set as if it were a controlled benchmark. The SteerViT gate signal is promising, but the next risk is benchmark construction, not model wiring.

### Why

The feedback points to one real interpretation risk:

```text
balanced family count != controlled counterfactual difficulty
```

The current fallback pairing path relaxes metadata constraints:

- `attr` fallback only requires different attributes, not the same object class.
- `part` fallback only requires different parts, not the same parent object.
- `part_attr` fallback allows both part and attribute to change at once.
- `part_attr` strict pairing is also still too loose, because it only rejects exact `(part_name, attribute_name)` matches.

This means the high `part_attr` flip accuracy may partly reflect easy negatives where both the part and attribute change, rather than genuine part-attribute disentanglement.

### Next implementation step

Tighten `tools/build_flipset.py` before more model experiments.

Required additions:

1. Pair provenance fields on every flip record:

```text
pair_origin = strict | fallback
same_object_name
same_parent
same_part
same_attribute_name
same_attribute_type
num_changed_fields
changed_fields
```

2. Controlled family variants:

```text
attr_same_object_diff_attr
part_same_parent_diff_part
part_attr_same_part_diff_attr
part_attr_same_attr_diff_part
```

3. Summary tables in the sidecar summary JSON:

```text
pairs_by_family
pairs_by_pair_origin
pairs_by_control_family
pairs_by_num_changed_fields
```

4. Rebuild hard sets in this order:

```text
strict_attr_only
strict_part_only
controlled_part_attr_split
```

5. Rerun SteerViT gate sweep on those controlled hard sets.

### Training rule

Only after the controlled gate sweeps still show a strong signal:

```text
SteerViT released checkpoint
vs SteerViT + warm_refseg
vs SteerViT + warm_cf
```

The first `warm_cf` run should use only controlled `attr_same_object_diff_attr` and `part_same_parent_diff_part`. Keep `part_attr` out until the split is clean.

### Small code hygiene before larger sweeps

Add `torch.inference_mode()` or `torch.no_grad()` to `eval/gate_sweep.py` before running larger diagnostics.

Also avoid mixed positive-only and paired samples in `warm_cf` until `collate_refexp` is made explicit about partial negative masks. Otherwise, counterfactual loss can silently disappear on mixed batches.

## 2026-04-14: Controlled flip-set rebuild and gate sweep

### Goal

Act on the feedback above rather than moving directly to training.

Specifically:

- Add provenance and controlled subtype accounting to `tools/build_flipset.py`.
- Add `torch.inference_mode()` to `eval/gate_sweep.py`.
- Rebuild controlled flip sets.
- Rerun SteerViT gate sweep on whatever controlled set can actually be built from the current diagnostic data.

### Code changes

`tools/build_flipset.py` now writes these fields on every pair:

```text
base_family
pair_origin
control_family
same_object_name
same_parent
same_part
same_attribute_name
same_attribute_type
attribute_type_compatible
changed_fields
num_changed_fields
mask_iou
```

It also writes summary counts:

```text
pairs_by_family
pairs_by_base_family
pairs_by_pair_origin
pairs_by_control_family
pairs_by_num_changed_fields
pairs_by_changed_fields
```

`eval/gate_sweep.py` now wraps the evaluation loop in `torch.inference_mode()`.

### Controlled set availability

The stricter construction exposed a data limitation.

| Manifest | Pairs | Interpretation |
| --- | ---: | --- |
| `data/processed/flipset_control_attr_strict.jsonl` | 0 | PACO attributes are mostly multiple labels on the same annotation, not separate same-object instances with different masks. |
| `data/processed/flipset_control_part_same_parent_strict.jsonl` | 0 | PACO part `parent_obj_ann_id` is effectively self-referential in the processed val manifest. |
| `data/processed/flipset_control_part_same_object.jsonl` | 0 | Same-image same-object part groups do not contain different part names after mask/metadata filters. |
| `data/processed/flipset_control_paco_part_attr_split.jsonl` | 0 | Controlled `same_part_diff_attr` / `same_attr_diff_part` pairs are not available with distinct masks under the current filters. |
| `data/processed/flipset_control_phrasecut_attr.jsonl` | 30 | PhraseCut miniv yields a small controlled attribute set. |

Controlled PhraseCut attr summary:

```text
num_pairs: 30
pairs_by_family: {"attr_same_object_diff_attr": 30}
pairs_by_base_family: {"attr": 20, "plain": 8, "relation": 2}
pairs_by_pair_origin: {"strict": 30}
pairs_by_control_family: {"attr_same_object_diff_attr": 30}
pairs_by_num_changed_fields: {"1": 30}
pairs_by_changed_fields: {"attribute_name": 30}
```

Provenance rebuild of the old balanced fallback set:

```text
manifest: data/processed/flipset_balanced_attr_part_provenance.jsonl
num_pairs: 768
pairs_by_family: {"attr": 256, "part": 256, "part_attr": 256}
pairs_by_pair_origin: {"fallback": 768}
pairs_by_control_family: {"attr_uncontrolled": 256, "part_uncontrolled": 256, "part_attr_uncontrolled": 256}
pairs_by_num_changed_fields: {"2": 10, "3": 361, "4": 234, "5": 163}
```

This confirms the earlier concern: the apparently strong balanced set was balanced by family label, but not controlled by counterfactual difficulty. Most pairs change object identity and multiple metadata fields at once.

### Run E: controlled PhraseCut attr gate sweep

Output:

```text
runs/gate_sweep_control_phrasecut_attr/summary.json
```

Command:

```bash
.venv/bin/python eval/gate_sweep.py \
  --config configs/smoke_zero_shot.yaml \
  --manifest data/processed/flipset_control_phrasecut_attr.jsonl \
  --checkpoint steervit_dinov2_base.pth \
  --output-dir runs/gate_sweep_control_phrasecut_attr \
  --limit 30 \
  --device cuda \
  --override batch_size=4 num_workers=2 save_overlays=False
```

Overall metrics:

| Gate | Flip acc | Mean gap |
| ---: | ---: | ---: |
| 0.00 | 0.5000 | -0.00000 |
| 0.25 | 0.6000 | 0.00015 |
| 0.50 | 0.6333 | 0.00388 |
| 0.75 | 0.6667 | 0.00396 |
| 1.00 | 0.7333 | 0.00524 |

Interpretation:

The controlled subset is tiny, so this is not a final benchmark. Still, the gate signal survives the stricter construction: flip accuracy rises from `0.50` at gate `0.0` to `0.7333` at gate `1.0`.

### Updated decision

Do not start `warm_cf` on the old balanced PACO fallback set.

The next model-training milestone needs a larger clean hard set first. Current options:

1. Expand PhraseCut beyond `miniv` and rebuild `attr_same_object_diff_attr`.
2. Add a dataset with explicit same-object attribute / part distractors.
3. Run a very small `warm_refseg` sanity check only to test the trainer, not to claim hard counterfactual gains.

For the paper path, the clean next step is option 1 or 2, then rerun controlled gate sweeps before `warm_cf`.
