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

## 2026-04-15: Attr-first scaffold integration

### Decision

Move the immediate mainline to:

```text
SteerViT + clean same-object attribute flips
```

Freeze PACO fallback benchmarks, part / part-attribute claims, Franca mainline migration, topology loss, and Grounding-DINO / VLM online loops until the attribute benchmark is clean and large enough.

The candidate patch was integrated from:

```text
/home/cvrt/Desktop/dev/topo_steer/toposteer_steervit_scaffold_attr_fix.zip
```

### Code changes

Added:

- `tools/mine_phrasecut_controlled_attr.py`
- `tools/audit_paco_controls.py`
- `docs/NEXT_STEPS_ATTR.md`
- `tests/test_datasets.py`

Updated:

- `src/toposteer/datasets/unified_refexp.py`
- `src/toposteer/losses/counterfactual.py`
- `train/train_refseg.py`
- `eval/gate_sweep.py`
- `eval/flip_accuracy.py`
- `eval/franca_space_diagnostic.py`
- `tests/test_losses.py`

Important behavior:

- `collate_refexp` now always returns zero-filled `masks_neg`, plus `valid_neg_mask` and `valid_prompt_neg_mask`.
- `warm_cf` computes counterfactual loss only on the valid negative subset, so mixed positive-only / paired batches no longer silently drop CF supervision.
- `gate_sweep.py` uses `torch.inference_mode()`, respects `valid_neg_mask`, writes family summaries, and can emit bootstrap CIs via `bootstrap_samples`.
- `flip_accuracy.py` and the Franca diagnostic no longer treat zero-filled negative masks as real negatives.
- Empty counterfactual subsets now return zero loss instead of producing invalid reductions.

### Verification

Commands:

```bash
.venv/bin/python -m compileall src tools eval train tests
.venv/bin/python -m pytest tests
```

Result:

```text
7 passed
```

## 2026-04-19: Locked color dev/test and train availability

### Goal

Act on the attr-first feedback:

- Treat the current benchmark as color-first, not broad attribute reasoning.
- Split the 604-pair gold eval into a group-wise locked color dev/test benchmark.
- Make sure reverse pairs and equivalent color flips cannot cross dev/test.
- Check whether train images are locally available before `warm_refseg`.
- Add locked eval hooks and gate-curve AUGC support for the next training milestone.

### Code changes

Integrated from the locked-eval update patch:

- `tools/materialize_existing_records.py`
- `tools/check_split_leakage.py`
- locked eval / best-checkpoint selection in `train/train_refseg.py`
- `curve_aggregates.flip_accuracy_augc` in `eval/gate_sweep.py`
- `normalized_trapz_area` in `src/toposteer/evaluation/metrics.py`

Added:

- `tools/split_locked_attr_benchmark.py`
- `tools/download_phrasecut_manifest_images.py`
- `docs/BENCHMARKS.md`

Local addition:

- `tools/materialize_existing_records.py` now supports `--keep-splits`, so train availability can be checked without accidentally retaining `val/test/miniv`.

### Locked color split

Command:

```bash
.venv/bin/python tools/split_locked_attr_benchmark.py \
  --input-manifest data/processed/phrasecut_controlled_attr_full/gold_eval.jsonl \
  --output-dir data/processed/phrasecut_locked_attr_v1 \
  --attr-type color \
  --dev-pairs 150 \
  --seed 42 \
  --prefix gold_color
```

Output:

```text
data/processed/phrasecut_locked_attr_v1/gold_color_all.jsonl
data/processed/phrasecut_locked_attr_v1/gold_color_dev.jsonl
data/processed/phrasecut_locked_attr_v1/gold_color_test.jsonl
data/processed/phrasecut_locked_attr_v1/gold_non_color_reference.jsonl
data/processed/phrasecut_locked_attr_v1/summary.json
```

Counts:

| Split | Pairs | Groups |
| --- | ---: | ---: |
| all color | 576 | 285 |
| dev | 150 | 75 |
| test | 426 | 210 |
| non-color reference | 28 | n/a |

Split unit:

```text
image_path + object_name + attribute_type + unordered(attribute_name_pos, attribute_name_neg)
```

Result:

```text
group_split_leakage_count = 0
```

Hashes and usage rules are recorded in:

```text
docs/BENCHMARKS.md
```

### Train image availability

Initial train-only materialization:

```bash
.venv/bin/python tools/materialize_existing_records.py \
  --input-manifest data/processed/phrasecut_full/manifest.jsonl \
  --output-manifest data/processed/phrasecut_full/manifest_existing_train.jsonl \
  --summary-json data/processed/phrasecut_full/manifest_existing_train_summary.json \
  --keep-splits train \
  --strip-missing-neg
```

Initial result:

```text
Kept 0 / 345486 records
```

This confirmed that `warm_refseg` should not start before addressing train image availability.

Selective image download was then run for the images referenced by controlled attr train manifests:

```bash
.venv/bin/python tools/download_phrasecut_manifest_images.py \
  --manifests \
    data/processed/phrasecut_controlled_attr_full/gold_train.jsonl \
    data/processed/phrasecut_controlled_attr_full/silver_train.jsonl \
  --image-meta-json data/raw/phrasecut/VGPhraseCut_v0/image_data_split.json \
  --images-dir data/raw/phrasecut/VGPhraseCut_v0/images \
  --summary-json data/raw/phrasecut/VGPhraseCut_v0/download_attr_train_images_summary.json \
  --workers 16 \
  --skip-existing
```

Result:

```text
unique requested train images: 1122
existing after download: 1122
failures: 0
```

Train-only materialization after selective download:

```text
data/processed/phrasecut_full/manifest_existing_train.jsonl
```

Result:

| Metric | Count |
| --- | ---: |
| train records kept | 6,090 |
| attr records kept | 2,445 |
| plain records kept | 3,498 |
| relation records kept | 147 |

### Leakage checks

Commands:

```bash
.venv/bin/python tools/check_split_leakage.py \
  --train-manifest data/processed/phrasecut_full/manifest_existing_train.jsonl \
  --eval-manifest data/processed/phrasecut_locked_attr_v1/gold_color_dev.jsonl \
  --summary-json runs/check_split_leakage_train_vs_gold_color_dev.json
```

```bash
.venv/bin/python tools/check_split_leakage.py \
  --train-manifest data/processed/phrasecut_full/manifest_existing_train.jsonl \
  --eval-manifest data/processed/phrasecut_locked_attr_v1/gold_color_test.jsonl \
  --summary-json runs/check_split_leakage_train_vs_gold_color_test.json
```

Results:

```text
train/dev overlap = 0
train/test overlap = 0
```

### Run J: released SteerViT on locked color dev

Output:

```text
runs/gate_sweep_phrasecut_locked_color_dev_released/summary.json
```

Metrics:

| Gate | Flip acc | 95% CI | Mean gap |
| ---: | ---: | ---: | ---: |
| 0.00 | 0.5000 | [0.4200, 0.5800] | 0.00000 |
| 0.25 | 0.6333 | [0.5600, 0.7067] | 0.00249 |
| 0.50 | 0.8133 | [0.7467, 0.8733] | 0.00899 |
| 0.75 | 0.8067 | [0.7400, 0.8667] | 0.01122 |
| 1.00 | 0.8133 | [0.7467, 0.8733] | 0.01086 |

Curve aggregates:

```text
flip_accuracy_augc = 0.72750
mean_gap_augc = 0.00704
```

### Run K: released SteerViT on locked color test

Output:

```text
runs/gate_sweep_phrasecut_locked_color_test_released/summary.json
```

Metrics:

| Gate | Flip acc | 95% CI | Mean gap |
| ---: | ---: | ---: | ---: |
| 0.00 | 0.5000 | [0.4507, 0.5469] | 0.00000 |
| 0.25 | 0.6338 | [0.5869, 0.6785] | 0.00262 |
| 0.50 | 0.7793 | [0.7394, 0.8192] | 0.00981 |
| 0.75 | 0.8333 | [0.7958, 0.8685] | 0.01280 |
| 1.00 | 0.8357 | [0.7981, 0.8709] | 0.01247 |

Curve aggregates:

```text
flip_accuracy_augc = 0.72858
mean_gap_augc = 0.00787
```

### Decision

Proceed to the `warm_refseg` sanity run, using:

```text
train: data/processed/phrasecut_full/manifest_existing_train.jsonl
locked eval/dev: data/processed/phrasecut_locked_attr_v1/gold_color_dev.jsonl
locked test: data/processed/phrasecut_locked_attr_v1/gold_color_test.jsonl
```

Use dev for checkpoint selection. Keep test for milestone reporting only.

### Run L: warm_refseg locked-eval smoke

Purpose:

```text
Verify that train/train_refseg.py can train on the materialized train subset,
run the locked dev eval hook, and write checkpoint_best.pt.
This is not a result run.
```

Command:

```bash
.venv/bin/python train/train_refseg.py \
  --config configs/warm_refseg.yaml \
  --train-manifest data/processed/phrasecut_full/manifest_existing_train.jsonl \
  --eval-manifest data/processed/phrasecut_locked_attr_v1/gold_color_dev.jsonl \
  --checkpoint /home/cvrt/.cache/huggingface/hub/models--JonaRuthardt--SteerViT/snapshots/cdc29ddb5ddb8cfb6c0194c461eba14309b5afc7/steervit_dinov2_base.pth \
  --output-dir runs/warm_refseg_locked_eval_smoke \
  --best-metric flip_acc \
  --best-tiebreak mean_gap \
  --override max_steps=2 batch_size=2 eval_batch_size=4 num_workers=0 eval_every=1 save_every=999999 lr=1e-4
```

Result:

```text
locked_num_pairs = 150
locked_flip_accuracy = 0.8133
locked_mean_gap = 0.01086
checkpoint_best.pt written
```

Decision:

```text
The 5k warm_refseg sanity run is unblocked.
```

### Run F: PhraseCut miniv controlled attr mining smoke

Output:

```text
data/processed/phrasecut_controlled_attr_miniv/summary.json
```

Command:

```bash
.venv/bin/python tools/mine_phrasecut_controlled_attr.py \
  --input-manifest data/processed/phrasecut_miniv/manifest.jsonl \
  --output-dir data/processed/phrasecut_controlled_attr_miniv
```

Headline counts:

| Split tier | Pairs |
| --- | ---: |
| `gold_train` | 0 |
| `gold_eval` | 14 |
| `silver_train` | 0 |
| `silver_eval` | 20 |

All `gold_eval` pairs are `color` attribute flips.

Interpretation:

This confirms the miner works on the current local data, but `miniv` is too small for the next milestone. The actual go/no-go still requires PhraseCut full or another source that yields at least 100 clean gold attr eval pairs.

### Run G: PACO strict control audit

Output:

```text
runs/audit_paco_controls_val_attr_first/summary.json
```

Command:

```bash
.venv/bin/python tools/audit_paco_controls.py \
  --input-manifest data/processed/paco_lvis_val/manifest.jsonl \
  --output-dir runs/audit_paco_controls_val_attr_first
```

Headline counts:

| Metric | Value |
| --- | ---: |
| records | 49,258 |
| groups | 45,622 |
| attr candidate pairs | 7,900 |
| part_attr candidate pairs | 1,740 |
| valid strict examples | 0 |

Invalid reasons:

```text
high_iou: 9640
same_ann: 9640
```

Interpretation:

Under the current PACO-LVIS val manifest and strict mask/metadata filters, PACO does not provide usable v1 controlled pairs. Keep PACO out of the main benchmark for now.

### Run H: miniv gold attr gate sweep with bootstrap CI

Output:

```text
runs/gate_sweep_phrasecut_attr_miniv_gold_bootstrap/summary.json
```

Command:

```bash
.venv/bin/python eval/gate_sweep.py \
  --config configs/smoke_zero_shot.yaml \
  --manifest data/processed/phrasecut_controlled_attr_miniv/gold_eval.jsonl \
  --checkpoint /home/cvrt/.cache/huggingface/hub/models--JonaRuthardt--SteerViT/snapshots/cdc29ddb5ddb8cfb6c0194c461eba14309b5afc7/steervit_dinov2_base.pth \
  --output-dir runs/gate_sweep_phrasecut_attr_miniv_gold_bootstrap \
  --override save_overlays=False bootstrap_samples=2000
```

Metrics:

| Gate | Flip acc | 95% CI | Mean gap |
| ---: | ---: | ---: | ---: |
| 0.00 | 0.5000 | [0.2143, 0.7857] | 0.00000 |
| 0.25 | 0.5714 | [0.3571, 0.7857] | 0.00357 |
| 0.50 | 0.7143 | [0.5000, 0.9286] | 0.01399 |
| 0.75 | 0.7143 | [0.5000, 0.9286] | 0.01177 |
| 1.00 | 0.7857 | [0.5714, 1.0000] | 0.00968 |

Interpretation:

The CI path works, and the released SteerViT signal is still visible even on the stricter miniv-derived gold set. Because `n=14`, treat this only as a smoke diagnostic, not a benchmark claim.

Additional compatibility check:

```bash
.venv/bin/python eval/flip_accuracy.py \
  --manifest data/processed/phrasecut_controlled_attr_miniv/gold_eval.jsonl \
  --checkpoint /home/cvrt/.cache/huggingface/hub/models--JonaRuthardt--SteerViT/snapshots/cdc29ddb5ddb8cfb6c0194c461eba14309b5afc7/steervit_dinov2_base.pth \
  --gate-factor 1.0 \
  --device cuda \
  --batch-size 4
```

Result:

```text
flip_accuracy=0.7857
mean_gap=0.0097
```

### Next action

Prepare PhraseCut full and rerun `tools/mine_phrasecut_controlled_attr.py`. The next go/no-go is:

```text
gold_eval >= 100 clean same-object attribute pairs
released SteerViT gate 1.0 > gate 0.0 with CI recorded
warm_refseg does not degrade locked gold attr eval
warm_cf(attr-only) beats warm_refseg on locked gold attr eval
```

## 2026-04-15: PhraseCut full controlled attr benchmark

### Goal

Build the first full-scale clean same-object attribute benchmark from PhraseCut, then rerun the released SteerViT gate sweep with bootstrap confidence intervals.

This milestone intentionally does not run `warm_refseg`, `warm_cf`, topology loss, Franca, or Grounding-DINO.

### Code changes

Added:

- `tools/download_phrasecut_full.py`

Updated:

- `tools/prepare_phrasecut.py`

Important behavior:

- `tools/download_phrasecut_full.py` downloads upstream PhraseCut annotations and image URLs into `data/raw/phrasecut/VGPhraseCut_v0/`, skips existing files, and writes both `summary.json` and `download_phrasecut_full_summary.json`.
- `tools/prepare_phrasecut.py` now accepts one or more `--refer-json` inputs and can run with `--allow-missing-images`. This allows full annotation mining before all train images are locally present.

### Data download status

Annotations are present for all requested splits:

```text
refer_train.json
refer_val.json
refer_test.json
refer_miniv.json
refer_input_train.json
refer_input_val.json
refer_input_test.json
refer_input_miniv.json
image_data_split.json
```

Image status:

| Split | Expected images | Existing images |
| --- | ---: | ---: |
| `val` | 2,900 | 2,900 |
| `test` | 2,601 | 2,601 |
| `miniv` | 100 | 100 |

The initial full image download was started with all splits, but the per-image upstream download path was too slow for the train split in this milestone. The run was then resumed for `val / test / miniv`, which are the splits required for the locked gold eval gate sweep. Train images must be completed before a full positive-only `warm_refseg` run that reads train images.

Reproducible eval-image completion command:

```bash
.venv/bin/python tools/download_phrasecut_full.py \
  --output-root data/raw/phrasecut/VGPhraseCut_v0 \
  --splits val test miniv \
  --image-workers 16 \
  --skip-existing
```

Output:

```text
data/raw/phrasecut/VGPhraseCut_v0/summary.json
data/raw/phrasecut/VGPhraseCut_v0/download_phrasecut_full_summary.json
```

### Full manifest preparation

Command:

```bash
.venv/bin/python tools/prepare_phrasecut.py \
  --refer-json \
    data/raw/phrasecut/VGPhraseCut_v0/refer_train.json \
    data/raw/phrasecut/VGPhraseCut_v0/refer_val.json \
    data/raw/phrasecut/VGPhraseCut_v0/refer_test.json \
    data/raw/phrasecut/VGPhraseCut_v0/refer_miniv.json \
  --image-meta-json data/raw/phrasecut/VGPhraseCut_v0/image_data_split.json \
  --images-dir data/raw/phrasecut/VGPhraseCut_v0/images \
  --output-dir data/processed/phrasecut_full \
  --allow-missing-images
```

Output:

```text
data/processed/phrasecut_full/manifest.jsonl
data/processed/phrasecut_full/summary.json
```

Headline counts:

| Metric | Count |
| --- | ---: |
| total records | 345,486 |
| `train` records | 310,816 |
| `val` records | 19,495 |
| `test` records | 14,354 |
| `miniv` records | 821 |
| `attr` records | 40,878 |
| `relation` records | 9,297 |
| `plain` records | 295,311 |
| missing-image records | 310,816 |

The missing-image count corresponds to train records under the current local image state.

### Controlled attr mining

Command:

```bash
.venv/bin/python tools/mine_phrasecut_controlled_attr.py \
  --input-manifest data/processed/phrasecut_full/manifest.jsonl \
  --output-dir data/processed/phrasecut_controlled_attr_full
```

Output:

```text
data/processed/phrasecut_controlled_attr_full/summary.json
data/processed/phrasecut_controlled_attr_full/gold_eval.jsonl
data/processed/phrasecut_controlled_attr_full/gold_train.jsonl
data/processed/phrasecut_controlled_attr_full/silver_eval.jsonl
data/processed/phrasecut_controlled_attr_full/silver_train.jsonl
```

Pair counts:

| Tier | Pairs |
| --- | ---: |
| `gold_train` | 2,322 |
| `gold_eval` | 604 |
| `silver_train` | 2,430 |
| `silver_eval` | 636 |

Gold eval attribute types:

| Attribute type | Pairs |
| --- | ---: |
| `color` | 576 |
| `size` | 26 |
| `material` | 2 |

Decision:

```text
GO for locked gold-eval gate sweep.
gold_eval = 604 >= 100.
```

### Run I: full PhraseCut gold attr gate sweep

Output:

```text
runs/gate_sweep_phrasecut_attr_full_gold_bootstrap/summary.json
```

Command:

```bash
.venv/bin/python eval/gate_sweep.py \
  --config configs/smoke_zero_shot.yaml \
  --manifest data/processed/phrasecut_controlled_attr_full/gold_eval.jsonl \
  --checkpoint /home/cvrt/.cache/huggingface/hub/models--JonaRuthardt--SteerViT/snapshots/cdc29ddb5ddb8cfb6c0194c461eba14309b5afc7/steervit_dinov2_base.pth \
  --output-dir runs/gate_sweep_phrasecut_attr_full_gold_bootstrap \
  --override save_overlays=False bootstrap_samples=2000 limit=None
```

The `limit=None` override is intentional because `configs/smoke_zero_shot.yaml` defaults to `limit: 256`, while this run evaluates all 604 gold pairs.

Metrics:

| Gate | Flip acc | 95% CI | Mean gap |
| ---: | ---: | ---: | ---: |
| 0.00 | 0.5000 | [0.4603, 0.5414] | 0.00000 |
| 0.25 | 0.6291 | [0.5911, 0.6689] | 0.00247 |
| 0.50 | 0.7831 | [0.7517, 0.8146] | 0.00925 |
| 0.75 | 0.8245 | [0.7947, 0.8560] | 0.01199 |
| 1.00 | 0.8278 | [0.7996, 0.8576] | 0.01169 |

Interpretation:

The released SteerViT checkpoint shows a strong prompt-conditioned signal on the full clean PhraseCut gold attr benchmark. Gate `0.0` is at chance, while gate `1.0` reaches `0.8278` flip accuracy with a positive mean gap and a narrow bootstrap CI.

### Decision

Proceed with the attr-first mainline.

Next milestone:

1. Complete or otherwise scope train-image availability before `warm_refseg`.
2. Run positive-only `warm_refseg` as a trainer and checkpoint-stability sanity check.
3. Keep `gold_eval.jsonl` locked for milestone checks only, not threshold tuning or checkpoint selection.
4. Start `warm_cf(attr-only)` only after the positive-only run is stable.

### Verification

Commands:

```bash
.venv/bin/python -m compileall src tools eval train tests
.venv/bin/python -m pytest tests
```

Result:

```text
7 passed
```
