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
