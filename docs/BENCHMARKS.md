# Benchmarks

This document records locked evaluation manifests, their construction rules, hashes, and usage restrictions.

## PhraseCut Locked Attr V1

Date locked: 2026-04-19

Source manifest:

```text
data/processed/phrasecut_controlled_attr_full/gold_eval.jsonl
```

Split command:

```bash
.venv/bin/python tools/split_locked_attr_benchmark.py \
  --input-manifest data/processed/phrasecut_controlled_attr_full/gold_eval.jsonl \
  --output-dir data/processed/phrasecut_locked_attr_v1 \
  --attr-type color \
  --dev-pairs 150 \
  --seed 42 \
  --prefix gold_color
```

Split unit:

```text
image_path + object_name + attribute_type + unordered(attribute_name_pos, attribute_name_neg)
```

This keeps reverse pairs and equivalent color flips in the same split.

### Manifests

| Name | Path | Pairs | SHA256 |
| --- | --- | ---: | --- |
| all color | `data/processed/phrasecut_locked_attr_v1/gold_color_all.jsonl` | 576 | `5b95d94c11c9a55159eb328cdfc25d5eabb4baf55f8be555e7db180b54ff5dda` |
| dev | `data/processed/phrasecut_locked_attr_v1/gold_color_dev.jsonl` | 150 | `1493e1baf37d18562c0d71401fa337f118cdd5781145dc347b80bad0f72a46b2` |
| test | `data/processed/phrasecut_locked_attr_v1/gold_color_test.jsonl` | 426 | `784cbdb5c5bc52928fa224b049b5589f9cca43f2abb23531452ef88d5baaa3c5` |
| non-color reference | `data/processed/phrasecut_locked_attr_v1/gold_non_color_reference.jsonl` | 28 | `21d63daffd35b75ea800c7a0ba1cf4b55e0dd4706b5988922df05a7033ee902b` |

Summary:

```text
num_input_records: 604
num_target_records(color): 576
num_reference_records(size/material): 28
num_target_groups: 285
num_dev_groups: 75
num_test_groups: 210
group_split_leakage_count: 0
```

### Usage Rules

- Use `gold_color_dev.jsonl` for checkpoint selection during warm-start experiments.
- Use `gold_color_test.jsonl` only for milestone reporting.
- Do not tune thresholds, margins, prompts, or checkpoint selection on the test split.
- Do not mix `gold_color_dev.jsonl` or `gold_color_test.jsonl` into training.
- Report size/material results separately as reference only until their pair counts are expanded.

### Released SteerViT Baseline

Checkpoint:

```text
/home/cvrt/.cache/huggingface/hub/models--JonaRuthardt--SteerViT/snapshots/cdc29ddb5ddb8cfb6c0194c461eba14309b5afc7/steervit_dinov2_base.pth
```

Dev output:

```text
runs/gate_sweep_phrasecut_locked_color_dev_released/summary.json
```

| Gate | Flip acc | 95% CI | Mean gap |
| ---: | ---: | ---: | ---: |
| 0.00 | 0.5000 | [0.4200, 0.5800] | 0.00000 |
| 0.25 | 0.6333 | [0.5600, 0.7067] | 0.00249 |
| 0.50 | 0.8133 | [0.7467, 0.8733] | 0.00899 |
| 0.75 | 0.8067 | [0.7400, 0.8667] | 0.01122 |
| 1.00 | 0.8133 | [0.7467, 0.8733] | 0.01086 |

Dev curve aggregates:

```text
flip_accuracy_augc: 0.72750
mean_gap_augc: 0.00704
```

Test output:

```text
runs/gate_sweep_phrasecut_locked_color_test_released/summary.json
```

| Gate | Flip acc | 95% CI | Mean gap |
| ---: | ---: | ---: | ---: |
| 0.00 | 0.5000 | [0.4507, 0.5469] | 0.00000 |
| 0.25 | 0.6338 | [0.5869, 0.6785] | 0.00262 |
| 0.50 | 0.7793 | [0.7394, 0.8192] | 0.00981 |
| 0.75 | 0.8333 | [0.7958, 0.8685] | 0.01280 |
| 1.00 | 0.8357 | [0.7981, 0.8709] | 0.01247 |

Test curve aggregates:

```text
flip_accuracy_augc: 0.72858
mean_gap_augc: 0.00787
```

### Leakage Checks

Train manifest:

```text
data/processed/phrasecut_full/manifest_existing_train.jsonl
```

Leakage check outputs:

```text
runs/check_split_leakage_train_vs_gold_color_dev.json
runs/check_split_leakage_train_vs_gold_color_test.json
```

Results:

```text
train unique image keys: 1122
dev unique image keys: 66
test unique image keys: 153
train/dev overlap: 0
train/test overlap: 0
```
