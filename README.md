# TopoSteer bootstrap on top of SteerViT

This repository is a **small-scale research scaffold** for testing one specific hypothesis:

> Prompt conditioning should not only rerank regions after the fact. It should actively rewrite local neighborhood structure inside a visual encoder.

The code is designed to sit **next to** the public `SteerViT` package, not to rewrite its repo in-place. The public release already exposes a frozen ViT with text-conditioned gated cross-attention, prompt-aware dense features, heatmaps, and gate scaling. This scaffold adds the missing research plumbing:

- manifest-based datasets for early diagnostics,
- zero-shot gate sweeps,
- paired flip-set evaluation for part / attribute disambiguation,
- warm-start fine-tuning with the original patch-level soft segmentation objective,
- counterfactual losses for prompt flipping,
- background-drift regularization,
- preprocessing scripts for RefCOCOg, PhraseCut, and PACO into a unified manifest format.

## Why this shape?

The current public `SteerViT` release is excellent for inference and qualitative probing, but it does not yet ship a full public training and evaluation pipeline. That makes it a strong **model dependency**, but a poor place to bury new experiment code. The recommended workflow is therefore:

1. install or clone `SteerViT`,
2. install this scaffold in editable mode,
3. point the scaffold at a released checkpoint,
4. run zero-shot diagnostics first,
5. only then spend compute on warm-start fine-tuning.

## Research plan encoded in this scaffold

### Stage 0: zero-shot diagnostics
Use the released checkpoint to answer a brutal question before training anything:

- Do part prompts actually move heatmaps?
- Do attribute prompts flip target vs distractor mass?
- Does gate sweeping make the model smoothly more conditional?

### Stage 1: warm-start sanity fine-tuning
Unfreeze only:

- gated cross-attention modules,
- text-to-vision connector,
- linear segmentation head.

Train with the original patch-level soft segmentation objective on a small processed manifest.

### Stage 2: counterfactual prompting
Add a paired loss that penalizes failures like:

- `"red car"` still firing on the blue car,
- `"dog eye"` still firing on the ear,
- `"wooden chair leg"` not separating from another chair part.

### Stage 3: topology-aware follow-up
Only after Stage 2 works, add more ambitious losses over pooled entity tokens and neighborhood graphs.

This repository implements Stage 0, Stage 1, and the first practical version of Stage 2.

## Project layout

```text
configs/                  YAML experiment configs
docs/                     design notes and manifest spec
eval/                     evaluation entrypoints
examples/manifests/       tiny synthetic example manifest
scripts/                  helper shell scripts
src/toposteer/            reusable library code
tools/                    preprocessing and manifest builders
train/                    training entrypoints
tests/                    lightweight unit tests
```

## Installation

### 1. Create an environment

```bash
conda create -n toposteer python=3.10 -y
conda activate toposteer
```

### 2. Install SteerViT

Either install directly from GitHub:

```bash
python -m pip install "git+https://github.com/JonaRuthardt/SteerViT.git"
```

or clone it separately for source-level inspection.

### 3. Install this scaffold

From this repository root:

```bash
python -m pip install -e .
```

### 4. Optional extra dependencies

For dataset preparation and mask decoding:

```bash
python -m pip install pycocotools
```

## Unified manifest format

Every experiment in this scaffold consumes a JSONL manifest with records like:

```json
{
  "id": "paco_000123_attr_red_car",
  "source": "paco",
  "split": "train",
  "family": "attr",
  "image_path": "/abs/path/to/image.jpg",
  "mask_pos_path": "/abs/path/to/positive_mask.png",
  "mask_neg_path": "/abs/path/to/negative_mask.png",
  "prompt_pos": "red car",
  "prompt_neg": "blue car",
  "meta": {
    "object_name": "car",
    "attribute_name": "red",
    "attribute_type": "color",
    "image_id": 123
  }
}
```

Only four fields are strictly required for Stage 0 and Stage 1:

- `image_path`
- `mask_pos_path`
- `prompt_pos`
- `id`

The paired fields `mask_neg_path` and `prompt_neg` become active in counterfactual experiments.

See `docs/DATA_MANIFEST_SPEC.md` for the full schema.

## First experiments to run

### A. Gate sweep on a tiny subset

```bash
python eval/gate_sweep.py \
  --config configs/smoke_zero_shot.yaml \
  --manifest path/to/flipset_small.jsonl \
  --checkpoint steervit_dinov2_base.pth \
  --output-dir runs/gate_sweep_small
```

Success signal:
- positive-mask mass rises with gate factor,
- negative-mask mass falls or stays flat,
- flip accuracy improves between gate `0.0` and `1.0`.

### B. Warm-start original objective

```bash
python train/train_refseg.py \
  --config configs/warm_refseg.yaml \
  --train-manifest path/to/train.jsonl \
  --val-manifest path/to/val_flipset.jsonl \
  --checkpoint steervit_dinov2_base.pth \
  --output-dir runs/warm_refseg
```

Success signal:
- training loss falls,
- validation flip accuracy does not collapse,
- mask-IoU proxy or positive-mass rises.

### C. Warm-start + counterfactual

```bash
python train/train_refseg.py \
  --config configs/warm_cf.yaml \
  --train-manifest path/to/train_flipset.jsonl \
  --val-manifest path/to/val_flipset.jsonl \
  --checkpoint steervit_dinov2_base.pth \
  --output-dir runs/warm_cf
```

Success signal:
- hard part / attribute flip accuracy rises over warm-start baseline,
- background drift remains controlled.

## Dataset preparation

### RefCOCOg
This script expects the common `refer`-style pickle plus COCO instance annotations:

```bash
python tools/prepare_refcocog.py \
  --refs-pkl /path/to/refs(umd).p \
  --instances-json /path/to/instances.json \
  --images-dir /path/to/train2014 \
  --output-dir data/processed/refcocog
```

### PhraseCut
This script expects the official phrase annotation JSON and Visual Genome metadata:

```bash
python tools/prepare_phrasecut.py \
  --refer-json /path/to/refer_train.json \
  --image-meta-json /path/to/image_data_split3000.json \
  --images-dir /path/to/VG_100K \
  --output-dir data/processed/phrasecut_train
```

### PACO
This script expects a PACO-LVIS style COCO JSON and the matching image directory:

```bash
python tools/prepare_paco.py \
  --annotations-json /path/to/paco_lvis_v1_train.json \
  --images-dir /path/to/lvis_images \
  --output-dir data/processed/paco_train
```

After preprocessing, build a paired hard set:

```bash
python tools/build_flipset.py \
  --input-manifests data/processed/paco_train/manifest.jsonl data/processed/phrasecut_train/manifest.jsonl \
  --output-manifest data/processed/flipset_train.jsonl
```

## Important practical notes

- Start with **RefCOCOg + PhraseCut + PACO**, not everything at once.
- For the first two weeks, ignore 3D completely.
- Do **not** add extra FFNs or architecture changes until Stage 2 already works.
- Use the released checkpoint first. A cold start is a later question.
- The real go/no-go is **flip accuracy**, not whether a generic RefCOCO metric nudges upward.

## Experiment log

Diagnostics and result-driven decisions should be recorded in `docs/EXPERIMENT_LOG.md`.

The current logged result is:

- Franca RASA reduces position leakage, but does not yet clearly beat raw tokens on semantic kNN purity or mask retrieval.
- SteerViT gate sweep shows a strong prompt-conditioning signal on the balanced hard set.
- Mainline should continue with SteerViT `warm_refseg -> warm_cf`; Franca stays as an ablation path for now.

## Recommended schedule

Read `docs/INITIAL_EXPERIMENTS.md`.

## License note

This scaffold does not bundle the upstream SteerViT code or checkpoints. Install those separately.

## What to inspect first

If you only open four files, open these:

- `tools/build_flipset.py`
- `eval/gate_sweep.py`
- `train/train_refseg.py`
- `src/toposteer/losses/counterfactual.py`
