# Unified manifest spec

The scaffold uses a single JSONL schema so that different datasets can feed the same training and evaluation code.

## Record schema

```json
{
  "id": "unique_record_id",
  "source": "refcocog | phrasecut | paco | custom",
  "split": "train | val | test | miniv | custom",
  "family": "plain | attr | part | relation | negation | custom",
  "image_path": "/abs/path/to/image.jpg",
  "mask_pos_path": "/abs/path/to/positive_mask.png",
  "mask_neg_path": "/abs/path/to/negative_mask.png",
  "prompt_pos": "red car",
  "prompt_neg": "blue car",
  "meta": {
    "image_id": 123,
    "ann_id": 456,
    "object_name": "car",
    "attribute_name": "red",
    "attribute_type": "color",
    "part_name": "wheel"
  }
}
```

## Required fields

Only these are strictly required by all code paths:

- `id`
- `image_path`
- `mask_pos_path`
- `prompt_pos`

## Optional but recommended

- `source`
- `split`
- `family`
- `prompt_neg`
- `mask_neg_path`
- `meta`

## Mask format

Masks should be stored as grayscale PNGs:

- foreground: nonzero / white
- background: zero / black

The loader binarizes them automatically.

## Family semantics

### `plain`
Standard referring segmentation without a hard structured distractor.

### `attr`
Prompt differs mainly by attribute.
Examples:
- `red car` vs `blue car`
- `wood chair` vs `metal chair`

### `part`
Prompt differs mainly by part.
Examples:
- `dog eye` vs `dog ear`
- `bicycle wheel` vs `bicycle seat`

### `relation`
Prompt differs by relationship.
Examples:
- `person holding umbrella`
- `cup on table`

### `negation`
Used only if explicit negative text is available.
Examples:
- `car without wheels`
- `not red`

## How the scripts use these fields

### `eval/gate_sweep.py`
Needs:
- `image_path`
- `prompt_pos`
- `mask_pos_path`
- optional `prompt_neg` and `mask_neg_path`

### `train/train_refseg.py`
Needs:
- the same four basic fields,
- paired negative fields only if `use_counterfactual=true`.

### `tools/build_flipset.py`
Relies heavily on `family` and `meta`.
It can still fall back to mask-IoU pairing if metadata is missing, but metadata makes the pairs much better.
