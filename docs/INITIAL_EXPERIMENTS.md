# Initial experiment plan

This note is the **research execution order**, not the final paper order.

For executed runs and decisions, see `docs/EXPERIMENT_LOG.md`.

## Objective

Before writing a method section, answer a narrower question:

> Can released SteerViT checkpoints already support prompt-conditioned target flipping for parts and attributes, and can warm-start counterfactual fine-tuning improve that without spraying activation across the rest of the image?

## Phase 0: install and smoke test

Target time: half day

Checklist:

1. install SteerViT,
2. install this scaffold,
3. run `tests/`,
4. run `eval/gate_sweep.py` on the synthetic example manifest.

Expected output:
- metrics JSON,
- optional heatmap overlays,
- no import or shape bugs.

## Phase 1: zero-shot diagnostic on a small real subset

Target time: 1 to 2 days

Data:
- 500 to 2,000 examples from PhraseCut and PACO.
- Favor `attr` and `part` families.
- Include paired distractors whenever possible.

Commands:
- `tools/prepare_phrasecut.py`
- `tools/prepare_paco.py`
- `tools/build_flipset.py`
- `eval/gate_sweep.py`

Primary metrics:
- positive mask mass,
- negative mask mass,
- positive-negative gap,
- flip accuracy,
- contrast ratio,
- optional heatmap entropy.

Decision rule:
- If the released checkpoint shows no gate-sensitive behavior on hard pairs, stop and revisit the hypothesis.
- If gate `1.0` consistently beats gate `0.0` on the flip set, continue.

## Phase 2: warm-start refseg only

Target time: 2 to 3 days

Unfreeze only:
- all `gated_cross_attn`,
- `connector`,
- `lin_seg_head`.

Loss:
- `L_refseg` only.

Purpose:
- prove the training loop is sound,
- establish a warm-start baseline before any new loss is added.

Decision rule:
- If training is unstable, fix optimization or data issues before touching the method.

## Phase 3: warm-start + counterfactual

Target time: 2 to 4 days

Loss:
- `L_refseg + λ_cf L_cf + λ_bg L_bg`

Interpretation:
- `L_cf` teaches the model to choose the intended region over a matched distractor.
- `L_bg` discourages global collateral damage outside the target / distractor region.

Primary comparison:
- released checkpoint
- warm-start refseg
- warm-start refseg + counterfactual

Decision rule:
- If `warm_cf` does not beat `warm_refseg` on flip accuracy, do not move to topology losses yet.

## Phase 4: only then consider topology loss

This scaffold intentionally leaves topology loss as a follow-up, because it is easy to romanticize and hard to debug.

Add it only when:
- the paired counterfactual signal is already clean,
- heatmaps are not collapsing to trivial broad activation,
- background drift is controlled.

## Recommended sample budgets

### Smoke diagnostic
- 128 to 512 examples

### Small real zero-shot
- 500 to 2,000 examples

### Warm-start sanity training
- 5k to 20k steps

### Warm-start counterfactual
- 10k to 50k steps

## Keep these plots

For every run, save:

1. gate factor vs flip accuracy,
2. gate factor vs positive-negative gap,
3. training step vs each loss term,
4. background drift histogram,
5. qualitative heatmaps for the same fixed examples across checkpoints.

These are not decorative. They determine whether the hypothesis is real or just a benchmark hiccup.
