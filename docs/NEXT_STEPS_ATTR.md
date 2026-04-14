# Next steps for the clean attribute track

The current recommended mainline is:

1. Prepare PhraseCut manifests.
2. Mine `gold_eval.jsonl` and `silver_train.jsonl` with `tools/mine_phrasecut_controlled_attr.py`.
3. Lock `gold_eval.jsonl` and never tune directly on it.
4. Run `eval/gate_sweep.py` on `gold_eval.jsonl` to confirm the released checkpoint still carries attribute signal.
5. Run `train/train_refseg.py` with `configs/warm_refseg.yaml` on positive supervision only.
6. Only after the mixed-batch fix is in place and the clean attribute set is large enough, turn on `configs/warm_cf.yaml`.

## Why this path is preferred

The bottleneck is currently **controlled data purity**, not model capacity.

- If the benchmark is noisy, stronger backbones only hide the problem.
- If paired and non-paired samples are mixed naively, the counterfactual loss disappears silently.
- If the clean benchmark is tiny, repeated design changes will overfit to it.

## What not to do yet

- Do not move the mainline to Franca yet.
- Do not add topology losses yet.
- Do not add Grounding-DINO or a VLM into the online training loop.
- Do not use PACO fallback pairs as if they were strict counterfactuals.

## Good failure modes

The project is still healthy if:

- `gold_eval` stays small but clean,
- `warm_refseg` improves dense localization without hurting the attribute gate sweep,
- `warm_cf` only gives modest gains at first.

Those outcomes are much better than inflating the benchmark with uncontrolled pairs.
