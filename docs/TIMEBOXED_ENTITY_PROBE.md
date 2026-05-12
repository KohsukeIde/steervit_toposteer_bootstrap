# Time-boxed entity-bank probe

This document encodes the next step after the first topology probe.

## Goal

Test whether the weak/negative entity-level localized edit signal is mostly a bank-quality issue, rather than an intrinsic limitation of SteerViT's patch-trained steering.

## Order of operations

1. Build a failure browser from the current `per_pair.jsonl`.
2. Manually inspect 50-100 worst cases and annotate the dominant failure mode.
3. Use those annotations to pick one tighter-bank design.
4. Re-run the topology probe with the tighter bank.
5. Check the result against the pre-registered criteria in `configs/topology_success_entity.yaml`.

## Time box

Two weeks from the first tighter-bank run.

If the tighter bank still fails the criteria, stop iterating on bank cleanup and pivot to:

> prompt steering edits patch-level topology but does not naturally lift to entity-level abstraction.

This is the honest fallback paper wedge.

## Non-goals during the time box

Do not add:

- Franca
- Grounding-DINO / VLM-generated supervision
- topology loss
- extra controller branches

The purpose of this window is diagnosis, not rescue by complexity.
