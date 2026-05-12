# Entity-level topology falsification plan

This follow-up is intentionally **time-boxed**. The goal is not to keep tightening the bank indefinitely. The goal is to falsify or support one concrete hypothesis:

> A tighter, pair-conditioned PhraseCut entity bank reveals prompt-conditioned entity-topology editing that is currently washed out by a coarse per-image bank.

## Order of operations

1. Build a failure browser from the current topology probe.
2. Inspect 50-100 worst examples and classify failure modes.
3. Build a tighter **pair-conditioned** bank using the observed failure modes.
4. Re-run `eval/topology_rearrangement.py` with `--pair-bank`.
5. Run `tools/check_topology_success.py` against released vs warm checkpoints.

## Pre-registered go/no-go rule

The entity-level LensGraph claim is considered **go** only if the tighter-bank probe satisfies all of the following on the locked test set:

- entity localized edit diff@5 is positive,
- its bootstrap CI lower bound is positive,
- far-drift does not materially worsen relative to the released checkpoint,
- local flip-rate does not collapse relative to the released checkpoint,
- the result is reproducible under the same locked benchmark.

If the tighter-bank probe still fails, the recommended pivot is:

> **Wedge A:** prompt steering edits patch-level topology but does not naturally lift to entity-level abstraction.

This is an acceptable and publishable negative-result / phenomenon-paper direction.
