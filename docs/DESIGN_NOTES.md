# Design notes

## Why not edit the upstream SteerViT repository directly?

Because the public release is currently strongest as an **inference / model** package. Keeping experimental training code in a separate scaffold gives cleaner iteration, easier versioning, and less merge pain when the upstream repo changes.

## Why start with counterfactual loss before topology loss?

Because it directly tests the core claim.

If `"red car"` cannot beat `"blue car"` on the same image, then a more elegant local-topology story is probably premature. Counterfactual losses are the smallest hammer that still strikes the actual hypothesis.

## Why preserve background instead of prompt-off global features?

Prompt-off already collapses to the vanilla backbone path by construction. The bigger failure mode is collateral activation spread when prompting is turned on. That is why the first preservation term here is **background drift**, not generic full-image distillation.

## Why use entity-like masks before entity tokens?

Because masks are concrete and debug-friendly. Neighborhood graphs over pooled entity tokens are interesting, but they are a second-order abstraction. Early experiments should remain visually inspectable.
