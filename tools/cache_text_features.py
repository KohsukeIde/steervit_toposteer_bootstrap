#!/usr/bin/env python
from __future__ import annotations

import argparse

from toposteer.models import SteerViTTrainable
from toposteer.utils.io import read_jsonl
from toposteer.utils.text_cache import build_text_cache


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cache unique text prompt encodings for a manifest.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_jsonl(args.manifest)
    prompts = [r.get("prompt_pos") for r in rows] + [r.get("prompt_neg") for r in rows if r.get("prompt_neg")]
    model = SteerViTTrainable(args.checkpoint, device=args.device, trainable_modules=())
    cache = build_text_cache(model, prompts=prompts, batch_size=args.batch_size, device=args.device)
    cache.save(args.output)
    print(f"Saved {len(cache.unique_prompts())} unique prompt encodings to {args.output}")


if __name__ == "__main__":
    main()
