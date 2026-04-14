#!/usr/bin/env python
from __future__ import annotations

import argparse
from collections import Counter, defaultdict

from toposteer.utils.io import read_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Print simple manifest statistics.")
    parser.add_argument("--manifest", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_jsonl(args.manifest)
    fam = Counter(r.get("family", "plain") for r in rows)
    src = Counter(r.get("source", "custom") for r in rows)
    split = Counter(r.get("split", "custom") for r in rows)

    print(f"Total rows: {len(rows)}")
    print("By family:")
    for k, v in fam.most_common():
        print(f"  {k:12s} {v}")
    print("By source:")
    for k, v in src.most_common():
        print(f"  {k:12s} {v}")
    print("By split:")
    for k, v in split.most_common():
        print(f"  {k:12s} {v}")


if __name__ == "__main__":
    main()
