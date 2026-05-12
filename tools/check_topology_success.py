#!/usr/bin/env python
from __future__ import annotations

import argparse
from typing import Any

from toposteer.config import load_yaml
from toposteer.evaluation import summarize_bootstrap_metrics
from toposteer.utils import read_json, read_jsonl, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check topology probe outputs against pre-registered criteria.")
    # New summary-vs-summary mode
    parser.add_argument("--criteria", default=None, help="YAML file describing required checks")
    parser.add_argument("--candidate-summary", default=None)
    parser.add_argument("--baseline-summary", default=None)
    # Backward-compatible per-pair bootstrap mode
    parser.add_argument("--released-per-pair", default=None)
    parser.add_argument("--candidate-per-pair", default=None)
    parser.add_argument("--bootstrap-samples", type=int, default=4000)
    parser.add_argument("--bootstrap-ci", type=float, default=0.95)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--min-entity-localized-diff", type=float, default=0.0)
    parser.add_argument("--min-entity-localized-gain", type=float, default=0.0)
    parser.add_argument("--max-far-drift-increase", type=float, default=0.01)
    parser.add_argument("--min-local-flip-gain", type=float, default=0.0)
    parser.add_argument("--strict-reject-on-fail", action="store_true")
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def _get_metric(summary: dict[str, Any], key: str) -> float | None:
    value = summary.get(key)
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _compare(lhs: float | None, rhs: float | None, op: str) -> bool:
    if lhs is None or rhs is None:
        return False
    if op == ">":
        return lhs > rhs
    if op == ">=":
        return lhs >= rhs
    if op == "<":
        return lhs < rhs
    if op == "<=":
        return lhs <= rhs
    raise ValueError(f"Unsupported op: {op}")


def _summary_mode(args: argparse.Namespace) -> dict[str, Any]:
    criteria = load_yaml(args.criteria)
    candidate = read_json(args.candidate_summary)
    baseline = read_json(args.baseline_summary) if args.baseline_summary else {}

    checks = []
    passed_all = True
    for item in criteria.get("criteria", []):
        name = item["name"]
        metric = item["metric"]
        op = item["op"]
        candidate_value = _get_metric(candidate, metric)

        if "value" in item:
            reference_value = float(item["value"])
            reference_desc = f"constant:{reference_value}"
        elif "baseline_metric" in item:
            reference_value = _get_metric(baseline, item["baseline_metric"])
            reference_desc = f"baseline:{item['baseline_metric']}"
        else:
            raise ValueError(f"Criterion {name} needs either `value` or `baseline_metric`.")

        ok = _compare(candidate_value, reference_value, op)
        passed_all = passed_all and ok
        checks.append(
            {
                "name": name,
                "metric": metric,
                "candidate_value": candidate_value,
                "reference": reference_desc,
                "reference_value": reference_value,
                "op": op,
                "pass": ok,
            }
        )

    return {
        "mode": "summary",
        "criteria_file": args.criteria,
        "candidate_summary": args.candidate_summary,
        "baseline_summary": args.baseline_summary,
        "pass": passed_all,
        "checks": checks,
        "decision_note": criteria.get("decision_note"),
        "timebox_note": criteria.get("timebox_note"),
        "fallback_wedge": criteria.get("fallback_wedge"),
    }


def _per_pair_mode(args: argparse.Namespace) -> dict[str, Any]:
    released_rows = read_jsonl(args.released_per_pair)
    candidate_rows = read_jsonl(args.candidate_per_pair)

    k = int(args.k)
    key_localized = f"entity_pos_neg_localized_edit_diff_k{k}"
    key_local_flip = f"entity_pos_neg_neighbor_flip_rate_k{k}_local"
    key_far_flip = f"entity_pos_neg_neighbor_flip_rate_k{k}_far"
    keys = [key_localized, key_local_flip, key_far_flip]

    released = summarize_bootstrap_metrics(released_rows, keys, samples=args.bootstrap_samples, seed=0, ci=args.bootstrap_ci)
    candidate = summarize_bootstrap_metrics(candidate_rows, keys, samples=args.bootstrap_samples, seed=0, ci=args.bootstrap_ci)

    def mean_of(bundle, key):
        item = bundle.get(key) or {}
        value = item.get("mean")
        return float(value) if value is not None else None

    def ci_low_of(bundle, key):
        item = bundle.get(key) or {}
        value = item.get("ci_low")
        return float(value) if value is not None else None

    released_localized = mean_of(released, key_localized)
    candidate_localized = mean_of(candidate, key_localized)
    candidate_localized_ci_low = ci_low_of(candidate, key_localized)
    released_far = mean_of(released, key_far_flip)
    candidate_far = mean_of(candidate, key_far_flip)
    released_local = mean_of(released, key_local_flip)
    candidate_local = mean_of(candidate, key_local_flip)

    checks = {
        "entity_localized_diff_positive": bool(candidate_localized is not None and candidate_localized > args.min_entity_localized_diff),
        "entity_localized_diff_ci_positive": bool(candidate_localized_ci_low is not None and candidate_localized_ci_low > 0.0),
        "entity_localized_gain_vs_released": bool(candidate_localized is not None and released_localized is not None and candidate_localized >= released_localized + args.min_entity_localized_gain),
        "far_drift_not_worse": bool(candidate_far is not None and released_far is not None and candidate_far <= released_far + args.max_far_drift_increase),
        "local_flip_not_worse": bool(candidate_local is not None and released_local is not None and candidate_local >= released_local + args.min_local_flip_gain),
    }
    passed = all(checks.values())
    return {
        "mode": "per_pair",
        "criteria": {
            "k": k,
            "min_entity_localized_diff": args.min_entity_localized_diff,
            "min_entity_localized_gain": args.min_entity_localized_gain,
            "max_far_drift_increase": args.max_far_drift_increase,
            "min_local_flip_gain": args.min_local_flip_gain,
            "strict_reject_on_fail": bool(args.strict_reject_on_fail),
        },
        "released_per_pair": args.released_per_pair,
        "candidate_per_pair": args.candidate_per_pair,
        "released": released,
        "candidate": candidate,
        "checks": checks,
        "entity_lensgraph_go": passed,
        "recommended_wedge": "entity_lensgraph" if passed else "patch_phenomenon_negative_result",
    }


def main() -> None:
    args = parse_args()
    if args.criteria or args.candidate_summary:
        if not args.criteria or not args.candidate_summary:
            raise SystemExit("summary mode requires --criteria and --candidate-summary")
        report = _summary_mode(args)
    else:
        if not args.released_per_pair or not args.candidate_per_pair:
            raise SystemExit("per-pair mode requires --released-per-pair and --candidate-per-pair")
        report = _per_pair_mode(args)
    write_json(args.output_json, report)
    print(f"PASS={report.get('pass', report.get('entity_lensgraph_go'))} -> {args.output_json}")


if __name__ == "__main__":
    main()
