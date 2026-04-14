from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = value
    return result


def parse_override_pairs(pairs: list[str]) -> dict[str, Any]:
    """
    Parses CLI overrides of the form:
        loss.cf_weight=1.0
        batch_size=8
        data.family_whitelist=["attr","part"]
    """
    import ast

    root: dict[str, Any] = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"Override '{pair}' must be in KEY=VALUE format.")
        key, raw_value = pair.split("=", 1)
        try:
            value = ast.literal_eval(raw_value)
        except Exception:
            value = raw_value

        cursor = root
        parts = key.split(".")
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})
        cursor[parts[-1]] = value
    return root
