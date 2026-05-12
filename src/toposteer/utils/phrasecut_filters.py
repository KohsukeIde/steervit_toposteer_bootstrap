from __future__ import annotations

import re
from typing import Any

COLOR_ALIASES = {
    "red": "red",
    "blue": "blue",
    "green": "green",
    "yellow": "yellow",
    "orange": "orange",
    "purple": "purple",
    "pink": "pink",
    "brown": "brown",
    "black": "black",
    "white": "white",
    "gray": "grey",
    "grey": "grey",
    "silver": "silver",
    "gold": "gold",
    "beige": "beige",
}
MATERIAL_ALIASES = {
    "wood": "wood",
    "wooden": "wood",
    "metal": "metal",
    "metallic": "metal",
    "plastic": "plastic",
    "glass": "glass",
    "paper": "paper",
    "cardboard": "cardboard",
    "leather": "leather",
    "fabric": "fabric",
    "cloth": "cloth",
    "rubber": "rubber",
    "stone": "stone",
    "concrete": "concrete",
    "ceramic": "ceramic",
}
SIZE_ALIASES = {
    "small": "small",
    "little": "small",
    "tiny": "small",
    "large": "large",
    "big": "large",
    "huge": "large",
    "tall": "tall",
    "short": "short",
    "long": "long",
    "wide": "wide",
    "narrow": "narrow",
}
STATE_ALIASES = {
    "open": "open",
    "closed": "closed",
    "on": "on",
    "off": "off",
    "empty": "empty",
    "full": "full",
    "broken": "broken",
    "clean": "clean",
    "dirty": "dirty",
    "folded": "folded",
    "unfolded": "unfolded",
    "ripe": "ripe",
    "unripe": "unripe",
    "wet": "wet",
    "dry": "dry",
}
SPATIAL_ALIASES = {
    "left": "left",
    "right": "right",
    "top": "top",
    "bottom": "bottom",
    "front": "front",
    "back": "back",
    "center": "center",
    "middle": "center",
    "upper": "upper",
    "lower": "lower",
}


def normalize_text(text: str | None) -> str:
    if not text:
        return ""
    text = text.lower().strip()
    text = text.replace("-", " ")
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_object_name(name: str | None) -> str | None:
    norm = normalize_text(name)
    return norm or None


def canonicalize_attribute_name(raw: str) -> dict[str, str] | None:
    attr = normalize_text(raw)
    if not attr:
        return None
    for prefix in ("light ", "dark ", "bright "):
        if attr.startswith(prefix):
            tail = attr[len(prefix) :]
            if tail in COLOR_ALIASES:
                return {"name": COLOR_ALIASES[tail], "type": "color", "raw": attr}
    if attr in COLOR_ALIASES:
        return {"name": COLOR_ALIASES[attr], "type": "color", "raw": attr}
    if attr in MATERIAL_ALIASES:
        return {"name": MATERIAL_ALIASES[attr], "type": "material", "raw": attr}
    if attr in SIZE_ALIASES:
        return {"name": SIZE_ALIASES[attr], "type": "size", "raw": attr}
    if attr in STATE_ALIASES:
        return {"name": STATE_ALIASES[attr], "type": "state", "raw": attr}
    if attr in SPATIAL_ALIASES:
        return {"name": SPATIAL_ALIASES[attr], "type": "spatial", "raw": attr}
    return None


def extract_canonical_attrs(record: dict[str, Any]) -> list[dict[str, str]]:
    meta = record.get("meta", {}) or {}
    attrs = meta.get("attributes") or []
    seen: set[tuple[str, str]] = set()
    out: list[dict[str, str]] = []
    for attr in attrs:
        canon = canonicalize_attribute_name(attr)
        if canon is None:
            continue
        key = (canon["type"], canon["name"])
        if key in seen:
            continue
        out.append(canon)
        seen.add(key)
    return out


def candidate_attr_types(record: dict[str, Any]) -> set[str]:
    return {item["type"] for item in extract_canonical_attrs(record)}


def has_relations(record: dict[str, Any]) -> bool:
    meta = record.get("meta", {}) or {}
    return bool(meta.get("relation_descriptions") or [])


def object_name_of(record: dict[str, Any]) -> str | None:
    meta = record.get("meta", {}) or {}
    return normalize_object_name(meta.get("object_name"))
