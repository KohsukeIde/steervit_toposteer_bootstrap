from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from tqdm import tqdm


@dataclass
class CachedTextBatch:
    text_feats: torch.Tensor
    attn_mask: torch.Tensor


class TextFeatureCache:
    """
    Very small helper for caching prompt encodings.
    The wrapper model still decides how to consume these tensors.
    """

    def __init__(self):
        self._store: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

    def __contains__(self, prompt: str) -> bool:
        return prompt in self._store

    def get(self, prompt: str) -> tuple[torch.Tensor, torch.Tensor]:
        return self._store[prompt]

    def add(self, prompt: str, text_feats: torch.Tensor, attn_mask: torch.Tensor) -> None:
        self._store[prompt] = (text_feats.cpu(), attn_mask.cpu())

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self._store, path)

    @classmethod
    def load(cls, path: str | Path) -> "TextFeatureCache":
        cache = cls()
        cache._store = torch.load(path, map_location="cpu")
        return cache

    def unique_prompts(self) -> list[str]:
        return sorted(self._store.keys())


def build_text_cache(model, prompts: Iterable[str], batch_size: int = 32, device: str = "cuda") -> TextFeatureCache:
    """
    model must expose encode_texts(list[str]) -> (text_feats, attn_mask).
    """
    unique_prompts = sorted({p for p in prompts if p})
    cache = TextFeatureCache()
    for start in tqdm(range(0, len(unique_prompts), batch_size), desc="Caching text prompts"):
        batch = unique_prompts[start : start + batch_size]
        text_feats, attn_mask = model.encode_texts(batch, device=device)
        for i, prompt in enumerate(batch):
            cache.add(prompt, text_feats[i], attn_mask[i])
    return cache
