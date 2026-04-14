from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

try:
    from pycocotools import mask as mask_utils
except Exception:  # pragma: no cover - optional dependency
    mask_utils = None


def polygons_to_mask(polygons: Iterable[Iterable[Iterable[float]]], height: int, width: int) -> np.ndarray:
    """
    polygons:
        iterable of polygon instances
        each polygon instance is a list of rings
        each ring can be:
            - [x1, y1, x2, y2, ...]
            - [[x1, y1], [x2, y2], ...]
    """
    canvas = Image.new("L", (width, height), 0)
    drawer = ImageDraw.Draw(canvas)

    for instance in polygons:
        for polygon in instance:
            if not polygon:
                continue
            if isinstance(polygon[0], (int, float)):
                coords = [(float(polygon[i]), float(polygon[i + 1])) for i in range(0, len(polygon), 2)]
            else:
                coords = [(float(x), float(y)) for x, y in polygon]
            if len(coords) >= 3:
                drawer.polygon(coords, outline=1, fill=1)
    return np.array(canvas, dtype=np.uint8)


def decode_coco_segmentation(segmentation, height: int, width: int) -> np.ndarray:
    if segmentation is None:
        return np.zeros((height, width), dtype=np.uint8)

    if isinstance(segmentation, list):
        polygons = []
        for poly in segmentation:
            polygons.append([poly])
        return polygons_to_mask(polygons, height=height, width=width)

    if isinstance(segmentation, dict):
        if mask_utils is None:
            raise ImportError("pycocotools is required for RLE segmentation decoding.")
        if isinstance(segmentation.get("counts"), list):
            rle = mask_utils.frPyObjects(segmentation, height, width)
        else:
            rle = segmentation
        mask = mask_utils.decode(rle)
        if mask.ndim == 3:
            mask = np.any(mask, axis=2).astype(np.uint8)
        return mask.astype(np.uint8)

    raise TypeError(f"Unsupported segmentation type: {type(segmentation)}")


def save_binary_mask(mask: np.ndarray | torch.Tensor, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(mask, torch.Tensor):
        mask_np = mask.detach().cpu().numpy()
    else:
        mask_np = np.asarray(mask)

    mask_np = (mask_np > 0).astype(np.uint8) * 255
    Image.fromarray(mask_np).save(path)


def load_binary_mask(path: str | Path) -> torch.Tensor:
    arr = np.array(Image.open(path).convert("L"), dtype=np.uint8)
    tensor = torch.from_numpy((arr > 0).astype(np.float32))
    return tensor.unsqueeze(0)


def _get_transform_ops(transform):
    if hasattr(transform, "transforms"):
        return list(transform.transforms)
    return [transform]


def _normalize_size(size) -> tuple[int, int]:
    if isinstance(size, int):
        return (size, size)
    if isinstance(size, (tuple, list)) and len(size) == 2:
        return (int(size[0]), int(size[1]))
    raise ValueError(f"Unsupported transform size specifier: {size}")


def _resize_pil(mask_pil: Image.Image, size) -> Image.Image:
    # torchvision Resize on PIL expects size=(h, w), PIL expects (w, h)
    h, w = _normalize_size(size)
    return mask_pil.resize((w, h), resample=Image.NEAREST)


def _center_crop_pil(mask_pil: Image.Image, size) -> Image.Image:
    th, tw = _normalize_size(size)
    w, h = mask_pil.size
    left = max(int(round((w - tw) / 2.0)), 0)
    top = max(int(round((h - th) / 2.0)), 0)
    right = min(left + tw, w)
    bottom = min(top + th, h)
    return mask_pil.crop((left, top, right, bottom))


def apply_geometric_ops_to_mask(mask_pil: Image.Image, image_transform) -> torch.Tensor:
    """
    Mirrors the common deterministic geometric ops used by timm / torchvision eval transforms.
    Supported ops:
      - Resize
      - CenterCrop
      - ToTensor / PILToTensor / MaybeToTensor
      - Normalize (ignored for masks)
    """
    ops = _get_transform_ops(image_transform)
    img = mask_pil

    for op in ops:
        name = op.__class__.__name__.lower()

        if "resize" in name:
            size = getattr(op, "size", None)
            if size is None:
                raise ValueError(f"Could not infer resize size from transform {op}")
            img = _resize_pil(img, size=size)

        elif "centercrop" in name:
            size = getattr(op, "size", None)
            if size is None:
                raise ValueError(f"Could not infer crop size from transform {op}")
            img = _center_crop_pil(img, size=size)

        elif name in {"totensor", "piltotensor", "maybetotensor"}:
            arr = np.array(img, dtype=np.float32) / 255.0
            tensor = torch.from_numpy(arr).unsqueeze(0)
            return (tensor > 0.5).float()

        elif "normalize" in name:
            continue

        else:
            # Conservative fallback: ignore non-geometric ops that often appear in eval transforms.
            if name in {"convertimagedtype", "totensorv2"}:
                continue
            raise NotImplementedError(
                f"Unsupported mask transform op '{op.__class__.__name__}'. "
                "Use a deterministic eval transform or adapt apply_geometric_ops_to_mask."
            )

    arr = np.array(img, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).unsqueeze(0)
    return (tensor > 0.5).float()


def patchify_soft_mask(mask: torch.Tensor, patch_grid: tuple[int, int], normalize: bool = False, eps: float = 1e-6) -> torch.Tensor:
    """
    mask: (B,1,H,W) or (1,H,W) or (H,W)
    returns: (B, Gh*Gw) soft occupancy values
    """
    if mask.ndim == 2:
        mask = mask.unsqueeze(0).unsqueeze(0)
    elif mask.ndim == 3:
        mask = mask.unsqueeze(0) if mask.shape[0] != 1 else mask.unsqueeze(0)
    elif mask.ndim != 4:
        raise ValueError(f"Unexpected mask shape: {tuple(mask.shape)}")

    mask = mask.float()
    pooled = F.adaptive_avg_pool2d(mask, patch_grid)
    flat = pooled.flatten(1)

    if normalize:
        denom = flat.sum(dim=1, keepdim=True).clamp_min(eps)
        flat = flat / denom
    return flat


def masked_mean(values: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    values: (B, N) or (B, 1, H, W)
    mask: same spatial layout or broadcastable
    """
    if values.ndim == 4:
        values = values.flatten(1)
    if mask.ndim == 4:
        mask = mask.flatten(1)
    if mask.ndim == 3:
        mask = mask.flatten(1)

    values = values.float()
    mask = mask.float()
    numer = (values * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp_min(eps)
    return numer / denom


def mask_iou(mask_a: torch.Tensor, mask_b: torch.Tensor, eps: float = 1e-6) -> float:
    a = (mask_a > 0).float()
    b = (mask_b > 0).float()
    inter = (a * b).sum().item()
    union = ((a + b) > 0).float().sum().item()
    return float(inter / max(union, eps))
