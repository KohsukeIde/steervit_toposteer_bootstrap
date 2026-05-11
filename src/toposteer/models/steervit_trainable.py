from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from steervit import SteerViT
except Exception:  # pragma: no cover
    from steervit.model import SteerViT  # type: ignore


@dataclass
class ForwardOutputs:
    tokens: torch.Tensor
    dense_tokens: torch.Tensor
    logits: torch.Tensor
    probs: torch.Tensor


class SteerViTTrainable(nn.Module):
    """
    Thin wrapper around the public SteerViT release.

    Key idea:
    - use the upstream checkpoint loader,
    - re-enable gradients only on modules we want to fine-tune,
    - expose a training-time forward path that returns patch logits and patch probabilities.
    """

    def __init__(
        self,
        checkpoint: str,
        device: str | torch.device = "cpu",
        trainable_modules: Iterable[str] | None = None,
        base_checkpoint: str | None = None,
    ):
        super().__init__()
        train_state = self._load_train_checkpoint_if_needed(checkpoint)
        release_checkpoint = base_checkpoint
        if train_state is not None:
            release_checkpoint = release_checkpoint or train_state.get("base_checkpoint")
            release_checkpoint = release_checkpoint or train_state.get("config", {}).get("base_checkpoint")
            if release_checkpoint is None:
                raise ValueError(
                    "TopoSteer training checkpoints require --base-checkpoint so the SteerViT "
                    "architecture can be initialized before loading model_state_dict."
                )
        else:
            release_checkpoint = checkpoint

        self.model = self._load_release_model(str(release_checkpoint), device=device)
        self.device_name = torch.device(device)
        self.trainable_modules = tuple(trainable_modules or ("gated_cross_attn", "connector", "lin_seg_head"))
        self._freeze_all()
        if train_state is not None:
            missing, unexpected = self.load_state_dict(train_state["model_state_dict"], strict=False)
            if unexpected:
                raise RuntimeError(f"Unexpected keys while loading training checkpoint: {unexpected[:20]}")
            if missing:
                # Missing keys are acceptable only for forward-compatible additions. Surface a compact warning.
                print({"training_checkpoint_missing_keys": missing[:20], "num_missing": len(missing)})
        self._unfreeze_requested_modules()

    @staticmethod
    def _load_train_checkpoint_if_needed(checkpoint: str) -> dict | None:
        path = Path(checkpoint)
        if not path.is_file():
            return None
        state = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(state, dict) and "model_state_dict" in state:
            return state
        return None

    @staticmethod
    def _load_release_model(checkpoint: str, device: str | torch.device | None = None):
        path = Path(checkpoint)
        if not path.is_file():
            return SteerViT.from_pretrained(checkpoint, device=device)

        state = torch.load(path, map_location="cpu", weights_only=False)
        if not (isinstance(state, dict) and "config" in state and "state_dict" in state):
            raise ValueError(f"Expected a SteerViT release checkpoint at {checkpoint}.")

        model = SteerViT(state["config"])
        model.load_state_dict(state["state_dict"], strict=False)
        for param in model.parameters():
            param.requires_grad = False
        model.eval()
        if device is not None:
            model = model.to(device)
        return model

    def _freeze_all(self) -> None:
        for param in self.model.parameters():
            param.requires_grad = False

    def _unfreeze_requested_modules(self) -> None:
        for name in self.trainable_modules:
            if name == "connector":
                for p in self.model.connector.parameters():
                    p.requires_grad = True
            elif name in {"lin_seg_head", "segmentation_head"}:
                for p in self.model.lin_seg_head.parameters():
                    p.requires_grad = True
            elif name == "gated_cross_attn":
                for blk in self.model.vision_model.trunk.blocks:
                    gca = getattr(blk, "gated_cross_attn", None)
                    if gca is not None:
                        for p in gca.parameters():
                            p.requires_grad = True
            elif name == "vision_backbone":
                for p in self.model.vision_model.parameters():
                    p.requires_grad = True
            elif name == "text_model":
                for p in self.model.text_model.parameters():
                    p.requires_grad = True
            else:
                raise ValueError(f"Unknown trainable module specifier: {name}")

    @property
    def patch_size(self) -> int:
        return self.model.patch_size

    @property
    def image_size(self) -> tuple[int, int]:
        return self.model.image_size

    @property
    def patch_grid(self) -> tuple[int, int]:
        return (self.image_size[0] // self.patch_size, self.image_size[1] // self.patch_size)

    @property
    def num_prefix_tokens(self) -> int:
        return self.model.vision_model.trunk.num_prefix_tokens

    @property
    def feature_dim(self) -> int:
        return self.model.feature_dim

    def get_transforms(self):
        return self.model.get_transforms()

    def set_gate_factor(self, factor: float) -> None:
        self.model.set_gate_factor(factor)

    def encode_texts(
        self,
        texts: list[str],
        device: str | torch.device | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not texts:
            raise ValueError("encode_texts received an empty text batch.")

        device = device or self.device_name
        roberta_dict = self.model.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        roberta_dict = {k: v.to(device) for k, v in roberta_dict.items()}
        text_feats = self.model.text_model(**roberta_dict).last_hidden_state
        attn_mask = roberta_dict["attention_mask"].bool()
        text_feats = F.normalize(text_feats, dim=-1)
        text_feats = self.model.connector(text_feats)

        img_tokens = torch.ones(text_feats.size(0), self.model.num_img_tokens, dtype=torch.bool, device=device)
        attn_mask = torch.cat([img_tokens, attn_mask], dim=-1)
        return text_feats, attn_mask

    def forward_tokens(
        self,
        images: torch.Tensor,
        texts: list[str] | None = None,
        text_feats: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        images = images.to(self.device_name)
        if texts is not None:
            if images.size(0) != len(texts):
                raise ValueError("Batch size of images and texts must match.")
            text_feats, attn_mask = self.encode_texts(texts, device=self.device_name)
        elif text_feats is not None:
            text_feats = text_feats.to(self.device_name)
            attn_mask = attn_mask.to(self.device_name) if attn_mask is not None else None
        else:
            text_feats = None
            attn_mask = None

        return self.model.vision_model(images, text_feats, attn_mask=attn_mask)

    def forward_outputs(
        self,
        images: torch.Tensor,
        texts: list[str] | None = None,
        text_feats: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
    ) -> ForwardOutputs:
        tokens = self.forward_tokens(images, texts=texts, text_feats=text_feats, attn_mask=attn_mask)
        dense = tokens[:, self.num_prefix_tokens :, :]
        logits = self.model.lin_seg_head(dense).squeeze(-1)
        probs = torch.softmax(logits, dim=1)
        return ForwardOutputs(tokens=tokens, dense_tokens=dense, logits=logits, probs=probs)

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def train(self, mode: bool = True):
        super().train(mode)
        # keep the text encoder frozen and in eval mode by default
        if hasattr(self.model, "text_model"):
            self.model.text_model.eval()
            for p in self.model.text_model.parameters():
                if not p.requires_grad:
                    p.requires_grad = False
        return self
