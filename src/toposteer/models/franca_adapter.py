from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode


@dataclass
class FrancaOutputs:
    raw_patch_tokens: torch.Tensor
    rasa_patch_tokens: torch.Tensor


class FrancaAdapter(nn.Module):
    """
    Frozen Franca wrapper exposing the two dense streams needed by TopoSteer.

    raw_patch_tokens:
        Franca's spatially faithful normalized patch tokens.
    rasa_patch_tokens:
        RASA-debiased patch tokens used as the semantic graph substrate.
    """

    def __init__(
        self,
        arch: Literal["vitb14", "vitl14", "vitg14"] = "vitb14",
        weights: str = "IN21K",
        device: str | torch.device = "cpu",
        image_size: int | None = None,
        pretrained: bool = True,
        use_rasa_head: bool = True,
    ) -> None:
        super().__init__()
        os.environ.setdefault("XFORMERS_DISABLED", "1")

        from franca.hub.backbones import franca_vitb14, franca_vitg14, franca_vitl14

        arch_fns = {
            "vitb14": franca_vitb14,
            "vitl14": franca_vitl14,
            "vitg14": franca_vitg14,
        }
        if arch not in arch_fns:
            raise ValueError(f"Unsupported Franca architecture: {arch}")

        self.arch = arch
        self.weights = weights
        self.use_rasa_head = use_rasa_head

        kwargs = {
            "pretrained": pretrained,
            "weights": weights,
            "use_rasa_head": False,
        }
        if image_size is not None:
            kwargs["img_size"] = int(image_size)

        self.model = arch_fns[arch](**kwargs)
        if use_rasa_head:
            self._attach_rasa_head(pretrained=pretrained)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        self.device_name = torch.device(device)
        self.model.to(self.device_name)

        self._patch_size = self._infer_patch_size()
        self._image_size = int(image_size or self._infer_image_size())

    def _infer_patch_size(self) -> int:
        patch_size = getattr(self.model, "patch_size", None)
        if patch_size is None and hasattr(self.model, "patch_embed"):
            patch_size = getattr(self.model.patch_embed, "patch_size", None)
        if isinstance(patch_size, tuple):
            return int(patch_size[0])
        if patch_size is not None:
            return int(patch_size)
        return 14

    def _infer_image_size(self) -> int:
        img_size = None
        if hasattr(self.model, "patch_embed"):
            img_size = getattr(self.model.patch_embed, "img_size", None)
        if isinstance(img_size, tuple):
            return int(img_size[0])
        if img_size is not None:
            return int(img_size)
        if self.arch == "vitb14" and self.weights != "DINOV2_IN21K":
            return 518
        return 224

    def _attach_rasa_head(self, pretrained: bool) -> None:
        from franca.hub.backbones import _FRANCA_BASE_URL, Weights, _make_rasa_model_name
        from franca.hub.utils import load_state_dict_from_url
        from rasa.src.rasa_head import RASAHead

        arch_names = {
            "vitb14": "vit_base",
            "vitl14": "vit_large",
            "vitg14": "vit_giant2",
        }
        weights = self.weights
        if isinstance(weights, str):
            weights = Weights[weights]

        n_pos_layers = 8
        state_dict = None
        if pretrained:
            rasa_model_name = _make_rasa_model_name(arch_names[self.arch], self._infer_patch_size(), weights.value)
            rasa_url = _FRANCA_BASE_URL + f"/{rasa_model_name}.pth"
            state_dict = load_state_dict_from_url(rasa_url, map_location="cpu", weights_only=True)
            layer_ids = [
                int(k.split(".")[1])
                for k in state_dict
                if k.startswith("pre_pos_layers.") and k.endswith(".weight")
            ]
            if layer_ids:
                n_pos_layers = max(layer_ids) + 1

        rasa_head = RASAHead(input_dim=self.model.embed_dim, n_pos_layers=n_pos_layers, pos_out_dim=2)
        if state_dict is not None:
            rasa_head.load_state_dict(state_dict, strict=True)
        self.model.rasa_head = rasa_head

    @property
    def patch_size(self) -> int:
        return self._patch_size

    @property
    def image_size(self) -> tuple[int, int]:
        return (self._image_size, self._image_size)

    @property
    def patch_grid(self) -> tuple[int, int]:
        return (self._image_size // self._patch_size, self._image_size // self._patch_size)

    @property
    def feature_dim(self) -> int:
        return int(getattr(self.model, "embed_dim"))

    def get_transforms(self):
        return transforms.Compose(
            [
                transforms.Resize(self.image_size, interpolation=InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )

    @torch.no_grad()
    def forward_outputs(self, images: torch.Tensor) -> FrancaOutputs:
        images = images.to(self.device_name)
        feats = self.model.forward_features(images, use_rasa_head=self.use_rasa_head)
        raw = feats["x_norm_patchtokens"]
        rasa = feats.get("patch_token_rasa")
        if rasa is None:
            rasa = raw
        return FrancaOutputs(raw_patch_tokens=raw, rasa_patch_tokens=rasa)
