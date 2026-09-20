"""Модель: общий бэкбон + многозадачная голова (позвоночник / бедро)."""

from __future__ import annotations

import torch
from torch import nn

from train.tasks import TASKS

BACKBONES = {
    "resnet18": ("resnet18", "ResNet18_Weights"),
    "resnet34": ("resnet34", "ResNet34_Weights"),
    "resnet50": ("resnet50", "ResNet50_Weights"),
    "efficientnet_b0": ("efficientnet_b0", "EfficientNet_B0_Weights"),
}


def build_backbone(name: str, pretrained: bool = True) -> tuple[nn.Module, int]:
    import torchvision.models as tvm

    if name not in BACKBONES:
        raise ValueError(f"неизвестный бэкбон {name}, доступны: {list(BACKBONES)}")
    fn_name, weights_enum = BACKBONES[name]
    weights = getattr(tvm, weights_enum).DEFAULT if pretrained else None
    net = getattr(tvm, fn_name)(weights=weights)
    if hasattr(net, "fc"):
        feat = net.fc.in_features
        net.fc = nn.Identity()
    else:  # efficientnet
        feat = net.classifier[-1].in_features
        net.classifier = nn.Identity()
    return net, feat


class DxaQualityNet(nn.Module):
    """Один бэкбон, по голове на каждую задачу из TASKS.

    Лосс считается только по задачам той области, к которой относится снимок (маска),
    поэтому обе области обучаются совместно и делят представление.
    """

    def __init__(self, backbone: str = "resnet18", pretrained: bool = True, dropout: float = 0.2) -> None:
        super().__init__()
        self.backbone, feat = build_backbone(backbone, pretrained)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(feat, len(TASKS))
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.dropout(self.backbone(x)))


def masked_bce(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    pos_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    loss = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none", pos_weight=pos_weight)
    denom = mask.sum().clamp(min=1.0)
    return (loss * mask).sum() / denom
