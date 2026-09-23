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

# Бэкбоны, предобученные на рентгенограммах (TorchXRayVision, Apache-2.0): имя -> веса
# библиотеки. «all» — общая модель, обученная сразу на семи наборах снимков грудной
# клетки; к DXA она ближе, чем ImageNet с его фотографиями.
XRV_BACKBONES = {
    "xrv-densenet121": "densenet121-res224-all",
}


def input_spec(backbone: str) -> str:
    """Нормировка входа, в которой учился бэкбон.

    `imagenet` — три одинаковых канала, среднее и разброс ImageNet;
    `xrv` — один канал в диапазоне [-1024, 1024], как у TorchXRayVision.
    """
    return "xrv" if backbone in XRV_BACKBONES else "imagenet"


class XrvDenseNetFeatures(nn.Module):
    """Свёрточная часть DenseNet из TorchXRayVision без встроенного ресайза.

    Штатный `features2` библиотеки сам сжимает вход до 224×224. Для DXA это вредно:
    малый вертел занимает несколько пикселей, поэтому разрешение задаётся параметрами
    обучения, а не бэкбоном. Свёртки DenseNet с этим справляются — на выходе глобальное
    усреднение.
    """

    def __init__(self, features: nn.Module) -> None:
        super().__init__()
        self.features = features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = nn.functional.relu(self.features(x))
        return nn.functional.adaptive_avg_pool2d(f, 1).flatten(1)


def _build_xrv(name: str, pretrained: bool) -> tuple[nn.Module, int]:
    import torchxrayvision as xrv

    net = xrv.models.DenseNet(weights=XRV_BACKBONES[name] if pretrained else None)
    return XrvDenseNetFeatures(net.features), net.classifier.in_features


def build_backbone(name: str, pretrained: bool = True) -> tuple[nn.Module, int]:
    import torchvision.models as tvm

    if name in XRV_BACKBONES:
        return _build_xrv(name, pretrained)
    if name not in BACKBONES:
        raise ValueError(f"неизвестный бэкбон {name}, доступны: {[*BACKBONES, *XRV_BACKBONES]}")
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


def load_backbone_weights(backbone: nn.Module, path: str) -> dict:
    """Подгружает веса бэкбона, сохранённые предобучением (`train.pretrain`).

    Архитектура должна совпадать: загрузка строгая, чтобы тихо не получить наполовину
    случайную сеть.
    """
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    backbone.load_state_dict(state, strict=True)
    return ckpt if isinstance(ckpt, dict) else {}


class DxaQualityNet(nn.Module):
    """Один бэкбон, по голове на каждую задачу из TASKS.

    Лосс считается только по задачам той области, к которой относится снимок (маска),
    поэтому обе области обучаются совместно и делят представление.
    """

    def __init__(
        self,
        backbone: str = "resnet18",
        pretrained: bool = True,
        dropout: float = 0.2,
        init_backbone: str | None = None,
    ) -> None:
        super().__init__()
        self.backbone, feat = build_backbone(backbone, pretrained)
        if init_backbone:
            load_backbone_weights(self.backbone, init_backbone)
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
