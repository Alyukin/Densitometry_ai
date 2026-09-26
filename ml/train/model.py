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


def fp16_cudnn_ok(device: torch.device) -> bool:
    """Даёт ли cuDNN конечный результат в fp16 на этой видеокарте.

    На GTX 16xx (проверено на 1660 Ti, cuDNN 9.10) свёртка 3×3 в fp16 через cuDNN
    возвращает NaN на всех выходах, и обучение с AMP молча идёт по NaN. Проверка —
    несколько свёрток на случайном входе; глобальный генератор случайных чисел не трогается.
    """
    if device.type != "cuda" or not torch.backends.cudnn.enabled:
        return True
    gen = torch.Generator(device=device).manual_seed(0)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        for cin, cout, k in ((64, 64, 3), (128, 32, 3), (3, 64, 7), (64, 128, 1)):
            x = torch.randn(16, cin, 96, 80, device=device, generator=gen)
            w = torch.randn(cout, cin, k, k, device=device, generator=gen) * (2.0 / (cin * k * k)) ** 0.5
            if not torch.isfinite(nn.functional.conv2d(x, w, padding=k // 2)).all():
                return False
    return True


def setup_amp(device: torch.device, amp: bool) -> str:
    """Готовит смешанную точность и возвращает, какими ядрами считаются свёртки.

    Точность при этом не меняется — fp16 под autocast, как объявлено в протоколе (К2).
    Если cuDNN в fp16 даёт NaN, он отключается и свёртки идут встроенными ядрами
    PyTorch: медленнее, но результат конечный. Решение зависит только от видеокарты и
    одно на все конфигурации.
    """
    if not amp or device.type != "cuda":
        return "fp32" if not amp else f"autocast на {device.type}"
    if fp16_cudnn_ok(device):
        return "fp16, свёртки cuDNN"
    torch.backends.cudnn.enabled = False
    return "fp16, свёртки без cuDNN: cuDNN в fp16 на этой видеокарте даёт NaN"


def check_finite(loss: torch.Tensor) -> None:
    """Loss NaN или inf — дальше учиться бессмысленно: веса и выбор эпохи будут мусором."""
    if not torch.isfinite(loss):
        raise FloatingPointError(
            f"loss = {float(loss.detach())}: не число уже в прямом проходе. Обучение остановлено. "
            "Если включён --amp, проверьте fp16 на этой видеокарте (train.model.setup_amp)."
        )


def masked_bce(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    pos_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    loss = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none", pos_weight=pos_weight)
    denom = mask.sum().clamp(min=1.0)
    return (loss * mask).sum() / denom
