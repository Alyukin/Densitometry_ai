"""Бэкбоны и нормировка входа.

Главное, что здесь проверяется: вход сети при обучении и при инференсе готовится одной
функцией и в той нормировке, в которой учился бэкбон. Ошибка в этом месте не роняет
программу — модель просто тихо получает не то, на чём училась.

Без torch и torchxrayvision (они нужны только для обучения) тесты пропускаются.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ml"))

from train.dataset import to_tensor  # noqa: E402
from train.model import (  # noqa: E402
    XRV_BACKBONES,
    DxaQualityNet,
    build_backbone,
    input_spec,
    load_backbone_weights,
)
from train.tasks import TASKS  # noqa: E402


def test_imagenet_input_has_three_normalised_channels() -> None:
    x = to_tensor(np.full((8, 6), 0.449, dtype=np.float32), "imagenet")
    assert x.shape == (3, 8, 6)
    assert torch.allclose(x, torch.zeros_like(x), atol=1e-6)  # среднее ImageNet -> 0


def test_xrv_input_is_single_channel_in_xrv_range() -> None:
    """TorchXRayVision ждёт один канал в диапазоне [-1024, 1024]."""
    a = np.array([[0.0, 0.5, 1.0]], dtype=np.float32)
    x = to_tensor(a, "xrv")
    assert x.shape == (1, 1, 3)
    assert x.flatten().tolist() == pytest.approx([-1024.0, 0.0, 1024.0])


def test_unknown_normalisation_is_rejected() -> None:
    with pytest.raises(ValueError, match="нормировка"):
        to_tensor(np.zeros((2, 2), dtype=np.float32), "что-то")


def test_input_spec_follows_the_backbone() -> None:
    assert input_spec("resnet18") == "imagenet"
    assert input_spec("efficientnet_b0") == "imagenet"
    for name in XRV_BACKBONES:
        assert input_spec(name) == "xrv"


def test_unknown_backbone_lists_the_available_ones() -> None:
    with pytest.raises(ValueError, match="xrv-densenet121"):
        build_backbone("vgg11", pretrained=False)


# --- сам бэкбон TorchXRayVision (веса не качаются: pretrained=False) ------------


@pytest.fixture(scope="module")
def xrv_net():
    pytest.importorskip("torchxrayvision")
    return DxaQualityNet("xrv-densenet121", pretrained=False).eval()


def test_xrv_backbone_gives_one_output_per_task(xrv_net) -> None:
    with torch.no_grad():
        out = xrv_net(torch.zeros(2, 1, 384, 320))
    assert out.shape == (2, len(TASKS))


def test_xrv_backbone_keeps_the_requested_resolution() -> None:
    """Штатный features2 библиотеки сжимает вход до 224 — наш бэкбон не должен.

    Проверяем косвенно: карта признаков на входе 384×320 больше, чем на 224×224.
    Будь внутри принудительный ресайз, размеры совпали бы.
    """
    pytest.importorskip("torchxrayvision")
    backbone, feat = build_backbone("xrv-densenet121", pretrained=False)
    assert feat == 1024
    with torch.no_grad():
        big = backbone.features(torch.zeros(1, 1, 384, 320)).shape[-2:]
        small = backbone.features(torch.zeros(1, 1, 224, 224)).shape[-2:]
    assert big != small
    assert big[0] > small[0]


def test_pretrained_backbone_weights_are_loaded(tmp_path: Path) -> None:
    """Веса после предобучения (train.pretrain) доходят до модели, а не теряются молча."""
    src, _ = build_backbone("resnet18", pretrained=False)
    with torch.no_grad():
        for p in src.parameters():
            p.fill_(0.123)
    path = tmp_path / "backbone.pt"
    torch.save({"backbone": "resnet18", "state_dict": src.state_dict()}, path)

    net = DxaQualityNet("resnet18", pretrained=False, init_backbone=str(path))
    first = next(net.backbone.parameters())
    assert torch.allclose(first, torch.full_like(first, 0.123))


def test_wrong_architecture_is_refused(tmp_path: Path) -> None:
    src, _ = build_backbone("resnet18", pretrained=False)
    path = tmp_path / "backbone.pt"
    torch.save({"state_dict": src.state_dict()}, path)
    other, _ = build_backbone("resnet34", pretrained=False)
    with pytest.raises(RuntimeError):
        load_backbone_weights(other, str(path))
