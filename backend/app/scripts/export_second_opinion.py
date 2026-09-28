"""Экспорт замороженного DenseNet121 (TorchXRayVision) в ONNX для «второго мнения».

Сеть не обучалась на наших данных: это публичные веса TorchXRayVision (Apache-2.0),
которые снимают со снимка 1024 признака. Поверх них в сервисе работает логистическая
регрессия (`second_opinion/xrv_probe.json`), обученная в `ml/train/second_opinion.py`.

Выход в ONNX, чтобы сервису не нужен был torch: `onnxruntime` весит в десятки раз меньше.
Скрипт запускается при сборке Docker-образа (отдельная стадия с torch) и вручную для
разработки без Docker:

    pip install torch==2.9.1 torchvision==0.24.1 torchxrayvision==1.5.4 onnx
    python -m app.scripts.export_second_opinion --out app/processing/second_opinion/xrv_densenet121.onnx

Архитектура повторяет `ml/train/model.py::XrvDenseNetFeatures`: свёрточная часть DenseNet,
ReLU, глобальное усреднение. Совпадение признаков с обучением проверяется сравнением на
всех 246 снимках (`ml/train/second_opinion.py --check-onnx`).
"""

from __future__ import annotations

import argparse
from pathlib import Path

WEIGHTS = "densenet121-res224-all"
HEIGHT, WIDTH = 384, 320  # разрешение, на котором считались признаки при обучении


def export(out: Path) -> None:
    import torch
    import torchxrayvision as xrv
    from torch import nn

    class Features(nn.Module):
        def __init__(self, features: nn.Module) -> None:
            super().__init__()
            self.features = features

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            f = nn.functional.relu(self.features(x))
            return nn.functional.adaptive_avg_pool2d(f, 1).flatten(1)

    net = Features(xrv.models.DenseNet(weights=WEIGHTS).features).eval()
    out.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        torch.onnx.export(
            net,
            (torch.zeros(1, 1, HEIGHT, WIDTH),),
            str(out),
            input_names=["image"],
            output_names=["features"],
            dynamic_axes={"image": {0: "batch"}, "features": {0: "batch"}},
            opset_version=17,
            dynamo=False,
        )
    print(f"{out} ({out.stat().st_size / 2**20:.1f} МБ)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    export(ap.parse_args().out)


if __name__ == "__main__":
    main()
