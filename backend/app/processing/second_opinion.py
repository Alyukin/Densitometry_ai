"""«Второе мнение» ИИ-модели для бедра — справочно, в вердикт не входит.

Гибрид правил и модели (этап 7, OVERVIEW.md, К2 от 27.09) у правил не выиграл: вердикт
выносят правила. Модель показывается рядом, как справочная проверка, и только для
бедра — там она вровень с правилами (ROC AUC 0.67–0.80), на позвоночнике почти случайна.

Модель — конфигурация 4: замороженный DenseNet121 с публичными весами TorchXRayVision
(на наших данных не обучался) снимает со снимка 1024 признака, поверх — логистическая
регрессия на каждую задачу бедра, обученная на 148 размеченных снимках
(`ml/train/second_opinion.py`, коэффициенты — `second_opinion_model/xrv_probe.json`).
Сеть выполняется в `onnxruntime` на CPU (`app/scripts/export_second_opinion.py`).

Если файлов модели нет или `onnxruntime` не установлен, второе мнение просто выключено:
вердикт от него не зависит.
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import asdict
from pathlib import Path

import numpy as np
from PIL import Image

from app.processing.dxaqc.rules import Check

logger = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).with_name("second_opinion_model")
ONNX_NAME = "xrv_densenet121.onnx"
PROBE_NAME = "xrv_probe.json"
SOURCE = "ИИ-модель"


def preprocess(arr: np.ndarray, height: int, width: int) -> np.ndarray:
    """Кадр -> вход сети так же, как при обучении.

    `dxa.build_dataset.export_png` (8 бит), затем `train.dataset.DxaDataset` (билинейно до
    384×320, [0, 1]) и `to_tensor(norm="xrv")` (шкала TorchXRayVision [-1024, 1024]).
    """
    a = np.asarray(arr)
    if a.dtype != np.uint8:
        f = a.astype(np.float32)
        lo, hi = np.percentile(f, (0.5, 99.5))
        a = (np.clip((f - lo) / max(hi - lo, 1e-6), 0, 1) * 255).astype(np.uint8)
    img = Image.fromarray(a).convert("L").resize((width, height), Image.BILINEAR)
    x = np.asarray(img, dtype=np.float32) / 255.0
    return ((2.0 * x - 1.0) * 1024.0)[None, None].astype(np.float32)


def _logit(p: float) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return float(np.log(p / (1 - p)))


def score(features: np.ndarray, task: dict) -> float:
    """Вероятность задачи, сдвинутая так, что порог модели приходится на 0.5 (как у правил)."""
    z = (np.asarray(features, dtype=np.float64) - np.asarray(task["mean"])) / np.asarray(task["scale"])
    logit = float(z @ np.asarray(task["coef"]) + task["intercept"])
    return float(1.0 / (1.0 + np.exp(-(logit - _logit(task["threshold"])))))


def make_check(key: str, task: dict, value: float) -> dict:
    label = task["label"][:1].upper() + task["label"][1:]
    return asdict(
        Check(
            rule_id=f"nn_{key}",
            violation=task.get("violation") or "",
            fired=value >= 0.5,
            value=round(value, 4),
            threshold=0.5,
            op=">",
            score=round(value, 4),
            title=f"ИИ-модель: {label}",
            measured=f"Оценка ИИ-модели «{label}»: {value:.2f}",
            criterion=(
                "замороженный DenseNet121 (TorchXRayVision) и логистическая регрессия; 0.5 и "
                "выше — ИИ-модель видит нарушение; точность на отложенных данных ROC AUC "
                f"{task['oof_roc_auc']:.2f}"
            ),
            source=SOURCE,
            decides=False,
            in_score=False,
        )
    )


class SecondOpinion:
    """Загружается один раз на процесс; `onnxruntime` потокобезопасен при вызове `run`."""

    def __init__(self, model_dir: str | Path | None = None) -> None:
        self.model_dir = Path(model_dir) if model_dir else MODEL_DIR
        self.error = ""
        self._session = None
        self._probe: dict | None = None
        self._lock = threading.Lock()

    def load(self) -> bool:
        with self._lock:
            if self._session is not None:
                return True
            onnx, probe = self.model_dir / ONNX_NAME, self.model_dir / PROBE_NAME
            missing = [p.name for p in (onnx, probe) if not p.exists()]
            if missing:
                self.error = f"нет файлов модели: {', '.join(missing)}"
                return False
            try:
                import onnxruntime as ort
            except ImportError:
                self.error = "не установлен onnxruntime"
                return False
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = 2  # обработка и так идёт в нескольких потоках
            self._probe = json.loads(probe.read_text(encoding="utf-8"))
            self._session = ort.InferenceSession(str(onnx), sess_options=opts, providers=["CPUExecutionProvider"])
            logger.info("Второе мнение: %s", self._probe.get("model", ""))
            return True

    @property
    def model(self) -> str:
        return (self._probe or {}).get("model", "")

    def features(self, arr: np.ndarray) -> np.ndarray:
        assert self._session is not None and self._probe is not None
        bb = self._probe["backbone"]
        x = preprocess(arr, bb["height"], bb["width"])
        return self._session.run(None, {"image": x})[0][0]

    def checks(self, arr: np.ndarray, region: str) -> list[dict]:
        """Справочные проверки ИИ-модели для области; пусто, если для неё модели нет."""
        assert self._probe is not None
        tasks = {k: t for k, t in self._probe["tasks"].items() if t["region"] == region}
        if not tasks:
            return []
        f = self.features(arr)
        return [make_check(k, t, score(f, t)) for k, t in tasks.items()]
