"""Processor contract.

Любая модель (mock сейчас, AI позже) реализует `BaseProcessor`:
  * `load()`     — один раз при старте (загрузка весов, прогрев GPU);
  * `predict()`  — обработка одного DICOM-изображения.

Оркестрация (очередь, статусы, запись результатов, экспорт) от процессора не зависит,
поэтому для подключения модели достаточно добавить новый класс и зарегистрировать его
в `app/processing/registry.py`, затем выставить `PROCESSOR_BACKEND=<name>`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ImageInput:
    image_id: str
    path: Path
    original_filename: str
    study_uid: str | None
    image_uid: str | None
    series_uid: str | None = None
    modality: str | None = None
    body_part_examined: str | None = None
    rows: int | None = None
    columns: int | None = None
    has_pixel_data: bool = False


@dataclass
class ImagePrediction:
    anatomical_region: str
    quality_class: str
    violation_types: list[str] = field(default_factory=list)
    confidence: float | None = None
    # Explainability payload: per-check verdicts, keypoints, contours, heatmap refs, etc.
    details: dict[str, Any] = field(default_factory=dict)


class ProcessingError(Exception):
    """Raised by a processor when a single image can't be processed."""


class BaseProcessor(ABC):
    name: str = "base"
    version: str = "0.0.0"
    is_mock: bool = False

    def load(self) -> None:  # noqa: B027 — optional hook
        """Load model weights / warm up. Called once."""

    @abstractmethod
    def predict(self, image: ImageInput) -> ImagePrediction:
        """Process a single image. Raise `ProcessingError` for expected failures."""
