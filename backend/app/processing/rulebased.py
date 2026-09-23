"""Процессор на явных правилах ТЗ: DICOM -> измерения -> нарушения -> объяснение.

Это первый рабочий baseline без обучаемой модели. Вся медицинская логика собрана в
пакете `dxaqc` и не зависит от FastAPI, поэтому её же использует офлайн-оценка на
размеченной выгрузке (`ml/baseline/`).

Значения на выходе — строго из закрытых списков заказчика:
  anatomical_region: «Поясничный отдел позвоночника» | «Проксимальный отдел бедра»
  violation_type для позвоночника: «Некорректная укладка», «Не выравнена ось
    позвоночника», «Присутствуют посторонние предметы»
  violation_type для бедра: «Некорректная укладка», «Некорректная область интереса»
  quality_class: «0» (нарушений нет) | «1» (есть)

Подключение обучаемой модели этот процессор не ломает: она регистрируется как
отдельный бэкенд и сравнивается с baseline на тех же метриках.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pydicom

from app.processing.base import BaseProcessor, ImageInput, ImagePrediction, ProcessingError
from app.processing.dxaqc.analyze import VERSION, Analyzer
from app.processing.dxaqc.image import PIXEL_MM_X, PIXEL_MM_Y, Spacing
from app.processing.intake import non_standard

logger = logging.getLogger(__name__)

QUALITY_OK = "0"
QUALITY_BAD = "1"


def read_pixels(path: Path) -> tuple[np.ndarray, pydicom.Dataset]:
    try:
        ds = pydicom.dcmread(path, force=True)
        arr = ds.pixel_array
    except Exception as exc:  # noqa: BLE001 — любой сбой декодера: файл не разбирается, это Failure
        raise ProcessingError(f"Не удалось прочитать пиксельные данные DICOM: {exc}") from exc
    if arr.ndim == 3:  # многокадровый или цветной — берём первый кадр / яркость
        arr = arr[0] if arr.shape[-1] not in (3, 4) else arr[..., 0]
    if arr.ndim != 2:
        raise ProcessingError(f"Неожиданная размерность изображения: {arr.shape}")
    if str(ds.get("PhotometricInterpretation", "")) == "MONOCHROME1":
        arr = arr.max() - arr  # инверсия: в MONOCHROME1 кость тёмная
    return arr, ds


def spacing_from_dicom(ds: pydicom.Dataset) -> Spacing:
    """PixelSpacing из тега, иначе константы сканера, сообщённые заказчиком."""
    for tag in ("PixelSpacing", "ImagerPixelSpacing"):
        val = ds.get(tag)
        if val is not None and len(val) == 2:
            try:
                return Spacing(y=float(val[0]), x=float(val[1]))
            except (TypeError, ValueError):
                pass
    return Spacing(y=PIXEL_MM_Y, x=PIXEL_MM_X)


class RuleBasedProcessor(BaseProcessor):
    name = "rulebased"
    version = VERSION
    is_mock = False

    def __init__(self, thresholds_path: str | None = None) -> None:
        self.thresholds_path = thresholds_path
        self._analyzer: Analyzer | None = None

    def load(self) -> None:
        self._analyzer = Analyzer(self.thresholds_path)
        n = len([c for c in self._analyzer.thresholds.values() if c.get("enabled", True)])
        logger.info("Rule-based processor loaded: %s активных правил", n)

    def predict(self, image: ImageInput) -> ImagePrediction:
        if self._analyzer is None:
            self.load()
        assert self._analyzer is not None

        arr, ds = read_pixels(image.path)
        analyzer = self._analyzer
        sp = spacing_from_dicom(ds)
        if (sp.y, sp.x) != (analyzer.spacing.y, analyzer.spacing.x):
            analyzer = Analyzer(self.thresholds_path, spacing=sp)

        res = analyzer.analyze(
            arr,
            body_part=str(ds.get("BodyPartExamined", "") or "") or None,
            series_description=str(ds.get("SeriesDescription", "") or "") or None,
            protocol_name=str(ds.get("ProtocolName", "") or "") or None,
        )
        if res.non_standard:
            return non_standard(res.non_standard)
        if not res.ok:
            raise ProcessingError(res.error or "Не удалось выполнить измерения")

        return ImagePrediction(
            anatomical_region=res.region,
            quality_class=QUALITY_BAD if res.violations else QUALITY_OK,
            violation_types=res.violations,
            confidence=round(res.quality_prob, 4),
            details={
                "version": VERSION,
                "side": res.side,
                "region_confidence": round(res.region_confidence, 3),
                "quality_prob": round(res.quality_prob, 4),
                "pixel_spacing_mm": {"y": sp.y, "x": sp.x},
                "explanation": res.explanation,
                "checks": res.checks,
                "measurements": res.measurements,
                "overlay": res.overlay,
            },
        )
