"""Приём снимка до модели: что считать отказом, а что — нестандартными данными.

Определение заказчика: `processing_status = Failure`, если файл не открывается или не
разбирается как DICOM. Всё, что открылось, получает `Success`.

Снимки, которые не являются ни поясничным отделом, ни бедром, в закрытом тесте
встречаться не должны, но сервис не должен выдавать по ним вердикт. Такой снимок
помечается как нестандартные данные: `Success`, область, класс и вероятность пустые,
причина — в `details["non_standard"]` и в объяснении.

Здесь — проверки по заголовку DICOM, они не зависят от модели и выполняются в
конвейере до неё. Проверки по содержимому кадра (похож ли он на снимок денситометра,
есть ли в нём кость) делает сам процессор и возвращает `non_standard(...)`.

Все пороги намеренно грубые: на 502 файлах выгрузки не срабатывает ни одна проверка.
Ошибиться в сторону «нестандартные» на настоящем снимке хуже, чем пропустить экзотику.
"""

from __future__ import annotations

from pathlib import Path

import pydicom

from app.processing.base import ImagePrediction, ProcessingError
from app.processing.dxaqc.region import region_from_tags

NON_STANDARD_KEY = "non_standard"

# Модальности, которые точно не проекционный снимок денситометра. Список запрещающий:
# неизвестная модальность пропускается дальше, решает содержимое кадра.
NON_DXA_MODALITIES = {
    "CT", "MR", "US", "NM", "PT", "XA", "RF", "MG", "IO", "PX", "ES", "SM", "OP", "OPT", "OCT",
    "IVUS", "ECG", "EEG", "HD", "AU", "SR", "KO", "PR", "SEG", "REG", "DOC", "PLAN",
    "RTSTRUCT", "RTPLAN", "RTDOSE", "RTRECORD", "RTIMAGE", "XC", "GM", "VL",
}  # fmt: skip

# Другие части тела. Проверяются, только если в тех же тегах нет позвоночника или бедра.
OTHER_BODY_PARTS = (
    "FOREARM", "RADIUS", "ULNA", "WRIST", "HAND", "FINGER", "WHOLE BODY", "WHOLEBODY",
    "TOTAL BODY", "TOTALBODY", "TBODY", "SKULL", "CHEST", "THORAX", "BREAST", "KNEE", "FOOT",
    "ANKLE", "HEEL", "CALCANEUS", "SHOULDER", "ELBOW",
    "ПРЕДПЛЕЧ", "ЛУЧЕЗАПЯСТ", "КИСТ", "ВСЁ ТЕЛО", "ВСЕ ТЕЛО", "ЧЕРЕП", "ГРУДН", "КОЛЕН",
    "СТОП", "ПЯТОЧ", "ПЛЕЧ", "ЛОКТ",
)  # fmt: skip

# Кадры денситометра — сотни пикселей (в выгрузке 248–300 по ширине и 230–320 по
# высоте). Тысячи пикселей — это рентгенограмма, десятки — не снимок вовсе.
MAX_DXA_SIDE_PX = 1024
MIN_DXA_SIDE_PX = 60

COLOR_PHOTOMETRIC = ("RGB", "YBR", "PALETTE")


def non_standard(reason: str) -> ImagePrediction:
    """Результат для снимка, который не относится к задаче: без области и класса."""
    return ImagePrediction(
        anatomical_region="",
        quality_class="",
        violation_types=[],
        confidence=None,
        details={
            NON_STANDARD_KEY: reason,
            "explanation": (
                f"Нестандартные данные: {reason}. Это не снимок поясничного отдела позвоночника "
                "или проксимального отдела бедра, качество не оценивается."
            ),
        },
    )


def is_non_standard(pred: ImagePrediction | None) -> bool:
    return bool(pred is not None and (pred.details or {}).get(NON_STANDARD_KEY))


def _text(ds: pydicom.Dataset, keyword: str) -> str:
    value = ds.get(keyword)
    return str(value).strip() if value not in (None, "") else ""


def header_reason(ds: pydicom.Dataset) -> str | None:
    """Причина считать снимок нестандартным — по заголовку DICOM, без чтения пикселей."""
    if not any(k in ds for k in ("PixelData", "FloatPixelData", "DoubleFloatPixelData")):
        sop = ds.get("SOPClassUID")
        kind = getattr(sop, "name", "") if sop else ""
        return f"в файле нет изображения{f' ({kind})' if kind else ''}"

    modality = _text(ds, "Modality").upper()
    if modality in NON_DXA_MODALITIES:
        return f"модальность {modality} — не проекционный снимок денситометра"

    tags = [_text(ds, k) for k in ("BodyPartExamined", "SeriesDescription", "ProtocolName")]
    if region_from_tags(*tags) is None:
        joined = " ".join(tags).upper()
        other = next((w for w in OTHER_BODY_PARTS if w in joined), None)
        if other is not None:
            return f"по тегам DICOM это другая часть тела ({other.lower()})"

    samples = ds.get("SamplesPerPixel", 1)
    photometric = _text(ds, "PhotometricInterpretation").upper()
    if (samples not in (None, "") and int(samples) > 1) or photometric.startswith(COLOR_PHOTOMETRIC):
        return "цветное изображение — похоже на снимок экрана или отчёт, а не на кадр денситометра"

    rows, cols = ds.get("Rows"), ds.get("Columns")
    if rows and cols:
        rows, cols = int(rows), int(cols)
        if max(rows, cols) > MAX_DXA_SIDE_PX:
            return f"кадр {rows}×{cols} px — разрешение рентгенограммы, а не денситометра"
        if min(rows, cols) < MIN_DXA_SIDE_PX:
            return f"кадр {rows}×{cols} px — слишком маленький для снимка денситометра"
    return None


def inspect(path: Path) -> str | None:
    """Читает заголовок: отказ, если файл не разбирается как DICOM, иначе причина нестандартности или None."""
    # defer_size, а не stop_before_pixels: так элемент PixelData виден, но не читается
    forced = False
    try:
        ds = pydicom.dcmread(path, defer_size="1 KB")
    except Exception:  # noqa: BLE001 — без преамбулы DICM пробуем ещё раз
        forced = True
    if forced:
        try:
            ds = pydicom.dcmread(path, defer_size="1 KB", force=True)
        except Exception as exc:  # noqa: BLE001
            raise ProcessingError(f"Файл не разбирается как DICOM: {type(exc).__name__}") from exc
        if not any(k in ds for k in ("SOPInstanceUID", "StudyInstanceUID", "PixelData")):
            raise ProcessingError("Файл не является DICOM")
    return header_reason(ds)
