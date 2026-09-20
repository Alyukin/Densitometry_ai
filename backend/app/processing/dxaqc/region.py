"""Определение анатомической области и стороны по снимку.

В выгрузке заказчика теги BodyPartExamined / ViewPosition / Laterality пустые,
поэтому область определяется по изображению. Проверено на 252 уникальных снимках
обучающей выгрузки и на эталонной папке «Для теста» (ПОП / ППОБ / ЛПОБ):

* ширина кадра строго трёхмодальная: 300 px — поясничный отдел (99 снимков),
  280 и 248 px — бедро (153 снимка), промежуточных значений нет;
* у правого бедра диафиз уходит влево (shaft_x ≈ 0.37 в эталоне), у левого —
  вправо (≈ 0.65).

Если теги DICOM всё же заполнены, они имеют приоритет: сервис должен работать и с
выгрузками других учреждений.

Сторона в выходную таблицу не попадает (по ответу заказчика на вопрос 15 она не
имеет значения), но нужна для трактовки медиального/латерального края бедра.
"""

from __future__ import annotations

import numpy as np

from .image import AIR_LEVEL, to_float

REGION_SPINE = "Поясничный отдел позвоночника"
REGION_FEMUR = "Проксимальный отдел бедра"

SPINE_MIN_COLUMNS = 296  # эмпирический порог: позвоночник 300 px, бедро <= 280 px

_SPINE_WORDS = ("SPINE", "LUMBAR", "L-SPINE", "LSPINE", "ПОЗВОНОЧ", "ПОП")
_FEMUR_WORDS = ("HIP", "FEMUR", "PELVIS", "БЕДР", "ПОБ")


def region_from_tags(*values: str | None) -> str | None:
    """Область по текстовым тегам DICOM, если они заполнены."""
    for v in values:
        if not v:
            continue
        up = str(v).upper()
        if any(w in up for w in _SPINE_WORDS):
            return REGION_SPINE
        if any(w in up for w in _FEMUR_WORDS):
            return REGION_FEMUR
    return None


def shaft_x(arr: np.ndarray) -> float:
    """Положение диафиза по горизонтали в нижней полосе кадра, 0 — слева, 1 — справа."""
    a = to_float(arr)
    h, w = a.shape
    band = a[int(h * 0.85) :]
    thr = max(float(band.max()) * 0.45, 30.0)
    mask = band > thr
    if mask.sum() >= 20:
        xs = np.argwhere(mask)[:, 1]
        return float(xs.mean()) / max(w - 1, 1)
    col = a.sum(axis=0)
    return float((col * np.arange(w)).sum() / max(col.sum(), 1e-6)) / max(w - 1, 1)


def detect(
    arr: np.ndarray,
    body_part: str | None = None,
    series_description: str | None = None,
    protocol_name: str | None = None,
) -> tuple[str, str, float]:
    """Возвращает (область, сторона, уверенность). Сторона пустая для позвоночника."""
    a = to_float(arr)
    tagged = region_from_tags(body_part, series_description, protocol_name)
    if tagged is not None:
        region, conf = tagged, 0.99
    else:
        columns = a.shape[1]
        region = REGION_SPINE if columns >= SPINE_MIN_COLUMNS else REGION_FEMUR
        # чем дальше ширина от порога, тем увереннее
        conf = float(min(1.0, 0.6 + abs(columns - SPINE_MIN_COLUMNS) / 40.0))

    if region == REGION_SPINE:
        return region, "", conf

    sx = shaft_x(a)
    side = "right" if sx < 0.5 else "left"
    side_conf = float(min(1.0, abs(sx - 0.5) * 6))
    return region, side, min(conf, max(side_conf, 0.3))


def is_dxa_like(arr: np.ndarray) -> tuple[bool, str]:
    """Грубая проверка, что кадр вообще похож на DXA-снимок этого сканера."""
    a = to_float(arr)
    if a.ndim != 2 or min(a.shape) < 60:
        return False, "слишком маленький кадр"
    nz = float((a > AIR_LEVEL).mean())
    if nz < 0.02:
        return False, "кадр почти пустой"
    if nz > 0.995:
        return False, "нет фона: кадр не похож на снимок денситометра"
    return True, ""
