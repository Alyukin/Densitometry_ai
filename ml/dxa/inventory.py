"""Инвентаризация DICOM: обход дерева, дедупликация, определение области и стороны.

Что делает и почему (по результатам анализа выгрузки на 100 исследований):

* В одном исследовании один и тот же снимок встречается по 2–5 раз с разными
  SOPInstanceUID и разным тегом ExposedArea. Пиксели при этом побитово совпадают,
  поэтому дедупликация идёт по md5 пиксельного массива.
* Теги BodyPartExamined / ViewPosition / Laterality пустые, PixelSpacing отсутствует,
  поэтому область и сторона определяются по изображению:
    - область: ширина кадра 300 px = поясничный отдел, 280 px (и уже) = бедро;
    - сторона: диафиз бедра в нижней части кадра. По эталону из папки «Для теста»
      (CR000000_ППОБ, CR000001_ЛПОБ) у правого бедра диафиз слева (shaft_x < 0.5),
      у левого — справа. Если в исследовании два снимка бедра, сторона назначается
      сравнением их shaft_x между собой (устойчивее абсолютного порога).
* Размер пикселя не записан в DICOM; заказчик сообщил 0.6 мм по X и 1.05 мм по Y —
  константы PIXEL_SPACING_MM используются для перевода в сантиметры.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pydicom
from pydicom.errors import InvalidDicomError

logger = logging.getLogger(__name__)

PIXEL_SPACING_MM = (1.05, 0.60)  # (row/Y, column/X), со слов заказчика
SPINE_MIN_COLUMNS = 296  # эмпирический порог: позвоночник 300 px, бедро <= 292 px

REGION_SPINE = "Поясничный отдел позвоночника"
REGION_FEMUR = "Проксимальный отдел бедра"


@dataclass
class ImageRecord:
    path: str  # относительный путь от корня данных
    study_uid: str
    series_uid: str
    sop_uid: str
    study_dir: str  # имя папки исследования (обычно тоже StudyInstanceUID)
    rows: int
    columns: int
    bits_allocated: int
    photometric: str
    manufacturer: str
    model: str
    software: str
    exposed_area: str
    pixel_md5: str
    region: str
    side: str  # left | right | "" (для позвоночника)
    side_confidence: float  # 0..1, насколько уверенно определена сторона
    shaft_x: float  # положение диафиза по горизонтали, 0=слева, 1=справа
    mean_intensity: float
    nonzero_fraction: float
    is_duplicate: bool  # True, если такие же пиксели уже встречались в этом исследовании
    duplicate_of: str
    warnings: str


def _pixel_features(a: np.ndarray) -> tuple[float, float, float]:
    """Возвращает (shaft_x, mean, nonzero_fraction)."""
    arr = a.astype(np.float32)
    h, w = arr.shape[:2]
    band = arr[int(h * 0.85) :]
    thr = max(float(band.max()) * 0.45, 30.0)
    mask = band > thr
    if mask.sum() >= 20:
        xs = np.argwhere(mask)[:, 1]
        shaft_x = float(xs.mean()) / max(w - 1, 1)
    else:  # нижняя полоса пустая — берём центр масс всего кадра
        col_mass = arr.sum(axis=0)
        shaft_x = float((col_mass * np.arange(w)).sum() / max(col_mass.sum(), 1e-6)) / max(w - 1, 1)
    return shaft_x, float(arr.mean()), float((arr > 10).mean())


def classify_region(columns: int) -> str:
    return REGION_SPINE if columns >= SPINE_MIN_COLUMNS else REGION_FEMUR


def _read_one(path: Path, root: Path) -> ImageRecord | None:
    try:
        ds = pydicom.dcmread(path, force=True)
        arr = ds.pixel_array
    except (InvalidDicomError, AttributeError, ValueError) as exc:
        logger.warning("пропущен %s: %s", path, exc)
        return None
    if arr.ndim != 2:
        logger.warning("пропущен %s: неожиданная размерность %s", path, arr.shape)
        return None

    shaft_x, mean_i, nz = _pixel_features(arr)
    region = classify_region(int(ds.Columns))
    warns = []
    if nz < 0.02:
        warns.append("почти пустой кадр")
    if int(ds.get("BitsAllocated", 0)) != 8:
        warns.append(f"BitsAllocated={ds.get('BitsAllocated')}")
    return ImageRecord(
        path=str(path.relative_to(root)),
        study_uid=str(ds.get("StudyInstanceUID", "")),
        series_uid=str(ds.get("SeriesInstanceUID", "")),
        sop_uid=str(ds.get("SOPInstanceUID", "")),
        study_dir=path.relative_to(root).parts[0],
        rows=int(ds.Rows),
        columns=int(ds.Columns),
        bits_allocated=int(ds.get("BitsAllocated", 0)),
        photometric=str(ds.get("PhotometricInterpretation", "")),
        manufacturer=str(ds.get("Manufacturer", "")),
        model=str(ds.get("ManufacturerModelName", "")),
        software=str(ds.get("SoftwareVersions", "")),
        exposed_area=str(list(ds.get("ExposedArea", []) or [])),
        pixel_md5=hashlib.md5(arr.tobytes()).hexdigest(),
        region=region,
        side="",
        side_confidence=0.0,
        shaft_x=round(shaft_x, 4),
        mean_intensity=round(mean_i, 2),
        nonzero_fraction=round(nz, 4),
        is_duplicate=False,
        duplicate_of="",
        warnings="; ".join(warns),
    )


def _assign_sides(records: list[ImageRecord]) -> None:
    """Проставляет сторону бедра внутри одного исследования (по уникальным снимкам)."""
    femurs = [r for r in records if r.region == REGION_FEMUR and not r.is_duplicate]
    if len(femurs) == 2:
        a, b = sorted(femurs, key=lambda r: r.shaft_x)
        gap = b.shaft_x - a.shaft_x
        a.side, b.side = (
            "right",
            "left",
        )  # меньший shaft_x = диафиз левее = правое бедро
        conf = min(1.0, 0.5 + gap * 4)
        a.side_confidence = b.side_confidence = round(conf, 3)
        if gap < 0.05:
            for r in (a, b):
                r.warnings = "; ".join(filter(None, [r.warnings, "сторона определена ненадёжно"]))
    else:
        for r in femurs:
            r.side = "right" if r.shaft_x < 0.5 else "left"
            r.side_confidence = round(min(1.0, abs(r.shaft_x - 0.5) * 6), 3)
            if r.side_confidence < 0.3:
                r.warnings = "; ".join(filter(None, [r.warnings, "сторона определена ненадёжно"]))
    # дубликаты наследуют сторону оригинала
    by_md5 = {r.pixel_md5: r for r in records if not r.is_duplicate}
    for r in records:
        if r.is_duplicate and r.pixel_md5 in by_md5:
            src = by_md5[r.pixel_md5]
            r.side, r.side_confidence = src.side, src.side_confidence


def scan(root: str | Path, pattern: str = "**/*.dcm") -> list[ImageRecord]:
    """Обходит дерево DICOM и возвращает записи по всем файлам (с пометкой дубликатов)."""
    root = Path(root)
    records: list[ImageRecord] = []
    for path in sorted(root.glob(pattern)):
        if not path.is_file():
            continue
        rec = _read_one(path, root)
        if rec is not None:
            records.append(rec)

    by_study: dict[str, list[ImageRecord]] = {}
    for r in records:
        by_study.setdefault(r.study_dir, []).append(r)
    for group in by_study.values():
        first_by_md5: dict[str, ImageRecord] = {}
        for r in sorted(group, key=lambda r: r.path):
            if r.pixel_md5 in first_by_md5:
                r.is_duplicate = True
                r.duplicate_of = first_by_md5[r.pixel_md5].path
            else:
                first_by_md5[r.pixel_md5] = r
        _assign_sides(group)
    return records


def to_rows(records: list[ImageRecord]) -> list[dict]:
    return [asdict(r) for r in records]


if __name__ == "__main__":  # pragma: no cover
    import argparse
    import collections
    import csv

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("root", help="папка с исследованиями (НД_для_обучения/Исследования)")
    ap.add_argument("--out", default="manifest.csv")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    recs = scan(args.root)
    uniq = [r for r in recs if not r.is_duplicate]
    print(f"файлов: {len(recs)}, уникальных снимков: {len(uniq)}, исследований: {len({r.study_dir for r in recs})}")
    print("области:", dict(collections.Counter(r.region for r in uniq)))
    print("стороны:", dict(collections.Counter(r.side for r in uniq)))
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(asdict(recs[0])))
        w.writeheader()
        w.writerows(to_rows(recs))
    print("записано:", args.out)
