"""Выдача заключения в виде DICOM Structured Report и вторичной серии с разметкой.

ТЗ (раздел «Дополнительно»): DICOM SR и дополнительные серии. Смысл в том, чтобы
результат можно было положить обратно в PACS рядом с исследованием, а не только
скачать таблицей.

Что формируется:

* **SR** — Comprehensive SR (1.2.840.10008.5.1.4.1.1.88.33). Выбран не Basic Text SR,
  потому что заключение содержит числа с единицами измерения (углы в градусах, поля
  в сантиметрах) и координаты найденных структур — для NUM и SCOORD нужен
  Comprehensive. Каждая проверка попадает в отчёт как измеренная величина, критерий и
  вердикт, то есть ровно то же, что видно в интерфейсе.
* **Вторичная серия** — Secondary Capture с наложенной разметкой, по кадру на каждое
  изображение. Это тот же PNG, что отдаёт `/overlay`, упакованный в DICOM.

Персональные данные сервис у себя не хранит — в базе лежат только технические теги.
Но SR без идентификаторов пациента не ляжет в PACS к нужному исследованию, поэтому
они читаются из исходного файла в момент формирования отчёта и переносятся как есть,
никуда не сохраняясь.
"""

from __future__ import annotations

import io
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pydicom
from PIL import Image
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.sequence import Sequence
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

from app.models import REVIEW_CONFIRMED, REVIEW_CORRECTED, ImageResult, Study

COMPREHENSIVE_SR = "1.2.840.10008.5.1.4.1.1.88.33"
SECONDARY_CAPTURE = "1.2.840.10008.5.1.4.1.1.7"

# Собственная схема кодирования для понятий, которых нет в стандартных словарях.
# Обозначения, начинающиеся с «99», зарезервированы стандартом под локальные схемы.
SCHEME = "99DXAQC"

# Единицы измерения — коды UCUM, как требует стандарт для NUM.
UCUM = {
    "°": ("deg", "градус"),
    "см": ("cm", "сантиметр"),
    "мм": ("mm", "миллиметр"),
    "px": ("{pixels}", "пиксели"),
}
UCUM_RATIO = ("{ratio}", "отношение")

# Теги пациента и исследования, которые переносятся из исходного файла, чтобы SR
# встал в PACS к тому же исследованию. Сервис их не хранит.
PASSTHROUGH_TAGS = (
    "PatientName",
    "PatientID",
    "PatientBirthDate",
    "PatientSex",
    "StudyInstanceUID",
    "StudyDate",
    "StudyTime",
    "StudyID",
    "AccessionNumber",
    "ReferringPhysicianName",
)


def _code(value: str, meaning: str, scheme: str = SCHEME) -> Dataset:
    ds = Dataset()
    # Code Value — VR SH, не длиннее 16 символов. Стандарт предусматривает для более
    # длинных обозначений отдельный атрибут Long Code Value; идентификаторы правил
    # вроде spine_axis_segment.criterion в 16 символов не влезают.
    if len(value) <= 16:
        ds.CodeValue = value
    else:
        ds.LongCodeValue = value
    ds.CodingSchemeDesignator = scheme
    ds.CodeMeaning = meaning[:64]  # Code Meaning — VR LO, 64 символа
    return ds


def _item(value_type: str, concept: Dataset, relationship: str = "CONTAINS") -> Dataset:
    ds = Dataset()
    ds.RelationshipType = relationship
    ds.ValueType = value_type
    ds.ConceptNameCodeSequence = Sequence([concept])
    return ds


def _text(code: str, meaning: str, value: str) -> Dataset:
    ds = _item("TEXT", _code(code, meaning))
    ds.TextValue = value or "—"
    return ds


def _num(code: str, meaning: str, value: float, unit: str) -> Dataset:
    ds = _item("NUM", _code(code, meaning))
    measured = Dataset()
    measured.NumericValue = f"{float(value):.4f}"
    ucum, ucum_meaning = UCUM.get(unit, UCUM_RATIO)
    measured.MeasurementUnitsCodeSequence = Sequence([_code(ucum, ucum_meaning, scheme="UCUM")])
    ds.MeasuredValueSequence = Sequence([measured])
    return ds


def _image_ref(sop_class_uid: str, sop_instance_uid: str) -> Dataset:
    ds = _item("IMAGE", _code("121079", "Изображение, к которому относится заключение", scheme="DCM"))
    ref = Dataset()
    ref.ReferencedSOPClassUID = sop_class_uid
    ref.ReferencedSOPInstanceUID = sop_instance_uid
    ds.ReferencedSOPSequence = Sequence([ref])
    return ds


def _scoord(name: str, graphic_type: str, points: list[float], sop_class_uid: str, sop_uid: str) -> Dataset:
    """Координаты найденной структуры на изображении (ось, контур, ориентир)."""
    ds = _item("SCOORD", _code("GEOM", f"Разметка: {name}"))
    ds.GraphicType = graphic_type
    ds.GraphicData = [float(v) for v in points]
    ref = _item("IMAGE", _code("121079", "Изображение", scheme="DCM"), relationship="SELECTED FROM")
    sop = Dataset()
    sop.ReferencedSOPClassUID = sop_class_uid
    sop.ReferencedSOPInstanceUID = sop_uid
    ref.ReferencedSOPSequence = Sequence([sop])
    ds.ContentSequence = Sequence([ref])
    return ds


def _overlay_items(overlay: dict, sop_class_uid: str, sop_uid: str) -> list[Dataset]:
    """Геометрия из `details.overlay` -> SCOORD. Координаты в пикселях исходного кадра."""
    items: list[Dataset] = []
    for key, name in (("axis", "ось позвоночника"), ("shaft_axis", "ось диафиза")):
        seg = overlay.get(key)
        if seg and len(seg) == 2:
            flat = [seg[0][0], seg[0][1], seg[1][0], seg[1][1]]
            items.append(_scoord(name, "POLYLINE", flat, sop_class_uid, sop_uid))
    for key, name in (
        ("trochanter", "уровень вертелов"),
        ("neck", "шейка бедра"),
        ("shaft_top", "верх диафиза"),
        ("lesser_trochanter", "малый вертел"),
    ):
        pt = overlay.get(key)
        if pt and len(pt) == 2:
            items.append(_scoord(name, "POINT", [pt[0], pt[1]], sop_class_uid, sop_uid))
    for box in overlay.get("foreign") or []:
        if not (isinstance(box, list | tuple) and len(box) == 4):
            continue
        y0, y1, x0, x1 = (float(v) for v in box)  # порядок как в overlay
        items.append(
            _scoord(
                "посторонний объект",
                "POLYLINE",
                [x0, y0, x1, y0, x1, y1, x0, y1, x0, y0],
                sop_class_uid,
                sop_uid,
            )
        )
    return items


def _checks_container(result: ImageResult) -> list[Dataset]:
    """По одному NUM на проверку + TEXT с критерием и вердиктом рядом."""
    items: list[Dataset] = []
    for i, check in enumerate((result.details or {}).get("checks") or []):
        rule_id = check.get("rule_id") or f"check_{i}"
        title = check.get("title") or rule_id
        unit = ""
        measured = check.get("measured") or ""
        for u in UCUM:
            if measured.endswith(u) or f" {u}" in measured:
                unit = u
                break
        num = _num(rule_id, title, float(check.get("value") or 0.0), unit)
        verdict = "нарушение" if check.get("fired") else "норма"
        if not check.get("decides", True):
            verdict += " (справочно, в вердикт не входит)"
        details = [
            _text(f"{rule_id}.criterion", "Критерий", check.get("criterion") or ""),
            _text(f"{rule_id}.verdict", "Результат проверки", verdict),
            _text(f"{rule_id}.source", "Источник критерия", check.get("source") or ""),
        ]
        if check.get("tz_threshold") is not None:
            by_tz = "превышен" if check.get("tz_fired") else "в норме"
            details.append(_text(f"{rule_id}.tz", "Порог ТЗ", f"{check['tz_threshold']} — по букве ТЗ {by_tz}"))
        for d in details:
            d.RelationshipType = "HAS PROPERTIES"
        num.ContentSequence = Sequence(details)
        items.append(num)
    return items


def _image_container(result: ImageResult, sop_class_uid: str, sop_uid: str | None, filename: str) -> Dataset:
    container = _item("CONTAINER", _code("IMGQC", f"Контроль качества: {filename}"))
    container.ContinuityOfContent = "SEPARATE"
    content: list[Dataset] = []
    if sop_uid:
        content.append(_image_ref(sop_class_uid, sop_uid))
    content.append(_text("REGION", "Анатомическая область", result.anatomical_region or ""))
    content.append(_text("QCLASS", "Класс качества (0 — годно, 1 — нарушение)", result.quality_class or ""))
    content.append(_text("VIOL", "Тип нарушения", result.violation_type or "нарушений не найдено"))
    if result.confidence is not None:
        content.append(_num("QPROB", "Вероятность нарушения", float(result.confidence), ""))
    content.append(
        _text("STATUS", "Статус обработки", "Success" if result.processing_status == "success" else "Failure")
    )
    if result.error_message:
        content.append(_text("ERROR", "Причина отказа", result.error_message))

    if result.review_status in (REVIEW_CONFIRMED, REVIEW_CORRECTED):
        verdict = "подтверждено специалистом" if result.review_status == REVIEW_CONFIRMED else "исправлено специалистом"
        content.append(_text("REVIEW", "Проверка специалистом", verdict))
        content.append(_text("REVIEW.QCLASS", "Класс качества после проверки", result.final_quality_class or ""))
        content.append(
            _text("REVIEW.VIOL", "Тип нарушения после проверки", result.final_violation_type or "нарушений не найдено")
        )
        if result.reviewed_by:
            content.append(_text("REVIEW.BY", "Проверил", result.reviewed_by))
        if result.review_comment:
            content.append(_text("REVIEW.NOTE", "Комментарий специалиста", result.review_comment))

    content.extend(_checks_container(result))
    overlay = (result.details or {}).get("overlay") or {}
    if overlay and sop_uid:
        content.extend(_overlay_items(overlay, sop_class_uid, sop_uid))
    container.ContentSequence = Sequence(content)
    return container


def _passthrough(sr: Dataset, source: Dataset | None) -> None:
    for tag in PASSTHROUGH_TAGS:
        value = source.get(tag) if source is not None else None
        if value not in (None, ""):
            setattr(sr, tag, value)
    for tag, default in (("PatientName", ""), ("PatientID", ""), ("StudyID", "1"), ("AccessionNumber", "")):
        if not hasattr(sr, tag):
            setattr(sr, tag, default)


def build_sr(study: Study, results: list[ImageResult], data_dir: Path) -> FileDataset:
    """Одно заключение на исследование, со ссылками на все его изображения."""
    now = datetime.now(UTC)
    by_image = {i.id: i for i in study.images}

    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = COMPREHENSIVE_SR
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    sr = FileDataset("sr.dcm", {}, file_meta=meta, preamble=b"\0" * 128)

    first_source: Dataset | None = None
    for image in study.images:
        path = data_dir / image.stored_path
        if path.exists():
            try:
                first_source = pydicom.dcmread(path, stop_before_pixels=True, force=True)
                break
            except Exception:  # noqa: BLE001 — исходник может быть повреждён, отчёт всё равно нужен
                continue
    _passthrough(sr, first_source)
    if not getattr(sr, "StudyInstanceUID", ""):
        sr.StudyInstanceUID = study.study_instance_uid or generate_uid()

    sr.SOPClassUID = COMPREHENSIVE_SR
    sr.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    sr.SeriesInstanceUID = generate_uid()
    sr.SeriesNumber = 900
    sr.InstanceNumber = 1
    sr.Modality = "SR"
    sr.SeriesDescription = "Контроль качества DXA"
    sr.Manufacturer = "Densitometry AI"
    sr.ManufacturerModelName = study.processor_name or "rulebased"
    sr.SoftwareVersions = study.processor_version or ""
    sr.ContentDate = now.strftime("%Y%m%d")
    sr.ContentTime = now.strftime("%H%M%S")
    sr.SpecificCharacterSet = "ISO_IR 192"  # UTF-8: заключение на русском
    sr.CompletionFlag = "COMPLETE"
    sr.VerificationFlag = (
        "VERIFIED" if any(r.review_status in (REVIEW_CONFIRMED, REVIEW_CORRECTED) for r in results) else "UNVERIFIED"
    )

    sr.ValueType = "CONTAINER"
    sr.ConceptNameCodeSequence = Sequence([_code("126000", "Отчёт по измерениям изображения", scheme="DCM")])
    sr.ContinuityOfContent = "SEPARATE"

    content: list[Dataset] = [
        _text("SUMMARY", "Заключение", _summary_text(results)),
        _text("DISCLAIMER", "Ограничение", "Результат автоматического контроля качества, не медицинское заключение"),
    ]
    for r in results:
        image = by_image.get(r.image_id or "")
        sop_class = "1.2.840.10008.5.1.4.1.1.1"  # CR Image Storage — модальность выгрузки
        if image is not None:
            path = data_dir / image.stored_path
            if path.exists():
                try:
                    src = pydicom.dcmread(path, stop_before_pixels=True, force=True)
                    sop_class = str(src.get("SOPClassUID", sop_class)) or sop_class
                except Exception:  # noqa: BLE001
                    pass
        content.append(
            _image_container(r, sop_class, r.image_uid, image.original_filename if image else (r.image_uid or "—"))
        )
    sr.ContentSequence = Sequence(content)
    return sr


def _summary_text(results: list[ImageResult]) -> str:
    ok = [r for r in results if r.processing_status == "success"]
    bad = [r for r in ok if (r.final_quality_class or "") == "1"]
    if not ok:
        return "Ни одно изображение не удалось обработать"
    if not bad:
        return f"Нарушений не найдено ({len(ok)} изображений)"
    viol = sorted({v for r in bad for v in (r.final_violation_type or "").split(";") if v})
    return f"Нарушения на {len(bad)} из {len(ok)} изображений: {', '.join(viol)}"


def build_secondary_capture(
    png_bytes: bytes, source_path: Path, description: str, instance_number: int, series_uid: str
) -> FileDataset:
    """Разметка поверх снимка, упакованная в DICOM Secondary Capture."""
    img = Image.open(io.BytesIO(png_bytes)).convert("RGB")
    arr = np.asarray(img, dtype=np.uint8)

    source: Dataset | None = None
    try:
        source = pydicom.dcmread(source_path, stop_before_pixels=True, force=True)
    except Exception:  # noqa: BLE001
        source = None

    now = datetime.now(UTC)
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = SECONDARY_CAPTURE
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = FileDataset("sc.dcm", {}, file_meta=meta, preamble=b"\0" * 128)
    _passthrough(ds, source)
    if not getattr(ds, "StudyInstanceUID", ""):
        ds.StudyInstanceUID = generate_uid()

    ds.SOPClassUID = SECONDARY_CAPTURE
    ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    ds.SeriesInstanceUID = series_uid
    ds.SeriesNumber = 901
    ds.InstanceNumber = instance_number
    ds.Modality = "OT"
    ds.SpecificCharacterSet = "ISO_IR 192"
    ds.SeriesDescription = "Разметка контроля качества"
    ds.ImageComments = description
    ds.Manufacturer = "Densitometry AI"
    ds.ConversionType = "WSD"  # workstation — изображение построено программно
    ds.ContentDate = now.strftime("%Y%m%d")
    ds.ContentTime = now.strftime("%H%M%S")

    ds.SamplesPerPixel = 3
    ds.PhotometricInterpretation = "RGB"
    ds.PlanarConfiguration = 0
    ds.Rows, ds.Columns = arr.shape[0], arr.shape[1]
    ds.BitsAllocated = 8
    ds.BitsStored = 8
    ds.HighBit = 7
    ds.PixelRepresentation = 0
    ds.PixelData = arr.tobytes()
    return ds


def to_bytes(ds: FileDataset) -> bytes:
    buf = io.BytesIO()
    ds.save_as(buf, enforce_file_format=True)
    return buf.getvalue()
