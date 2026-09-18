"""DICOM helpers: validation, non-personal metadata extraction, PNG preview rendering."""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pydicom
from PIL import Image
from pydicom.errors import InvalidDicomError

logger = logging.getLogger(__name__)

DICOMDIR_SOP_CLASS = "1.2.840.10008.1.3.10"


class DicomValidationError(Exception):
    pass


@dataclass
class DicomMeta:
    study_instance_uid: str | None
    series_instance_uid: str | None
    sop_instance_uid: str | None
    modality: str | None
    manufacturer: str | None
    body_part_examined: str | None
    rows: int | None
    columns: int | None
    has_pixel_data: bool


def _str(ds: pydicom.Dataset, keyword: str) -> str | None:
    val = ds.get(keyword)
    if val is None or val == "":
        return None
    return str(val).strip() or None


def _int(ds: pydicom.Dataset, keyword: str) -> int | None:
    val = ds.get(keyword)
    try:
        return int(val) if val is not None else None
    except (TypeError, ValueError):
        return None


def read_metadata(path: Path) -> DicomMeta:
    """Read and validate a DICOM file. Only technical (non-PHI) attributes are returned."""
    try:
        ds = pydicom.dcmread(path, defer_size="1 KB")
    except InvalidDicomError:
        # Files without the 128-byte preamble / DICM prefix
        try:
            ds = pydicom.dcmread(path, defer_size="1 KB", force=True)
        except Exception as exc:  # noqa: BLE001
            raise DicomValidationError("Файл не является DICOM") from exc
        if not any(k in ds for k in ("SOPInstanceUID", "StudyInstanceUID", "PixelData")):
            raise DicomValidationError("Файл не является DICOM") from None
    except Exception as exc:  # noqa: BLE001
        raise DicomValidationError(f"Не удалось прочитать DICOM: {exc}") from exc

    media_sop = getattr(getattr(ds, "file_meta", None), "MediaStorageSOPClassUID", None)
    if media_sop == DICOMDIR_SOP_CLASS or "DirectoryRecordSequence" in ds:
        raise DicomValidationError("DICOMDIR пропущен (служебный индексный файл)")

    return DicomMeta(
        study_instance_uid=_str(ds, "StudyInstanceUID"),
        series_instance_uid=_str(ds, "SeriesInstanceUID"),
        sop_instance_uid=_str(ds, "SOPInstanceUID"),
        modality=_str(ds, "Modality"),
        manufacturer=_str(ds, "Manufacturer"),
        body_part_examined=_str(ds, "BodyPartExamined"),
        rows=_int(ds, "Rows"),
        columns=_int(ds, "Columns"),
        has_pixel_data="PixelData" in ds,
    )


def render_preview_png(path: Path, max_size: int = 512) -> bytes:
    """Render the first frame of a DICOM image to an 8-bit PNG."""
    from pydicom.pixels import apply_modality_lut, apply_voi_lut

    ds = pydicom.dcmread(path, force=True)
    arr = ds.pixel_array
    samples = int(ds.get("SamplesPerPixel", 1))
    frames = int(ds.get("NumberOfFrames", 1) or 1)
    if frames > 1:
        arr = arr[0]

    if samples == 1:
        try:
            arr = apply_modality_lut(arr, ds)
            arr = apply_voi_lut(arr, ds)
        except Exception:  # noqa: BLE001
            logger.debug("LUT application failed for %s", path, exc_info=True)
        arr = arr.astype(np.float32)
        lo, hi = np.percentile(arr, (0.5, 99.5))
        if hi <= lo:
            lo, hi = float(arr.min()), float(arr.max()) or 1.0
        arr = np.clip((arr - lo) / (hi - lo + 1e-6), 0, 1)
        if str(ds.get("PhotometricInterpretation", "")).upper() == "MONOCHROME1":
            arr = 1.0 - arr
        img = Image.fromarray((arr * 255).astype(np.uint8))  # 2-D uint8 -> mode "L"
    else:
        if arr.dtype != np.uint8:
            arr = (arr / max(float(arr.max()), 1.0) * 255).astype(np.uint8)
        img = Image.fromarray(arr).convert("RGB")

    img.thumbnail((max_size, max_size))
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()
