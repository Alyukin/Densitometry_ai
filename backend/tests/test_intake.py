"""Приём снимка: отказ, нестандартные данные или обычная обработка.

Главное свойство — защита не срабатывает на настоящих снимках денситометра. Заголовок
ниже повторяет выгрузку заказчика: CR, MONOCHROME2, 8 бит, 280–300 px по ширине,
теги области пустые.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

from app.processing import intake
from app.processing.base import ProcessingError


def _dxa_header(**overrides) -> Dataset:  # noqa: ANN003
    ds = Dataset()
    ds.SpecificCharacterSet = "ISO_IR 192"
    ds.SOPClassUID = "1.2.840.10008.5.1.4.1.1.1"
    ds.SOPInstanceUID = generate_uid()
    ds.StudyInstanceUID = generate_uid()
    ds.Modality = "CR"
    ds.BodyPartExamined = ""
    ds.SeriesDescription = "Изображения DXA"
    ds.ProtocolName = "Anonymized"
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.Rows, ds.Columns = 291, 280
    ds.BitsAllocated = ds.BitsStored = 8
    ds.HighBit = 7
    ds.PixelRepresentation = 0
    ds.PixelData = np.zeros((ds.Rows, ds.Columns), dtype=np.uint8).tobytes()
    for k, v in overrides.items():
        if v is None:
            delattr(ds, k)
        else:
            setattr(ds, k, v)
    return ds


def test_scanner_export_passes() -> None:
    assert intake.header_reason(_dxa_header()) is None
    assert intake.header_reason(_dxa_header(Columns=300, SeriesDescription="DXA Images")) is None


def test_spine_or_hip_in_tags_wins_over_other_words() -> None:
    assert intake.header_reason(_dxa_header(BodyPartExamined="HIP", SeriesDescription="Hip and forearm")) is None


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({"PixelData": None}, "нет изображения"),
        ({"Modality": "CT"}, "модальность CT"),
        ({"Modality": "SR"}, "модальность SR"),
        ({"BodyPartExamined": "FOREARM"}, "forearm"),
        ({"SeriesDescription": "Предплечье"}, "предплеч"),
        ({"SeriesDescription": "Total Body"}, "total body"),
        ({"SamplesPerPixel": 3, "PhotometricInterpretation": "RGB"}, "цветное"),
        ({"Rows": 2500, "Columns": 2048}, "рентгенограммы"),
        ({"Rows": 24, "Columns": 24}, "маленький"),
    ],
)
def test_non_standard_by_header(overrides: dict, expected: str) -> None:
    reason = intake.header_reason(_dxa_header(**overrides))
    assert reason is not None and expected in reason


def test_non_standard_prediction_has_no_region_or_class() -> None:
    pred = intake.non_standard("модальность CT — не проекционный снимок денситометра")
    assert pred.anatomical_region == "" and pred.quality_class == ""
    assert pred.violation_types == [] and pred.confidence is None
    assert intake.is_non_standard(pred)
    assert "качество не оценивается" in pred.details["explanation"]


def test_unreadable_file_is_a_failure(tmp_path: Path) -> None:
    f = tmp_path / "x.dcm"
    f.write_bytes(b"\x00\x01 not a dicom at all" * 10)
    with pytest.raises(ProcessingError):
        intake.inspect(f)


def test_inspect_reads_a_real_file(tmp_path: Path) -> None:
    ds = _dxa_header(Modality="MR")
    ds.file_meta = FileMetaDataset()
    ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds.file_meta.MediaStorageSOPClassUID = ds.SOPClassUID
    ds.file_meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID
    f = tmp_path / "mr.dcm"
    ds.save_as(f, enforce_file_format=True)
    assert "модальность MR" in intake.inspect(f)
    ds.Modality = "CR"
    ds.save_as(f, enforce_file_format=True)
    assert intake.inspect(f) is None
