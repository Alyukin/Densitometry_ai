from pathlib import Path

import pytest

from app.processing.base import ImageInput, ProcessingError
from app.processing.mock import VIOLATION_CATALOG, MockProcessor


def _img(uid: str, body_part: str | None = "LSPINE", pixels: bool = True) -> ImageInput:
    return ImageInput(
        image_id="x",
        path=Path("/nonexistent"),
        original_filename=f"{uid}.dcm",
        study_uid="1.2",
        image_uid=uid,
        body_part_examined=body_part,
        has_pixel_data=pixels,
    )


def test_deterministic() -> None:
    p = MockProcessor(delay_per_image_sec=0)
    a = p.predict(_img("1.2.3"))
    b = p.predict(_img("1.2.3"))
    assert a == b


def test_region_from_body_part() -> None:
    p = MockProcessor(delay_per_image_sec=0)
    assert p.predict(_img("1", "LSPINE")).anatomical_region == "lumbar_spine"
    assert p.predict(_img("2", "HIP")).anatomical_region == "proximal_femur"


def test_violations_consistent_with_quality() -> None:
    p = MockProcessor(delay_per_image_sec=0)
    seen_bad = seen_ok = False
    for i in range(200):
        pred = p.predict(_img(f"uid.{i}", "HIP" if i % 2 else "LSPINE"))
        allowed = set(VIOLATION_CATALOG[pred.anatomical_region])
        assert set(pred.violation_types) <= allowed
        if pred.violation_types:
            seen_bad = True
            assert pred.quality_class == "unacceptable"
        else:
            seen_ok = True
            assert pred.quality_class == "acceptable"
    assert seen_bad and seen_ok


def test_no_pixels_raises() -> None:
    with pytest.raises(ProcessingError):
        MockProcessor(delay_per_image_sec=0).predict(_img("1", pixels=False))
