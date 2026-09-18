"""Mock processor — детерминированные тестовые результаты без AI.

ВНИМАНИЕ: коды регионов, классов качества и нарушений ниже — ВРЕМЕННЫЕ ЗАГЛУШКИ,
собранные по тексту ТЗ. Реальные классы должны быть определены по экспертной разметке
датасета и заменены при подключении модели.
"""

from __future__ import annotations

import hashlib
import random
import time
from dataclasses import dataclass

from app.processing.base import BaseProcessor, ImageInput, ImagePrediction, ProcessingError

REGION_SPINE = "lumbar_spine"
REGION_HIP = "proximal_femur"

QUALITY_OK = "acceptable"
QUALITY_BAD = "unacceptable"

ROTATION_OK = "correct"
ROTATION_OVER = "over_rotation"
ROTATION_UNDER = "under_rotation"


@dataclass(frozen=True)
class _Check:
    code: str
    title: str


SPINE_CHECKS = [
    _Check("iliac_crest_not_visible", "Видны верхние края подвздошных костей"),
    _Check("th12_not_visible", "Видна половина Th12"),
    _Check("spine_axis_tilt", "Наклон оси позвоночника ≤ 5°"),
    _Check("artifacts_present", "Нет выраженных артефактов и металла"),
]

HIP_CHECKS = [
    _Check("greater_trochanter_not_visible", "Виден большой вертел"),
    _Check("femoral_neck_not_visible", "Видна шейка бедра"),
    _Check("ischium_not_visible", "Видна седалищная кость"),
    _Check("rotation", "Ротация по малому вертелу корректна"),
    _Check("roi_margin_insufficient", "ROI: ≥ 3 см сверху/снизу, ≥ 2 см справа/слева"),
]

VIOLATION_CATALOG: dict[str, list[str]] = {
    REGION_SPINE: [c.code for c in SPINE_CHECKS],
    REGION_HIP: [c.code for c in HIP_CHECKS if c.code != "rotation"] + [ROTATION_OVER, ROTATION_UNDER],
}


def _region_from_body_part(body_part: str | None) -> str | None:
    if not body_part:
        return None
    bp = body_part.upper()
    if any(k in bp for k in ("SPINE", "LSPINE", "LUMBAR", "L-SPINE")):
        return REGION_SPINE
    if any(k in bp for k in ("HIP", "FEMUR", "PELVIS")):
        return REGION_HIP
    return None


class MockProcessor(BaseProcessor):
    name = "mock"
    version = "mock-0.1.0"
    is_mock = True

    def __init__(self, delay_per_image_sec: float = 1.5, seed: int = 42, bad_ratio: float = 0.4) -> None:
        self.delay = delay_per_image_sec
        self.seed = seed
        self.bad_ratio = bad_ratio

    def _rng(self, image: ImageInput) -> random.Random:
        key = image.image_uid or image.original_filename
        digest = hashlib.sha256(f"{self.seed}:{key}".encode()).hexdigest()
        return random.Random(int(digest[:16], 16))

    def predict(self, image: ImageInput) -> ImagePrediction:
        rng = self._rng(image)
        if self.delay:
            time.sleep(self.delay * rng.uniform(0.7, 1.3))

        if not image.has_pixel_data:
            raise ProcessingError("DICOM не содержит пиксельных данных (PixelData)")

        region = _region_from_body_part(image.body_part_examined) or rng.choice([REGION_SPINE, REGION_HIP])
        checks_def = SPINE_CHECKS if region == REGION_SPINE else HIP_CHECKS

        is_bad = rng.random() < self.bad_ratio
        failed_codes: set[str] = set()
        if is_bad:
            k = 1 if rng.random() < 0.7 else 2
            failed_codes = {c.code for c in rng.sample(checks_def, k)}

        checks: list[dict] = []
        violations: list[str] = []
        for c in checks_def:
            failed = c.code in failed_codes
            entry: dict = {"code": c.code, "title": c.title, "passed": not failed}

            if c.code == "spine_axis_tilt":
                angle = rng.uniform(5.5, 12.0) if failed else rng.uniform(0.0, 4.5)
                entry.update(value=round(angle, 1), unit="deg", threshold=5.0)
            elif c.code == "rotation":
                verdict = rng.choice([ROTATION_OVER, ROTATION_UNDER]) if failed else ROTATION_OK
                entry.update(value=verdict)
                if failed:
                    violations.append(verdict)
                    checks.append(entry)
                    continue
            elif c.code == "roi_margin_insufficient":
                margins = {
                    "top_cm": round(rng.uniform(3.1, 5.0), 1),
                    "bottom_cm": round(rng.uniform(3.1, 5.0), 1),
                    "left_cm": round(rng.uniform(2.1, 4.0), 1),
                    "right_cm": round(rng.uniform(2.1, 4.0), 1),
                }
                if failed:
                    side = rng.choice(list(margins))
                    margins[side] = round(rng.uniform(0.5, 1.9 if side in ("left_cm", "right_cm") else 2.9), 1)
                entry.update(value=margins, unit="cm", threshold={"vertical_cm": 3.0, "horizontal_cm": 2.0})

            if failed:
                violations.append(c.code)
            checks.append(entry)

        confidence = rng.uniform(0.55, 0.95) if is_bad else rng.uniform(0.75, 0.99)
        return ImagePrediction(
            anatomical_region=region,
            quality_class=QUALITY_BAD if violations else QUALITY_OK,
            violation_types=violations,
            confidence=round(confidence, 3),
            details={
                "mock": True,
                "checks": checks,
                "note": "Тестовый результат. Классы и нарушения — заглушки до подключения модели.",
            },
        )
