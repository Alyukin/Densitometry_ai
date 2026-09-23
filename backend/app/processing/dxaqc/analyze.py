"""Единая точка входа: массив пикселей -> область, измерения, нарушения, объяснение.

Модуль не зависит от FastAPI и от способа загрузки DICOM, поэтому одинаково
используется и сервисом, и офлайн-оценкой на размеченной выгрузке.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .femur import STRUCTURE_REASONS as FEMUR_STRUCTURE_REASONS
from .femur import measure_femur
from .image import NON_STANDARD_REASONS, Spacing
from .region import REGION_FEMUR, REGION_SPINE, detect, is_dxa_like
from .rules import Verdict, evaluate, load_thresholds, structures_not_found
from .spine import STRUCTURE_REASONS as SPINE_STRUCTURE_REASONS
from .spine import measure_spine

# Отказы разбора, после которых выдаётся вердикт, а не Failure: кость в кадре есть,
# но обязательные по ТЗ структуры не найдены. «Пустой кадр» и «кость не найдена»
# остаются отказами — там оценивать нечего.
STRUCTURE_REASONS = frozenset(FEMUR_STRUCTURE_REASONS + SPINE_STRUCTURE_REASONS)

VERSION = "rulebased-1.1.0"


@dataclass
class Analysis:
    ok: bool
    region: str = ""
    side: str = ""
    region_confidence: float = 0.0
    quality_class: int = 0
    quality_prob: float = 0.0
    violations: list[str] = field(default_factory=list)
    explanation: str = ""
    measurements: dict = field(default_factory=dict)
    checks: list[dict] = field(default_factory=list)
    overlay: dict = field(default_factory=dict)
    error: str = ""
    # Не снимок позвоночника или бедра: вердикта нет, это не отказ (см. app.processing.intake)
    non_standard: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        d["version"] = VERSION
        return d


class Analyzer:
    """Держит пороги в памяти; создаётся один раз на процесс."""

    def __init__(self, thresholds_path: str | Path | None = None, spacing: Spacing | None = None) -> None:
        self.thresholds = load_thresholds(thresholds_path)
        self.spacing = spacing or Spacing()
        self.version = VERSION

    def analyze(
        self,
        arr: np.ndarray,
        body_part: str | None = None,
        series_description: str | None = None,
        protocol_name: str | None = None,
        region: str | None = None,
    ) -> Analysis:
        ok, why = is_dxa_like(arr)
        if not ok:
            return Analysis(ok=False, error=why, non_standard=f"{why}: кадр не похож на снимок денситометра")

        if region in (REGION_SPINE, REGION_FEMUR):
            side = "" if region == REGION_SPINE else detect(arr)[1]
            conf = 1.0
        else:
            region, side, conf = detect(arr, body_part, series_description, protocol_name)

        if region == REGION_SPINE:
            m = measure_spine(arr, self.spacing)
        else:
            m = measure_femur(arr, self.spacing, side=side)

        meas = {k: v for k, v in asdict(m).items() if isinstance(v, int | float) and not isinstance(v, bool)}
        if not m.ok and m.reason in NON_STANDARD_REASONS:
            return Analysis(
                ok=False,
                region=region,
                error=m.reason,
                non_standard=f"{m.reason}: кадр не похож на снимок позвоночника или бедра",
            )
        if not m.ok and m.reason in STRUCTURE_REASONS:
            verdict = structures_not_found(region, m.reason)
            return Analysis(
                ok=True,
                region=region,
                side=side,
                region_confidence=conf,
                quality_class=verdict.quality_class,
                quality_prob=verdict.quality_prob,
                violations=verdict.violations,
                explanation=verdict.explanation,
                measurements={k: round(float(v), 4) for k, v in meas.items()},
                checks=[c for c in verdict.to_dict()["checks"]],
                overlay=m.overlay,
            )
        if not m.ok:
            return Analysis(
                ok=False,
                region=region,
                side=side,
                region_confidence=conf,
                measurements=meas,
                error=m.reason or "не удалось выполнить измерения",
            )

        verdict: Verdict = evaluate(region, meas, self.thresholds)
        return Analysis(
            ok=True,
            region=region,
            side=side,
            region_confidence=conf,
            quality_class=verdict.quality_class,
            quality_prob=verdict.quality_prob,
            violations=verdict.violations,
            explanation=verdict.explanation,
            measurements={k: round(float(v), 4) for k, v in meas.items()},
            checks=[c for c in verdict.to_dict()["checks"]],
            overlay=m.overlay,
        )
