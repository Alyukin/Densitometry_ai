"""Определение задач (голов сети) и разбор меток датасета в векторы целей."""

from __future__ import annotations

from dataclasses import dataclass

from dxa.labels import (
    REGION_FEMUR,
    REGION_SPINE,
    VIOL_FEMUR_ROI,
    VIOL_FOREIGN,
    VIOL_POSITION,
    VIOL_SPINE_AXIS,
)


@dataclass(frozen=True)
class Task:
    key: str  # имя выхода сети
    region: str  # к какой области относится
    label: str  # человекочитаемое имя
    violation: str | None  # None => это бинарный класс качества области


TASKS: list[Task] = [
    Task("spine_quality", REGION_SPINE, "Позвоночник: есть нарушение", None),
    Task("spine_position", REGION_SPINE, f"Позвоночник: {VIOL_POSITION}", VIOL_POSITION),
    Task("spine_axis", REGION_SPINE, f"Позвоночник: {VIOL_SPINE_AXIS}", VIOL_SPINE_AXIS),
    Task("spine_foreign", REGION_SPINE, f"Позвоночник: {VIOL_FOREIGN}", VIOL_FOREIGN),
    Task("femur_quality", REGION_FEMUR, "Бедро: есть нарушение", None),
    Task("femur_position", REGION_FEMUR, f"Бедро: {VIOL_POSITION}", VIOL_POSITION),
    Task("femur_roi", REGION_FEMUR, f"Бедро: {VIOL_FEMUR_ROI}", VIOL_FEMUR_ROI),
]
TASK_INDEX = {t.key: i for i, t in enumerate(TASKS)}
QUALITY_TASK = {REGION_SPINE: "spine_quality", REGION_FEMUR: "femur_quality"}


def targets_for(
    region: str,
    quality_class: int,
    violations: list[str],
    violations_known: bool = True,
) -> tuple[list[float], list[float]]:
    """Возвращает (targets, mask) длины len(TASKS): mask=1 там, где цель определена.

    `violations_known=False` — случай, когда эксперт забраковал область целиком, но не
    отметил ни одного нарушения из закрытого списка (правило R3 в dxa.label_qc).
    Тогда класс качества известен и учится, а головы видов нарушений маскируются:
    считать их нулями значило бы придумать за эксперта, что нарушений нет.
    """
    targets = [0.0] * len(TASKS)
    mask = [0.0] * len(TASKS)
    for i, t in enumerate(TASKS):
        if t.region != region:
            continue
        if t.violation is None:
            mask[i] = 1.0
            targets[i] = float(quality_class)
        elif violations_known:
            mask[i] = 1.0
            targets[i] = float(t.violation in violations)
    return targets, mask
