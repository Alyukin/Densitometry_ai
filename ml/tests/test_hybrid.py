"""Этап 7: гибрид правил и модели. Проверяется то, что задано протоколом К2 (27.09)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ml"))

from train.hybrid import REGION_TASKS, accept, choose, combine, hybrid_scores, macro_f1  # noqa: E402
from train.tasks import TASKS  # noqa: E402

FEMUR = TASKS[5].region
POS, ROI = 5, 6  # femur_position, femur_roi


def test_variants_on_the_same_scale() -> None:
    r, m = np.array([0.2, 0.6, 0.7]), np.array([0.8, 0.3, 0.9])
    assert combine("rules", r, m).tolist() == r.tolist()
    assert combine("model", r, m).tolist() == m.tolist()
    assert np.allclose(combine("mean", r, m), [0.5, 0.45, 0.8])
    # OR: нарушение, если его видят правила или модель — то есть оценка выше 0.5 у кого-то
    assert (combine("or", r, m) > 0.5).tolist() == [True, True, True]


def test_rules_only_tasks_stay_with_rules_in_every_variant() -> None:
    R, Mo = np.full((2, len(TASKS)), 0.1), np.full((2, len(TASKS)), 0.9)  # noqa: N806
    for v in ("mean", "or", "model"):
        out = hybrid_scores(v, R, Mo, {ROI})
        assert np.allclose(out[:, ROI], 0.1)
        assert not np.allclose(out[:, POS], 0.1)


def _femur_case(n: int = 8):  # noqa: ANN202
    T = np.zeros((n, len(TASKS)))  # noqa: N806
    Mk = np.zeros_like(T)  # noqa: N806
    Mk[:, REGION_TASKS[FEMUR]] = 1
    T[:4, POS] = 1
    T[:2, ROI] = 1
    return T, Mk, np.arange(n)


def test_tie_goes_to_rules() -> None:
    T, Mk, idx = _femur_case()  # noqa: N806
    same = np.where(T > 0, 0.9, 0.1)
    best, scores = choose(same, same.copy(), T, Mk, idx, FEMUR, set())
    assert len({round(s, 10) for s in scores.values()}) == 1
    assert best == "rules"


def test_better_variant_wins() -> None:
    T, Mk, idx = _femur_case()  # noqa: N806
    rules = np.where(T > 0, 0.9, 0.1)
    rules[:4, POS] = 0.1  # правила не видят укладку
    model = np.where(T > 0, 0.8, 0.1)  # среднее на укладке — 0.45, ниже порога
    best, _ = choose(rules, model, T, Mk, idx, FEMUR, set())
    assert best in ("or", "model")
    assert macro_f1(combine(best, rules, model), T, Mk, idx, FEMUR) == pytest.approx(1.0)


def test_masked_images_do_not_count() -> None:
    T, Mk, idx = _femur_case()  # noqa: N806
    S = np.where(T > 0, 0.9, 0.1)  # noqa: N806
    S[0, ROI] = 0.1  # ошибка на снимке…
    Mk[0, ROI] = 0  # …у которого вид нарушения неизвестен (R3)
    T[0, ROI] = 0
    assert macro_f1(S, T, Mk, idx, FEMUR) == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("production", "nested", "rules", "specs", "ok"),
    [
        ("rules", 0.40, 0.30, [0.9, 0.9], False),  # выбор — правила
        ("mean", 0.30, 0.30, [0.9, 0.9], False),  # не выше правил
        ("mean", 0.35, 0.30, [0.9, 0.65], False),  # специфичность ниже 0.70
        ("mean", 0.35, 0.30, [0.9, 0.70], True),
    ],
)
def test_acceptance_needs_all_three_conditions(production, nested, rules, specs, ok) -> None:  # noqa: ANN001
    assert accept(production, nested, rules, specs)[0] is ok
