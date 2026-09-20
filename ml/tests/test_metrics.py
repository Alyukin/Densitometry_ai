"""Тесты метрик ТЗ: классификация, локализация, пропускная способность.

Отдельно проверяется, что «истина» фантомов не разошлась с самими фантомами: метрики
локализации считаются относительно неё, и молчаливое расхождение испортило бы все
цифры разом.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ml"))
sys.path.insert(0, str(ROOT / "backend"))

from app.scripts.generate_samples import _hip_image, _spine_image, hip_truth, spine_truth  # noqa: E402

from train.metrics import (  # noqa: E402
    binary_metrics,
    dice,
    iou,
    keypoint_distance,
    summarize_localization,
    throughput,
)

# --- Dice / IoU ---------------------------------------------------------------


def _box(h, w, y0, y1, x0, x1):
    m = np.zeros((h, w), dtype=bool)
    m[y0:y1, x0:x1] = True
    return m


def test_dice_and_iou_on_identical_masks() -> None:
    m = _box(20, 20, 4, 14, 4, 14)
    assert dice(m, m) == pytest.approx(1.0)
    assert iou(m, m) == pytest.approx(1.0)


def test_dice_and_iou_on_disjoint_masks() -> None:
    a = _box(20, 20, 0, 5, 0, 5)
    b = _box(20, 20, 10, 15, 10, 15)
    assert dice(a, b) == 0.0
    assert iou(a, b) == 0.0


def test_dice_and_iou_on_half_overlap() -> None:
    a = _box(20, 20, 0, 10, 0, 10)  # 100 px
    b = _box(20, 20, 5, 15, 0, 10)  # 100 px, пересечение 50
    assert dice(a, b) == pytest.approx(2 * 50 / 200)
    assert iou(a, b) == pytest.approx(50 / 150)


def test_metrics_reject_mismatched_shapes() -> None:
    with pytest.raises(ValueError, match="формы масок"):
        dice(np.zeros((4, 4), bool), np.zeros((5, 5), bool))
    with pytest.raises(ValueError, match="формы масок"):
        iou(np.zeros((4, 4), bool), np.zeros((5, 5), bool))


def test_empty_masks_are_not_counted_as_perfect() -> None:
    empty = np.zeros((10, 10), dtype=bool)
    assert np.isnan(dice(empty, empty))
    assert np.isnan(iou(empty, empty))


# --- расстояние между точками -------------------------------------------------


def test_keypoint_distance_uses_anisotropic_pixel() -> None:
    """Пиксель 1.05 x 0.60 мм: шаг по Y и по X — разные расстояния."""
    assert keypoint_distance((0, 0), (0, 1)) == pytest.approx(1.05)
    assert keypoint_distance((0, 0), (1, 0)) == pytest.approx(0.60)
    assert keypoint_distance((0, 0), (3, 4)) == pytest.approx(np.hypot(3 * 0.6, 4 * 1.05))


def test_keypoint_distance_is_nan_when_point_missing() -> None:
    assert np.isnan(keypoint_distance(None, (1, 1)))
    assert np.isnan(keypoint_distance((1, 1), None))


def test_summarize_localization_counts_missing() -> None:
    s = summarize_localization([1.0, 2.0, 3.0, float("nan")])
    assert s["n"] == 4
    assert s["n_measured"] == 3
    assert s["n_missing"] == 1
    assert s["median"] == pytest.approx(2.0)
    assert len(s["median_ci95"]) == 2


def test_summarize_localization_survives_all_missing() -> None:
    s = summarize_localization([float("nan"), float("nan")])
    assert s["n_measured"] == 0 and "median" not in s


# --- пропускная способность ---------------------------------------------------


def test_throughput_reports_success_share_and_limit() -> None:
    t = throughput([0.01, 0.02, 0.03], n_total=4)
    assert t["файлов"] == 4
    assert t["успешно_обработано"] == 3
    assert t["доля_успеха"] == pytest.approx(0.75)
    assert t["время_мс_медиана"] == pytest.approx(20.0)
    assert t["лимит_ТЗ_сек"] == 180


def test_throughput_without_successes() -> None:
    t = throughput([], n_total=5)
    assert t["успешно_обработано"] == 0 and "время_мс_медиана" not in t


def test_binary_metrics_balance_sensitivity_and_specificity() -> None:
    y = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
    p = np.array([0.9, 0.8, 0.2, 0.1, 0.7, 0.1, 0.1, 0.1, 0.1, 0.1])
    m = binary_metrics(y, p, thr=0.5)
    assert m["sensitivity"] == pytest.approx(0.5)
    assert m["specificity"] == pytest.approx(5 / 6)
    assert m["balanced_accuracy"] == pytest.approx((0.5 + 5 / 6) / 2)


# --- согласованность фантома и его «истины» -----------------------------------


def test_spine_truth_matches_the_phantom() -> None:
    """Маска столба должна накрывать именно кость, а не воздух и не мягкие ткани."""
    tilt = 0.08
    img = _spine_image(np.random.default_rng(0), tilt=tilt)
    truth = spine_truth(tilt=tilt)
    assert truth["column"].shape == img.shape
    inside = img[truth["column"]]
    outside = img[~truth["column"] & (img > 0)]
    assert inside.mean() > outside.mean() + 40  # кость заметно ярче окружения
    # ось проходит по середине столба
    for y in (60, 140, 220):
        xs = np.flatnonzero(truth["column"][y])
        assert xs[0] < truth["axis_x"](y) < xs[-1]


def test_hip_truth_is_the_outer_envelope_of_the_femur() -> None:
    img = _hip_image(np.random.default_rng(0), shaft_tilt=0.1, lesser_troch=0.45)
    truth = hip_truth(shaft_tilt=0.1, lesser_troch=0.45)
    assert truth["femur"].shape == img.shape
    # огибающая построчно сплошная: между крайними точками нет разрывов
    for y in (160, 200, 240):
        xs = np.flatnonzero(truth["femur"][y])
        assert len(xs) == xs[-1] - xs[0] + 1
    # седалищная кость в эталон бедра не входит
    assert not truth["femur"][150, 250]
    # вершина малого вертела действительно выступает за край огибающей шейки
    tip = truth["lesser_trochanter"]
    assert tip is not None
    row = np.flatnonzero(truth["femur"][int(tip[1])])
    assert row[-1] == pytest.approx(tip[0], abs=1.5)
