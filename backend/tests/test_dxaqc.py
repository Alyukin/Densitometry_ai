"""Тесты измерительного ядра и правил ТЗ.

Проверяется то, что можно проверить точно: перевод в физические единицы с учётом
анизотропии пикселя, определение области и стороны, поведение правил на границе
порога, устойчивость к мусорному входу и формат выдачи.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.processing.dxaqc import analyze as analyze_mod
from app.processing.dxaqc.analyze import Analyzer
from app.processing.dxaqc.femur import REASON_NO_LANDMARKS, FemurMeasurements
from app.processing.dxaqc.image import Spacing, body_mask, otsu_threshold, theil_sen_slope, to_float
from app.processing.dxaqc.region import REGION_FEMUR, REGION_SPINE, detect, is_dxa_like
from app.processing.dxaqc.rules import (
    QUALITY_SCORE,
    RULES,
    STRUCTURES_PROB,
    VIOL_POSITION,
    evaluate,
    load_thresholds,
    structures_not_found,
)
from app.processing.dxaqc.spine import measure_spine
from app.scripts.generate_samples import _hip_image, _spine_image

# --- единицы измерения ------------------------------------------------------


def test_angle_accounts_for_anisotropic_pixel() -> None:
    """Пиксель 1.05x0.60 мм: наклон 1 px по X на 1 px по Y — это не 45°."""
    sp = Spacing()
    assert sp.angle_from_vertical(1.0) == pytest.approx(np.degrees(np.arctan(0.6 / 1.05)), abs=1e-6)
    assert sp.angle_from_vertical(0.0) == 0.0
    # если бы анизотропию не учли, вышло бы 45° — ошибка почти в полтора раза
    assert sp.angle_from_vertical(1.0) < 30.0


def test_distances_in_cm() -> None:
    sp = Spacing()
    assert sp.cm_y(100) == pytest.approx(10.5)
    assert sp.cm_x(100) == pytest.approx(6.0)


def test_isotropic_spacing_gives_45_degrees() -> None:
    assert Spacing(y=1.0, x=1.0).angle_from_vertical(1.0) == pytest.approx(45.0)


# --- примитивы --------------------------------------------------------------


def test_otsu_splits_two_modes() -> None:
    """Порог должен разделять две моды: маска строится как `значение > порог`."""
    v = np.concatenate([np.full(500, 40.0), np.full(500, 200.0)])
    thr = otsu_threshold(v)
    assert 40 <= thr < 200
    assert not (40.0 > thr)
    assert 200.0 > thr


def test_theil_sen_is_robust_to_outliers() -> None:
    y = np.arange(60, dtype=float)
    x = 2.0 * y + 5.0
    x[10] += 500  # выброс
    x[40] -= 500
    assert theil_sen_slope(y, x) == pytest.approx(2.0, abs=0.05)


def test_body_mask_keeps_only_largest_blob() -> None:
    a = np.zeros((60, 60), dtype=np.float32)
    a[10:50, 10:50] = 100
    a[0:3, 0:3] = 100  # мелкий шум в углу
    m = body_mask(a)
    assert m[30, 30]
    assert not m[1, 1]


def test_to_float_rejects_non_2d() -> None:
    with pytest.raises(ValueError, match="двумерный"):
        to_float(np.zeros((4, 4, 3)))


# --- область и сторона ------------------------------------------------------


def test_region_from_image_width() -> None:
    rng = np.random.default_rng(0)
    region, side, _ = detect(_spine_image(rng))
    assert region == REGION_SPINE
    assert side == ""
    region, side, _ = detect(_hip_image(rng))
    assert region == REGION_FEMUR
    assert side in ("left", "right")


def test_region_tags_win_over_size() -> None:
    rng = np.random.default_rng(0)
    region, _, conf = detect(_hip_image(rng), body_part="LUMBAR SPINE")
    assert region == REGION_SPINE
    assert conf > 0.9


def test_hip_side_flips_with_image() -> None:
    rng = np.random.default_rng(1)
    img = _hip_image(rng)
    _, side_a, _ = detect(img)
    _, side_b, _ = detect(np.fliplr(img).copy())
    assert {side_a, side_b} == {"left", "right"}


def test_is_dxa_like_rejects_garbage() -> None:
    assert not is_dxa_like(np.zeros((10, 10)))[0]
    assert not is_dxa_like(np.zeros((300, 300)))[0]  # пустой кадр
    assert not is_dxa_like(np.full((300, 300), 200.0))[0]  # нет фона
    assert is_dxa_like(_spine_image(np.random.default_rng(0)))[0]


# --- правила ----------------------------------------------------------------


def test_rule_fires_strictly_past_threshold() -> None:
    th = {"spine_axis": {"threshold": 5.0, "soft_width": 1.0, "enabled": True}}
    below = evaluate(REGION_SPINE, {"axis_edge_deg": 4.9}, th)
    above = evaluate(REGION_SPINE, {"axis_edge_deg": 5.1}, th)
    assert below.quality_class == 0 and below.violations == []
    assert above.quality_class == 1
    assert above.violations == ["Не выравнена ось позвоночника"]
    assert above.quality_prob > below.quality_prob


def test_disabled_rule_is_silent() -> None:
    th = {"spine_axis": {"threshold": 0.1, "soft_width": 1.0, "enabled": False}}
    assert evaluate(REGION_SPINE, {"axis_edge_deg": 99.0}, th).violations == []


def test_rule_with_unreliable_measurement_does_not_fire() -> None:
    """Если вертел не локализован, правило ротации не высказывается."""
    th = {"femur_rotation_low": {"threshold": 0.5, "soft_width": 0.1, "enabled": True}}
    silent = evaluate(REGION_FEMUR, {"lt_prominence_rel": 0.0, "lt_measured": 0}, th)
    speaks = evaluate(REGION_FEMUR, {"lt_prominence_rel": 0.0, "lt_measured": 1}, th)
    assert silent.violations == []
    assert speaks.violations == ["Некорректная укладка"]


def test_explanation_contains_number_and_criterion() -> None:
    th = {"spine_axis": {"threshold": 5.0, "soft_width": 1.0, "enabled": True}}
    v = evaluate(REGION_SPINE, {"axis_edge_deg": 7.25}, th)
    assert "7.2" in v.explanation
    assert "5" in v.explanation
    assert v.checks[0].criterion and v.checks[0].source in ("ТЗ", "разметка")


def test_tz_verdict_is_reported_next_to_working_threshold() -> None:
    """Проверка ТЗ «наклон до 5°» считается всегда, даже если рабочий порог другой."""
    th = {"spine_axis_tz": {"threshold": 2.2, "soft_width": 1.0, "enabled": True}}
    c = evaluate(REGION_SPINE, {"axis_tz_deg": 6.3}, th).checks[0]
    assert c.rule_id == "spine_axis_tz"
    assert c.tz_threshold == 5.0
    assert c.tz_fired is True
    c2 = evaluate(REGION_SPINE, {"axis_tz_deg": 3.0}, th).checks[0]
    assert c2.tz_fired is False  # по букве ТЗ норма
    assert c2.fired is True  # но рабочий порог ниже


def test_reference_only_check_does_not_decide() -> None:
    """Справочная проверка показывается, но не попадает в violation_type."""
    th = {"spine_axis_tz": {"threshold": 1.0, "soft_width": 1.0, "enabled": True}}
    v = evaluate(REGION_SPINE, {"axis_tz_deg": 9.0}, th)
    assert v.checks[0].fired is True
    assert v.checks[0].decides is False
    assert v.violations == []
    assert v.quality_class == 0


def test_rule_demoted_by_calibration_is_shown_but_does_not_decide() -> None:
    """Проверку ТЗ, которую отбор не взял в вердикт, калибровка оставляет справочной."""
    th = {"femur_roi_horizontal": {"threshold": 2.0, "soft_width": 0.5, "enabled": True, "decides": False}}
    v = evaluate(REGION_FEMUR, {"margin_min_horizontal_cm": 1.5}, th)
    assert v.checks[0].fired is True
    assert v.checks[0].tz_fired is True
    assert v.checks[0].decides is False
    assert v.violations == []
    assert v.quality_class == 0


def test_shipped_thresholds_show_every_femur_tz_check() -> None:
    """Поля 3 / 2 см и ротация в вердикт не входят, но показываются — с порогом ТЗ, где он есть."""
    meas = {spec["feature"]: 1.0 for spec in RULES.values() if spec["region"] == REGION_FEMUR}
    meas["lt_measured"] = 1.0
    checks = {c.rule_id: c for c in evaluate(REGION_FEMUR, meas, load_thresholds()).checks}
    for rid in ("femur_roi_vertical", "femur_roi_horizontal", "femur_rotation_low", "femur_rotation_high"):
        assert rid in checks, rid
        assert checks[rid].decides is False, rid
    assert checks["femur_roi_vertical"].threshold == 3.0
    assert checks["femur_roi_horizontal"].threshold == 2.0


def test_all_tz_checks_are_implemented() -> None:
    """Каждая проверка из раздела 2.3 ТЗ должна иметь правило."""
    features = {spec["feature"] for spec in RULES.values()}
    # позвоночник: нижний уровень, верхний уровень, ось, посторонние предметы
    assert "iliac_score" in features  # верхние края подвздошных костей
    assert "vertebrae" in features  # верхний уровень (половина Th12)
    assert "axis_tz_deg" in features  # ось, методика рис. 2
    assert "ridge_max" in features  # посторонние предметы
    # бедро: структуры, ротация, область интереса
    assert "ischium_score" in features  # седалищная кость
    assert "troch_visible" in features  # большой вертел
    assert "lt_prominence_rel" in features  # ротация по малому вертелу
    assert "margin_min_vertical_cm" in features  # 3 см сверху/снизу
    assert "margin_min_horizontal_cm" in features  # 2 см справа/слева


def test_violations_come_from_closed_list() -> None:
    allowed = {
        REGION_SPINE: {"Некорректная укладка", "Не выравнена ось позвоночника", "Присутствуют посторонние предметы"},
        REGION_FEMUR: {"Некорректная укладка", "Некорректная область интереса"},
    }
    for spec in RULES.values():
        assert spec["violation"] in allowed[spec["region"]]


# --- сквозной разбор --------------------------------------------------------


def test_analyzer_on_synthetic_spine() -> None:
    a = Analyzer()
    res = a.analyze(_spine_image(np.random.default_rng(3)))
    assert res.ok
    assert res.region == REGION_SPINE
    assert res.quality_class in (0, 1)
    assert 0.0 <= res.quality_prob <= 1.0
    assert res.measurements["axis_edge_deg"] < 5.0  # фантом ровный
    assert res.overlay["axis"]


def test_analyzer_detects_tilted_spine() -> None:
    a = Analyzer()
    straight = a.analyze(_spine_image(np.random.default_rng(3), tilt=0.0))
    tilted = a.analyze(_spine_image(np.random.default_rng(3), tilt=0.20))
    assert tilted.measurements["axis_edge_deg"] > straight.measurements["axis_edge_deg"] + 3


def test_analyzer_detects_missing_iliac_crests() -> None:
    a = Analyzer()
    with_crest = a.analyze(_spine_image(np.random.default_rng(4), iliac=True))
    without = a.analyze(_spine_image(np.random.default_rng(4), iliac=False))
    assert without.measurements["iliac_score"] < with_crest.measurements["iliac_score"]


def test_analyzer_detects_foreign_object() -> None:
    a = Analyzer()
    clean = a.analyze(_spine_image(np.random.default_rng(5), foreign=False))
    dirty = a.analyze(_spine_image(np.random.default_rng(5), foreign=True))
    assert dirty.measurements["ridge_max"] > clean.measurements["ridge_max"]
    assert "Присутствуют посторонние предметы" in dirty.violations


def test_analyzer_reports_error_instead_of_crashing() -> None:
    a = Analyzer()
    res = a.analyze(np.zeros((300, 300), dtype=np.uint8))
    assert not res.ok
    assert res.error


def test_spine_measurement_is_deterministic() -> None:
    img = _spine_image(np.random.default_rng(6))
    first = measure_spine(img)
    second = measure_spine(img)
    assert first.axis_edge_deg == second.axis_edge_deg
    assert first.iliac_score == second.iliac_score


def test_shaft_tilt_grows_with_rotation() -> None:
    a = Analyzer()
    straight = a.analyze(_hip_image(np.random.default_rng(7), shaft_tilt=0.02))
    tilted = a.analyze(_hip_image(np.random.default_rng(7), shaft_tilt=0.40))
    assert tilted.measurements["shaft_deg"] > straight.measurements["shaft_deg"] + 5


def test_roi_violation_on_cropped_field() -> None:
    a = Analyzer()
    full = a.analyze(_hip_image(np.random.default_rng(8)))
    cropped = a.analyze(_hip_image(np.random.default_rng(8), crop_top=70))
    assert cropped.measurements["field_h_cm"] < full.measurements["field_h_cm"]
    assert "Некорректная область интереса" in cropped.violations


# --- quality_prob и ненайденные структуры -------------------------------------


@pytest.mark.parametrize("region", [REGION_SPINE, REGION_FEMUR])
def test_quality_prob_agrees_with_class(region: str) -> None:
    """Каким бы ни был способ сложения, quality_prob > 0.5 тогда и только тогда, когда класс 1."""
    th = load_thresholds()
    rng = np.random.default_rng(0)
    features = {spec["feature"] for spec in RULES.values() if spec["region"] == region}
    for _ in range(200):
        meas = {f: float(rng.uniform(0, 60)) for f in features}
        meas["lt_measured"] = 1.0
        v = evaluate(region, meas, th)
        assert 0.0 <= v.quality_prob <= 1.0
        assert (v.quality_prob > 0.5) == (v.quality_class == 1) == bool(v.violations)


def test_spine_score_uses_reference_checks_too() -> None:
    """Для позвоночника справочные проверки двигают quality_prob, но не класс."""
    assert QUALITY_SCORE[REGION_SPINE] == "mean_all"
    th = load_thresholds()
    base = {spec["feature"]: 0.0 for spec in RULES.values() if spec["region"] == REGION_SPINE}
    base["iliac_score"] = 10.0  # подвздошные кости в кадре
    calm = evaluate(REGION_SPINE, base, th)
    tilted = evaluate(REGION_SPINE, {**base, "axis_tz_deg": 30.0, "axis_segment_deg": 30.0}, th)
    assert calm.quality_class == tilted.quality_class == 0
    assert tilted.quality_prob > calm.quality_prob


def test_structures_not_found_is_a_verdict_from_closed_list() -> None:
    v = structures_not_found(REGION_FEMUR, REASON_NO_LANDMARKS)
    assert v.quality_class == 1
    assert v.violations == [VIOL_POSITION]
    assert v.quality_prob == STRUCTURES_PROB
    assert REASON_NO_LANDMARKS in v.explanation
    assert v.checks[0].source == "ТЗ"


def test_analyzer_gives_verdict_when_femur_structures_not_found(monkeypatch: pytest.MonkeyPatch) -> None:
    """Кость есть, но бедро не разбирается на диафиз и вертелы — это вердикт, а не отказ."""
    monkeypatch.setattr(
        analyze_mod, "measure_femur", lambda *a, **k: FemurMeasurements(ok=False, reason=REASON_NO_LANDMARKS)
    )
    res = Analyzer().analyze(_hip_image(np.random.default_rng(0)), region=REGION_FEMUR)
    assert res.ok
    assert res.quality_class == 1
    assert res.violations == [VIOL_POSITION]


def test_empty_frame_is_still_a_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """Технические отказы остаются отказами: там оценивать нечего."""
    monkeypatch.setattr(
        analyze_mod, "measure_femur", lambda *a, **k: FemurMeasurements(ok=False, reason="кость не найдена")
    )
    res = Analyzer().analyze(_hip_image(np.random.default_rng(0)), region=REGION_FEMUR)
    assert not res.ok
