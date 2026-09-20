"""Решающий слой: измеренные величины -> нарушения из закрытого списка + объяснение.

Каждое правило — это порог на одной измеримой величине. У правила есть источник:

* `ТЗ` — порог прямо записан в техническом задании (например, наклон оси до 5°,
  поля ROI 3 см сверху/снизу и 2 см справа/слева);
* `разметка` — порог подобран по размеченной выгрузке, потому что в ТЗ названа
  проверка, но не назван численный порог (например, «видны ли верхние края
  подвздошных костей»).

Медицинские критерии не придумываются: набор проверок целиком взят из раздела
«Проверки» ТЗ, подбирается только числовая граница и только там, где ТЗ её не задаёт.
Подбор идёт на обучающих фолдах, замер — на отложенных (см. ml/baseline/calibrate.py).

Пороги хранятся отдельно в thresholds.json, чтобы перекалибровка не требовала
изменения кода.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

REGION_SPINE = "Поясничный отдел позвоночника"
REGION_FEMUR = "Проксимальный отдел бедра"

VIOL_POSITION = "Некорректная укладка"
VIOL_SPINE_AXIS = "Не выравнена ось позвоночника"
VIOL_FOREIGN = "Присутствуют посторонние предметы"
VIOL_FEMUR_ROI = "Некорректная область интереса"

THRESHOLDS_PATH = Path(__file__).with_name("thresholds.json")


@dataclass
class Check:
    """Результат одной проверки по одному снимку.

    `decides=False` означает: проверка выполняется и показывается пользователю
    (её требует ТЗ), но в итоговый `violation_type` не попадает, потому что на
    размеченной выгрузке она не отделяет нарушения от нормы. См. REPORT.md.
    """

    rule_id: str
    violation: str
    fired: bool
    value: float
    threshold: float
    op: str
    score: float  # 0..1, мягкая оценка уверенности
    title: str
    measured: str  # «наклон оси 6.3°»
    criterion: str  # «допустимо до 5°»
    source: str  # ТЗ | разметка
    decides: bool = True  # участвует ли проверка в итоговом вердикте
    tz_threshold: float | None = None  # порог, прямо записанный в ТЗ
    tz_fired: bool | None = None  # сработала ли проверка по букве ТЗ
    note: str = ""


@dataclass
class Verdict:
    region: str
    quality_class: int
    quality_prob: float
    violations: list[str] = field(default_factory=list)
    checks: list[Check] = field(default_factory=list)
    explanation: str = ""

    def to_dict(self) -> dict:
        return {
            "region": self.region,
            "quality_class": self.quality_class,
            "quality_prob": round(self.quality_prob, 4),
            "violations": self.violations,
            "explanation": self.explanation,
            "checks": [asdict(c) for c in self.checks],
        }


def _soft(value: float, threshold: float, width: float, op: str) -> float:
    """Плавный переход 0..1 вокруг порога — нужен для quality_prob и ROC/PR."""
    if width <= 0:
        width = max(abs(threshold) * 0.2, 1e-6)
    z = (value - threshold) / width if op == ">" else (threshold - value) / width
    # логистическая функция без импорта math.exp на массивах
    if z > 30:
        return 1.0
    if z < -30:
        return 0.0
    import math

    return 1.0 / (1.0 + math.exp(-z))


# Описание проверок: id -> параметры. Пороги подставляются из thresholds.json.
# Пороги, прямо записанные в ТЗ (раздел 2.3). Они всегда проверяются и показываются
# отдельно от рабочего порога, даже если рабочий подобран по разметке.
TZ_THRESHOLDS = {
    "spine_axis_tz": 5.0,  # «допустимый наклон до 5 градусов»
    "femur_roi_vertical": 3.0,  # «по 3 см сверху и снизу от области интереса»
    "femur_roi_horizontal": 2.0,  # «2 см от края правого и левого»
}

# Описание проверок: id -> параметры. Пороги подставляются из thresholds.json.
# Набор проверок повторяет раздел 2.3 ТЗ один в один.
RULES: dict[str, dict] = {
    # --- позвоночник: «корректная укладка» (рис. 1) -------------------------
    "spine_iliac": {
        "region": REGION_SPINE,
        "violation": VIOL_POSITION,
        "feature": "iliac_score",
        "op": "<",
        "title": "Верхние края подвздошных костей в кадре",
        "unit": "× ширины столба",
        "criterion": "по ТЗ на нижнем уровне сканирования должны визуализироваться верхние края подвздошных костей",
        "source": "разметка",
    },
    "spine_top_level": {
        "decides": False,
        "region": REGION_SPINE,
        "violation": VIOL_POSITION,
        "feature": "vertebrae",
        "op": ">",
        "title": "Позвонков в кадре",
        "unit": "шт",
        "criterion": "по ТЗ верхний уровень сканирования — половина тела Th12; лишние позвонки означают,"
        " что кадр захватил выше нужного",
        "source": "разметка",
    },
    # --- позвоночник: «ось позвоночника» (рис. 2) ---------------------------
    "spine_axis_tz": {
        "decides": False,
        "region": REGION_SPINE,
        "violation": VIOL_SPINE_AXIS,
        "feature": "axis_tz_deg",
        "op": ">",
        "title": "Наклон оси позвоночника (методика ТЗ, рис. 2)",
        "unit": "°",
        "criterion": "по ТЗ допустимый наклон до 5°",
        "source": "ТЗ",
    },
    "spine_axis": {
        "region": REGION_SPINE,
        "violation": VIOL_SPINE_AXIS,
        "feature": "axis_edge_deg",
        "op": ">",
        "title": "Наклон оси по боковым границам столба",
        "unit": "°",
        "criterion": "устойчивая оценка того же наклона оси",
        "source": "разметка",
    },
    "spine_axis_segment": {
        "decides": False,
        "region": REGION_SPINE,
        "violation": VIOL_SPINE_AXIS,
        "feature": "axis_segment_deg",
        "op": ">",
        "title": "Излом оси между верхней и нижней третями",
        "unit": "°",
        "criterion": "при сколиозе суммарный наклон близок к нулю, а ось не выровнена",
        "source": "разметка",
    },
    # --- позвоночник: «посторонние предметы» (рис. 3) -----------------------
    "spine_foreign": {
        "region": REGION_SPINE,
        "violation": VIOL_FOREIGN,
        "feature": "ridge_max",
        "op": ">",
        "title": "Тонкие яркие структуры вне кости",
        "unit": "ед. яркости",
        "criterion": "по ТЗ посторонних предметов, выраженных артефактов и наложений быть не должно",
        "source": "разметка",
    },
    "spine_metal": {
        "region": REGION_SPINE,
        "violation": VIOL_FOREIGN,
        "feature": "metal_area",
        "op": ">",
        "title": "Насыщенные (металлические) объекты вне кости",
        "unit": "px",
        "criterion": "по ТЗ металлических артефактов от одежды быть не должно",
        "source": "разметка",
    },
    # --- бедро: «правильное позиционирование» (рис. 4) ----------------------
    "femur_ischium": {
        "region": REGION_FEMUR,
        "violation": VIOL_POSITION,
        "feature": "ischium_score",
        "op": "<",
        "title": "Седалищная кость в кадре",
        "unit": "доля кости медиальнее бедра",
        "criterion": "по ТЗ на изображении должны быть большой вертел, шейка бедра и седалищная кость",
        "source": "разметка",
    },
    "femur_troch": {
        "decides": False,
        "region": REGION_FEMUR,
        "violation": VIOL_POSITION,
        "feature": "troch_visible",
        "op": "<",
        "title": "Большой вертел в кадре",
        "unit": "× ширины диафиза",
        "criterion": "по ТЗ большой вертел должен визуализироваться; он шире диафиза",
        "source": "разметка",
    },
    # --- бедро: «отсутствие ротации» (рис. 5) -------------------------------
    "femur_rotation_low": {
        "region": REGION_FEMUR,
        "violation": VIOL_POSITION,
        "feature": "lt_prominence_rel",
        "op": "<",
        "requires": ("lt_measured", 1),
        "title": "Выступ малого вертела",
        "unit": "× ширины диафиза",
        "criterion": "по ТЗ (рис. 5б) при переротации контур плавный и не деформирован малым вертелом",
        "source": "разметка",
    },
    "femur_rotation_high": {
        "region": REGION_FEMUR,
        "violation": VIOL_POSITION,
        "feature": "lt_prominence_rel",
        "op": ">",
        "requires": ("lt_measured", 1),
        "title": "Выступ малого вертела",
        "unit": "× ширины диафиза",
        "criterion": "по ТЗ (рис. 5в) при недоротации малый вертел слишком большой",
        "source": "разметка",
    },
    "femur_shaft_tilt": {
        "region": REGION_FEMUR,
        "violation": VIOL_POSITION,
        "feature": "shaft_deg",
        "op": ">",
        "title": "Наклон диафиза к оси сканирования",
        "unit": "°",
        "criterion": "диафиз укладывают вдоль оси сканирования",
        "source": "разметка",
    },
    # --- бедро: «корректность области интереса» (рис. 6) --------------------
    "femur_roi_vertical": {
        "region": REGION_FEMUR,
        "violation": VIOL_FEMUR_ROI,
        "feature": "margin_min_vertical_cm",
        "op": "<",
        "title": "Запас поля сверху/снизу от области интереса",
        "unit": "см",
        "criterion": "по ТЗ (рис. 6) минимум 3 см сверху и снизу",
        "source": "ТЗ",
    },
    "femur_roi_horizontal": {
        "region": REGION_FEMUR,
        "violation": VIOL_FEMUR_ROI,
        "feature": "margin_min_horizontal_cm",
        "op": "<",
        "title": "Запас поля справа/слева от области интереса",
        "unit": "см",
        "criterion": "по ТЗ (рис. 6) минимум 2 см от края",
        "source": "ТЗ",
    },
    "femur_roi_field": {
        "region": REGION_FEMUR,
        "violation": VIOL_FEMUR_ROI,
        "feature": "field_h_cm",
        "op": "<",
        "title": "Высота отсканированного поля",
        "unit": "см",
        "criterion": "слишком короткое поле не вмещает область интереса с запасом",
        "source": "разметка",
    },
}


def load_thresholds(path: str | Path | None = None) -> dict:
    p = Path(path) if path else THRESHOLDS_PATH
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def _fmt(value: float, unit: str) -> str:
    if unit == "px":
        return f"{value:.0f} {unit}"
    if unit == "°":
        return f"{value:.1f}°"
    return f"{value:.2f} {unit}"


def evaluate(region: str, measurements: dict, thresholds: dict | None = None) -> Verdict:
    """Применяет все правила своей области и собирает вердикт с объяснением."""
    th = thresholds if thresholds is not None else load_thresholds()
    checks: list[Check] = []
    for rule_id, spec in RULES.items():
        if spec["region"] != region:
            continue
        cfg = th.get(rule_id)
        if cfg is None or not cfg.get("enabled", True):
            continue
        req = spec.get("requires")
        if req is not None and float(measurements.get(req[0], 0)) != req[1]:
            continue  # измерение недостоверно — правило не высказывается
        feature = spec["feature"]
        if feature not in measurements:
            continue
        value = float(measurements[feature])
        thr = float(cfg["threshold"])
        width = float(cfg.get("soft_width", max(abs(thr) * 0.25, 1e-3)))
        op = spec["op"]
        fired = value > thr if op == ">" else value < thr
        norm = "не более" if op == ">" else "не менее"
        tz = TZ_THRESHOLDS.get(rule_id)
        tz_fired = None if tz is None else (value > tz if op == ">" else value < tz)
        checks.append(
            Check(
                rule_id=rule_id,
                violation=spec["violation"],
                fired=bool(fired),
                value=round(value, 4),
                threshold=thr,
                op=op,
                score=round(_soft(value, thr, width, op), 4),
                title=spec["title"],
                measured=f"{spec['title']}: {_fmt(value, spec['unit'])}",
                criterion=f"{spec['criterion']}; норма {norm} {_fmt(thr, spec['unit'])}",
                source=spec["source"],
                decides=spec.get("decides", True),
                tz_threshold=tz,
                tz_fired=tz_fired,
                note=cfg.get("note", ""),
            )
        )

    violations = sorted({c.violation for c in checks if c.fired and c.decides})
    prob = max((c.score for c in checks if c.decides), default=0.0)
    if violations:
        prob = max(prob, 0.5 + 1e-6)
    else:
        prob = min(prob, 0.5 - 1e-6)
    lines = [
        f"{'НАРУШЕНИЕ' if c.fired else 'норма'}: {c.measured} — {c.criterion}"
        for c in sorted(checks, key=lambda c: (not c.fired, c.rule_id))
    ]
    return Verdict(
        region=region,
        quality_class=1 if violations else 0,
        quality_prob=float(prob),
        violations=violations,
        checks=checks,
        explanation="; ".join(lines),
    )
