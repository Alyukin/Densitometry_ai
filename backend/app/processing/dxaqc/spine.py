"""Измерения по поясничному отделу позвоночника.

Что меряется и откуда взято (ТЗ, раздел «Проверки / Позвоночник»):

* «ось позвоночника: допустимый наклон до 5°» (рис. 2 ТЗ) -> `axis_tz_deg`.
  Методика восстановлена по самому рисунку: на нём две линии из общей вершины
  внизу столба, одна строго вертикальная (проверено — её x постоянен по всей
  высоте), вторая проходит по оси столба; подписан угол между ними. То есть
  меряется наклон хорды «низ столба -> верх столба» к вертикали кадра.
  Дополнительно считаются устойчивая подгонка (`axis_edge_deg`) и угол между
  верхней и нижней третями (`axis_segment_deg`): при сколиозе хорда может быть
  почти вертикальной, а столб при этом не выровнен;
* «на нижнем уровне сканирования визуализированы верхние края подвздошных
  костей, верхний уровень — половина тела позвонка Th12» (рис. 1 ТЗ) ->
  нижний уровень: `iliac_score` (во сколько раз кость у нижнего края шире
  столба; если кадр обрезан выше крыльев, кость там не шире самого столба);
  верхний уровень: `top_gap_cm` (столб должен доходить до верхнего края поля,
  то есть кадр обрезает тело позвонка, а не заканчивается выше него) и
  `vertebrae` — число видимых межпозвонковых промежутков;
* «отсутствие выраженных артефактов и металлических предметов» -> площадь
  насыщенных объектов (`metal_area_px`) и протяжённость тонких ярких линий вне
  кости (`ridge_len_px`).

Ось столба берётся по его боковым границам, а не по центру яркости: переход
кость/фон резкий, поэтому границы устойчивее к рёбрам, газу в кишечнике и
асимметрии мягких тканей.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.ndimage as ndi

from .image import (
    Spacing,
    body_mask,
    bone_mask,
    ridge_response,
    saturated_blobs,
    theil_sen_slope,
    to_float,
)


@dataclass
class SpineMeasurements:
    ok: bool = False
    reason: str = ""
    axis_tz_deg: float = 0.0  # наклон по методике ТЗ (рис.2): угол хорды столба к вертикали
    axis_deg: float = 0.0  # наклон подгонной прямой по центру столба
    axis_edge_deg: float = 0.0  # то же по боковым границам столба
    axis_segment_deg: float = 0.0  # угол между верхней и нижней третями столба
    axis_local_max_deg: float = 0.0  # максимальный наклон на трети длины
    cobb_deg: float = 0.0  # угол между верхней и нижней третями
    curve_mm: float = 0.0  # максимальное боковое отклонение от прямой
    column_width_cm: float = 0.0
    column_len_cm: float = 0.0
    iliac_score: float = 0.0  # во сколько раз кость внизу шире столба
    top_gap_cm: float = 0.0  # от верха поля до начала позвоночного столба
    vertebrae: int = 0  # сколько межпозвонковых промежутков видно в столбе
    margin_top_cm: float = 0.0
    margin_bottom_cm: float = 0.0
    field_h_cm: float = 0.0
    field_w_cm: float = 0.0
    ridge_max: float = 0.0  # сила самой яркой тонкой структуры вне кости
    ridge_area: float = 0.0  # площадь таких структур, px
    ridge_len: float = 0.0  # суммарная протяжённость, px
    ridge_longest: float = 0.0  # самая длинная структура, px
    metal_area: float = 0.0  # площадь насыщенных пятен вне кости, px
    metal_max: float = 0.0
    overlay: dict = field(default_factory=dict)  # точки для отрисовки


def _column_runs(
    bone: np.ndarray, body: np.ndarray, lo_frac: float = 0.05, hi_frac: float = 0.95
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Для каждой строки — отрезок [левая, правая] позвоночного столба.

    Трекинг от середины вверх и вниз: в строке берётся связный участок кости,
    содержащий позицию столба с предыдущей строки.
    """
    h, w = bone.shape
    ys_all = np.flatnonzero(body.any(axis=1))
    if len(ys_all) < 40:
        return None
    y0, y1 = int(ys_all[0]), int(ys_all[-1])
    lo = int(y0 + (y1 - y0) * lo_frac)
    hi = int(y0 + (y1 - y0) * hi_frac)
    mid = (lo + hi) // 2
    band = bone[max(0, mid - 8) : mid + 8]
    if band.size == 0:
        return None
    prof = ndi.uniform_filter1d(band.sum(axis=0).astype(float), 5)
    if prof.max() < 1:
        return None
    start = float(np.argmax(prof))

    runs: dict[int, tuple[int, int]] = {}

    def run_at(y: int, cx: float) -> tuple[int, int] | None:
        row = bone[y]
        i = int(round(cx))
        if not (0 <= i < w):
            return None
        if not row[i]:
            idx = np.flatnonzero(row)
            if len(idx) == 0:
                return None
            i = int(idx[int(np.argmin(np.abs(idx - cx)))])
            if abs(i - cx) > w * 0.12:
                return None
        left = i
        while left > 0 and row[left - 1]:
            left -= 1
        right = i
        while right < w - 1 and row[right + 1]:
            right += 1
        return left, right

    for direction in (1, -1):
        cx = start
        rng = range(mid, hi + 1) if direction == 1 else range(mid - 1, lo - 1, -1)
        for y in rng:
            r = run_at(y, cx)
            if r is None:
                continue
            left, right = r
            width = right - left
            if width < w * 0.08 or width > w * 0.75:
                continue
            runs[y] = (left, right)
            cx = (left + right) / 2.0

    if len(runs) < 30:
        return None
    ys = np.array(sorted(runs), dtype=float)
    left = np.array([runs[int(y)][0] for y in ys], dtype=float)
    right = np.array([runs[int(y)][1] for y in ys], dtype=float)
    return ys, left, right


def _iliac_score(bone: np.ndarray, body: np.ndarray, left: np.ndarray, right: np.ndarray) -> float:
    """Во сколько раз кость у нижнего края поля шире позвоночного столба.

    Крылья подвздошных костей — широкие яркие структуры по бокам от столба. Если
    кадр обрезан выше них, кость у нижнего края не шире самого столба.
    Полоса берётся от нижней границы отсканированного поля, а не от конца трека
    столба: трек намеренно обрывается выше таза.
    """
    col_w = float(np.median(right - left))
    if col_w <= 0:
        return 0.0
    ys_f = np.flatnonzero(body.any(axis=1))
    if len(ys_f) < 20:
        return 0.0
    y1 = int(ys_f[-1])
    band_h = max(6, int(len(ys_f) * 0.12))
    band = bone[max(0, y1 - band_h) : y1 + 1]
    widths = []
    for row in band:
        idx = np.flatnonzero(row)
        if len(idx) > 0:
            widths.append(idx[-1] - idx[0])
    if not widths:
        return 0.0
    return float(np.percentile(widths, 75) / col_w)


def _axis_tz(ys: np.ndarray, center: np.ndarray, sp: Spacing) -> float:
    """Наклон оси по методике ТЗ (рис. 2): угол хорды «низ столба -> верх» к вертикали.

    На рисунке ТЗ угол построен двумя лучами из общей вершины внизу столба: один
    строго вертикальный, второй проходит по оси столба. Концы хорды берутся как
    средние по коротким полосам сверху и снизу — одиночная точка слишком шумная.
    """
    n = len(ys)
    if n < 20:
        return 0.0
    band = max(2, int(n * 0.05))
    y_lo, x_lo = float(ys[-band:].mean()), float(center[-band:].mean())
    y_hi, x_hi = float(ys[:band].mean()), float(center[:band].mean())
    dy = sp.mm_y(y_lo - y_hi)
    dx = sp.mm_x(x_hi - x_lo)
    return float(abs(np.degrees(np.arctan2(dx, max(dy, 1e-6)))))


def _vertebra_count(a: np.ndarray, ys: np.ndarray, left: np.ndarray, right: np.ndarray) -> int:
    """Число межпозвонковых промежутков в столбе.

    Промежутки — тёмные поперечные полосы внутри столба: профиль средней яркости
    вдоль столба имеет там локальные минимумы. Нужен для верхнего уровня по ТЗ
    (кадр должен доходить до середины тела Th12).
    """
    if len(ys) < 40:
        return 0
    prof = []
    for i, y in enumerate(ys):
        lo, hi = int(left[i]), int(right[i]) + 1
        seg = a[int(y), lo:hi]
        prof.append(float(seg.mean()) if seg.size else 0.0)
    v = ndi.uniform_filter1d(np.array(prof), 5)
    if v.size < 20 or v.max() <= v.min():
        return 0
    v = (v - v.min()) / (v.max() - v.min())
    win = max(4, len(v) // 20)
    count, last = 0, -(10**6)
    for i in range(win, len(v) - win):
        loc = v[i - win : i + win + 1]
        if v[i] == loc.min() and (loc.max() - v[i]) > 0.18 and i - last > win:
            count += 1
            last = i
    return count


# Пороги отклика подобраны по физике сканера: шкала яркости у этой модели общая для
# всех снимков, поэтому порог абсолютный, а не перцентильный.
RIDGE_ABS_THRESHOLD = 40.0


def _foreign(a: np.ndarray, body: np.ndarray, bone: np.ndarray) -> tuple[dict, list]:
    """Инородные объекты: тонкие яркие структуры и насыщенные пятна вне кости."""
    outside = ~ndi.binary_dilation(bone, np.ones((7, 7)))
    resp = ridge_response(a)
    resp_out = np.where(outside, resp, 0.0)
    # сглаживание 2x2 убирает одиночные шумовые пиксели, не убивая тонкие линии
    resp_sm = ndi.uniform_filter(resp_out, 2)

    strong = resp_sm > RIDGE_ABS_THRESHOLD
    strong = ndi.binary_closing(strong, np.ones((3, 3)))
    lab, n = ndi.label(strong)
    total_len = 0.0
    longest = 0.0
    boxes = []
    for i in range(1, n + 1):
        sel = lab == i
        area = int(sel.sum())
        if area < 8:
            continue
        ys, xs = np.nonzero(sel)
        span = float(np.hypot(ys.max() - ys.min(), xs.max() - xs.min()))
        total_len += span
        if span > longest:
            longest = span
        boxes.append((int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())))

    blobs = [b for b in saturated_blobs(a) if outside[int(b["centroid"][0]), int(b["centroid"][1])]]
    stats = {
        "ridge_max": float(resp_sm.max()),
        "ridge_area": float(strong.sum()),
        "ridge_len": total_len,
        "ridge_longest": longest,
        "metal_area": float(sum(b["area_px"] for b in blobs)),
        "metal_max": float(max((b["area_px"] for b in blobs), default=0.0)),
    }
    return stats, boxes + [b["bbox"] for b in blobs]


def measure_spine(arr: np.ndarray, spacing: Spacing | None = None) -> SpineMeasurements:
    sp = spacing or Spacing()
    a = to_float(arr)
    body = body_mask(a)
    m = SpineMeasurements()
    h, w = a.shape
    ys_f = np.flatnonzero(body.any(axis=1))
    xs_f = np.flatnonzero(body.any(axis=0))
    if len(ys_f) == 0 or len(xs_f) == 0:
        m.reason = "пустой кадр"
        return m
    m.field_h_cm = sp.cm_y(len(ys_f))
    m.field_w_cm = sp.cm_x(len(xs_f))

    bone = bone_mask(a, body)
    runs = _column_runs(bone, body)
    if runs is None:
        m.reason = "не удалось выделить позвоночный столб"
        stats, _ = _foreign(a, body, bone)
        for key, val in stats.items():
            setattr(m, key, val)
        return m

    ys, left, right = runs
    center = (left + right) / 2.0
    k = max(2, int(len(ys) * 0.08))  # края трека шумят
    ys_t, l_t, r_t, c_t = ys[k:-k], left[k:-k], right[k:-k], center[k:-k]
    if len(ys_t) < 20:
        ys_t, l_t, r_t, c_t = ys, left, right, center

    # методика ТЗ считается по ПОЛНОЙ видимой высоте столба, без обрезки краёв
    m.axis_tz_deg = _axis_tz(ys, center, sp)

    slope_c = theil_sen_slope(ys_t, c_t)
    m.axis_deg = abs(sp.angle_from_vertical(slope_c))
    m.axis_edge_deg = abs(
        (sp.angle_from_vertical(theil_sen_slope(ys_t, l_t)) + sp.angle_from_vertical(theil_sen_slope(ys_t, r_t))) / 2
    )

    win = max(15, len(ys_t) // 3)
    locals_ = [
        abs(sp.angle_from_vertical(theil_sen_slope(ys_t[i : i + win], c_t[i : i + win])))
        for i in range(0, len(ys_t) - win + 1, max(1, win // 3))
    ]
    m.axis_local_max_deg = max(locals_) if locals_ else m.axis_deg

    third = max(8, len(ys_t) // 3)
    a_up = sp.angle_from_vertical(theil_sen_slope(ys_t[:third], c_t[:third]))
    a_dn = sp.angle_from_vertical(theil_sen_slope(ys_t[-third:], c_t[-third:]))
    m.cobb_deg = abs(a_up - a_dn)
    m.axis_segment_deg = m.cobb_deg

    fit = c_t[0] + slope_c * (ys_t - ys_t[0])
    m.curve_mm = float(np.abs(c_t - fit).max() * sp.x)

    m.column_width_cm = sp.cm_x(float(np.median(right - left)))
    m.column_len_cm = sp.cm_y(float(ys[-1] - ys[0]))
    m.iliac_score = _iliac_score(bone, body, left, right)
    m.top_gap_cm = sp.cm_y(float(ys[0] - ys_f[0]))
    m.vertebrae = _vertebra_count(a, ys, left, right)
    m.margin_top_cm = m.top_gap_cm
    m.margin_bottom_cm = sp.cm_y(float(ys_f[-1] - ys[-1]))

    stats, boxes = _foreign(a, body, bone)
    for key, val in stats.items():
        setattr(m, key, val)

    step = max(1, len(ys) // 40)
    m.overlay = {
        # линия оси по методике ТЗ: от низа столба к верху, вершина угла внизу
        "axis": [[float(center[:3].mean()), float(ys[0])], [float(center[-3:].mean()), float(ys[-1])]],
        "axis_vertical": [[float(center[-3:].mean()), float(ys[0])], [float(center[-3:].mean()), float(ys[-1])]],
        "column_left": [[float(left[i]), float(ys[i])] for i in range(0, len(ys), step)],
        "column_right": [[float(right[i]), float(ys[i])] for i in range(0, len(ys), step)],
        "foreign": boxes,
        "shape": [int(h), int(w)],
    }
    m.ok = True
    return m
