"""Измерения по проксимальному отделу бедра.

Что меряется и откуда взято (ТЗ, раздел «Проверки / Бедро»):

* «проверить ротацию по малому вертелу: корректно / переротация / недоротация» ->
  выступ малого вертела на медиальном контуре ниже шейки (`lt_prominence_mm`).
  При правильной внутренней ротации вертел почти скрыт за диафизом, при
  недоротации выступает, при переротации не виден совсем;
* «проверить ROI: минимум 3 см сверху/снизу и 2 см справа/слева» -> расстояния
  от кости до границы отсканированного поля в сантиметрах (`margin_*_cm`);
* «должны быть видны большой вертел, шейка бедра и седалищная кость» ->
  длина видимого диафиза и наличие кости у медиального края (`ischium_score`).

Дополнительно меряется наклон диафиза к вертикали (`shaft_deg`): в протоколе DXA
бедро укладывают так, чтобы диафиз шёл вдоль оси сканирования. Этот признак не
взят из текста ТЗ, он проверен по размеченной выгрузке (см. ml/baseline/REPORT.md).

Сторона (левое/правое) определяется отдельно в inventory: у правого бедра диафиз
уходит влево, у левого — вправо. Медиальная сторона — та, куда смотрит шейка.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.ndimage as ndi

from .image import (
    Spacing,
    body_mask,
    bone_mask,
    largest_component,
    saturated_blobs,
    theil_sen_slope,
    to_float,
)

# Анатомические ограничения на поиск малого вертела (не критерии качества, а
# границы правдоподобия измерения).
LT_SEARCH_MM = 30.0  # выше верха диафиза вертел не встречается
LT_MAX_PLAUSIBLE_MM = 25.0  # больший «выступ» означает, что контур ушёл на шейку
# Проксимальный отдел бедра у взрослого: от верха диафиза до вертелов ~4-7 см,
# шейка и головка ещё ~3-5 см выше. Ориентиры ищутся только в этих окнах, иначе
# трек, дойдя до таза, отдаёт «вертелы» и «шейку» где-то у верхнего края кадра.
TROCH_SEARCH_MM = 70.0
NECK_SEARCH_MM = 55.0


@dataclass
class FemurMeasurements:
    ok: bool = False
    reason: str = ""
    side: str = ""  # left | right, определяется по положению диафиза
    shaft_deg: float = 0.0  # наклон диафиза к вертикали, градусы
    shaft_len_cm: float = 0.0
    shaft_width_cm: float = 0.0
    lt_prominence_mm: float = 0.0  # выступ малого вертела над контуром диафиза
    lt_prominence_rel: float = 0.0  # то же в долях ширины диафиза
    lt_measured: int = 0  # 1, если вертел удалось локализовать; 0 — измерения нет
    neck_deg: float = 0.0  # наклон шейки к горизонтали
    head_offset_cm: float = 0.0  # смещение головки от оси диафиза
    margin_top_cm: float = 0.0
    margin_bottom_cm: float = 0.0
    margin_lateral_cm: float = 0.0  # со стороны большого вертела
    margin_medial_cm: float = 0.0  # со стороны шейки/седалищной кости
    margin_min_vertical_cm: float = 0.0
    margin_min_horizontal_cm: float = 0.0
    field_h_cm: float = 0.0
    field_w_cm: float = 0.0
    bone_frac: float = 0.0
    femur_len_cm: float = 0.0  # длина видимой бедренной кости
    roi_h_cm: float = 0.0  # высота области интереса (проксимальный отдел бедра)
    roi_w_cm: float = 0.0  # ширина области интереса
    troch_visible: float = 0.0  # во сколько раз вертелы шире диафиза (виден ли большой вертел)
    neck_visible: float = 0.0  # насколько шейка уже вертелов (видна ли шейка)
    troch_width_cm: float = 0.0  # ширина в области вертелов
    neck_width_cm: float = 0.0  # ширина шейки
    troch_ratio: float = 0.0  # во сколько раз вертелы шире диафиза
    neck_ratio: float = 0.0  # ширина шейки к ширине диафиза
    margin_all_bone_top_cm: float = 0.0
    margin_all_bone_bottom_cm: float = 0.0
    ischium_score: float = 0.0  # доля строк с костью у медиального края
    metal_area: float = 0.0
    overlay: dict = field(default_factory=dict)


def _row_runs(row: np.ndarray) -> list[tuple[int, int]]:
    idx = np.flatnonzero(row)
    if len(idx) == 0:
        return []
    splits = np.flatnonzero(np.diff(idx) > 1)
    starts = np.concatenate([[0], splits + 1])
    ends = np.concatenate([splits, [len(idx) - 1]])
    return [(int(idx[s]), int(idx[e])) for s, e in zip(starts, ends, strict=True)]


def _bone_runs(bone: np.ndarray) -> dict[int, tuple[int, int]]:
    """Для каждой строки — самый широкий связный участок кости (для общей геометрии)."""
    out: dict[int, tuple[int, int]] = {}
    for y in range(bone.shape[0]):
        rr = _row_runs(bone[y])
        if rr:
            out[y] = max(rr, key=lambda r: r[1] - r[0])
    return out


def _outer_extent(row: np.ndarray, center: float, half: float) -> tuple[int, int] | None:
    """Внешние границы кости в окне [center-half, center+half].

    Берётся именно внешняя огибающая всех участков в окне, а не один связный
    участок: костномозговой канал диафиза темнее порога, поэтому диафиз в маске
    распадается на две кортикальные полосы, и «связный участок» дал бы половину
    ширины кости.
    """
    lo, hi = center - half, center + half
    parts = [r for r in _row_runs(row) if lo <= (r[0] + r[1]) / 2 <= hi]
    if not parts:
        return None
    return min(p[0] for p in parts), max(p[1] for p in parts)


def track_femur(bone: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Внешний контур бедренной кости снизу вверх.

    Снизу в кадре присутствует только диафиз, поэтому старт там однозначен. Вверх
    идём окном вокруг центра предыдущей строки; ширина окна растёт медленно, чтобы
    в области вертелов трек расширился вместе с костью, но не перескочил на таз,
    когда они соприкасаются.
    """
    ys_bone = np.flatnonzero(bone.any(axis=1))
    if len(ys_bone) < 30:
        return None
    y_hi, y_lo = int(ys_bone[0]), int(ys_bone[-1])
    span = y_lo - y_hi

    # Опорная ширина диафиза — по нижней трети кадра, где кроме диафиза ничего нет.
    # Берётся ВНЕШНЯЯ огибающая всех участков строки: костномозговой канал темнее
    # порога, поэтому диафиз в маске часто распадается на две кортикальные полосы,
    # и «самый широкий участок» дал бы половину истинной ширины.
    y_band = max(y_hi, y_lo - int(span * 0.3))
    widths_lo = []
    for y in range(y_band, y_lo + 1):
        rr = _row_runs(bone[y])
        if rr:
            widths_lo.append(max(r[1] for r in rr) - min(r[0] for r in rr) + 1)
    if not widths_lo:
        return None
    w_ref = float(np.median(widths_lo))
    if w_ref < 5:
        return None

    start_y, cur = None, None
    for y in range(y_lo, y_band - 1, -1):
        rr = _row_runs(bone[y])
        if not rr:
            continue
        cand = (min(r[0] for r in rr), max(r[1] for r in rr))
        if 0.55 * w_ref <= (cand[1] - cand[0] + 1) <= 1.7 * w_ref:
            start_y, cur = y, cand
            break
    if start_y is None:
        return None

    ys, left, right = [float(start_y)], [float(cur[0])], [float(cur[1])]
    width = float(cur[1] - cur[0] + 1)
    center = (cur[0] + cur[1]) / 2.0
    for y in range(start_y - 1, y_hi - 1, -1):
        half = max(width * 0.75, w_ref * 0.75)
        ext = _outer_extent(bone[y], center, half)
        if ext is None:
            break
        w_now = ext[1] - ext[0] + 1
        if w_now > width * 1.12 + 3:  # слишком резкий скачок — скорее всего таз
            w_now = width * 1.12 + 3
            c_new = (ext[0] + ext[1]) / 2.0
            ext = (int(round(c_new - w_now / 2)), int(round(c_new + w_now / 2)))
        ys.append(float(y))
        left.append(float(ext[0]))
        right.append(float(ext[1]))
        width = 0.7 * w_now + 0.3 * width
        center = (ext[0] + ext[1]) / 2.0

    if len(ys) < 20:
        return None
    o = np.argsort(ys)
    return np.array(ys)[o], np.array(left)[o], np.array(right)[o]


@dataclass
class Landmarks:
    """Ориентиры бедра, найденные по профилю ширины трека."""

    i_shaft_top: int  # верх диафиза (начало расширения к вертелам)
    i_troch: int  # самое широкое место — уровень вертелов
    i_neck: int  # самое узкое место выше вертелов — шейка
    shaft_width_px: float
    troch_width_px: float
    neck_width_px: float


def _landmarks(ys: np.ndarray, width: np.ndarray, sp: Spacing) -> Landmarks | None:
    """Разбор трека на диафиз / вертелы / шейку по профилю ширины.

    Трек идёт сверху вниз (ys по возрастанию), то есть последний элемент — низ кадра.
    Внизу только диафиз, выше он расширяется в вертелы, ещё выше сужается в шейку.
    """
    n = len(ys)
    if n < 40:
        return None
    w = ndi.uniform_filter1d(width.astype(float), 5)
    tail = max(8, n // 5)
    shaft_w = float(np.median(w[-tail:]))
    if shaft_w <= 0:
        return None

    # верх диафиза: идём снизу вверх, пока ширина держится около диафизарной
    i_shaft_top = n - 1
    for i in range(n - 1, -1, -1):
        if w[i] > shaft_w * 1.3:
            break
        i_shaft_top = i
    if n - i_shaft_top < 10:
        return None

    # вертелы ищем только в анатомическом окне над диафизом
    troch_lo = max(0, i_shaft_top - int(TROCH_SEARCH_MM / sp.y))
    upper = w[troch_lo:i_shaft_top] if i_shaft_top > troch_lo + 4 else w[: max(1, n // 2)]
    if len(upper) < 5:
        return None
    i_troch = troch_lo + int(np.argmax(upper)) if i_shaft_top > troch_lo + 4 else int(np.argmax(upper))

    # шейка — самое узкое место в окне над вертелами
    neck_lo = max(0, i_troch - int(NECK_SEARCH_MM / sp.y))
    above = w[neck_lo:i_troch] if i_troch > neck_lo + 3 else np.array([])
    i_neck = neck_lo + int(np.argmin(above)) if len(above) >= 3 else i_troch
    return Landmarks(
        i_shaft_top=i_shaft_top,
        i_troch=i_troch,
        i_neck=i_neck,
        shaft_width_px=shaft_w,
        troch_width_px=float(w[i_troch]),
        neck_width_px=float(w[i_neck]),
    )


def _lesser_trochanter(
    ys: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    lm: Landmarks,
    medial_is_right: bool,
    sp: Spacing,
) -> tuple[float, float, tuple[float, float] | None]:
    """Выступ малого вертела над продолжением медиального контура диафиза.

    Малый вертел — бугорок на медиально-задней поверхности сразу выше диафиза.
    По ТЗ ротация оценивается именно по нему: при правильной внутренней ротации он
    почти скрыт за диафизом, при недоротации заметно выступает медиально, при
    переротации не виден совсем. Поэтому признак двусторонний: нарушением считается
    и слишком большой выступ, и его отсутствие.

    Меряем превышение медиального контура над прямой, продолжающей вверх медиальный
    контур диафиза.
    """
    sl = slice(lm.i_shaft_top, len(ys))
    med_shaft = right[sl] if medial_is_right else left[sl]
    y_shaft = ys[sl]
    if len(y_shaft) < 8:
        return 0.0, 0.0, None
    slope = theil_sen_slope(y_shaft, med_shaft)
    base = float(np.median(med_shaft - slope * y_shaft))

    # Зона поиска ограничена анатомически: малый вертел лежит не выше ~3 см над
    # верхом диафиза. Без этого ограничения детектор уходит на шейку и головку,
    # и «выступ» получается в 5-7 см, чего у вертела не бывает.
    y_top = float(ys[lm.i_shaft_top])
    y_limit = y_top - LT_SEARCH_MM / sp.y
    best, best_pt = 0.0, None
    for i in range(0, min(len(ys), lm.i_shaft_top + 1)):
        if ys[i] < y_limit:
            continue
        edge = right[i] if medial_is_right else left[i]
        pred = base + slope * ys[i]
        delta = (edge - pred) if medial_is_right else (pred - edge)
        if delta > best:
            best, best_pt = float(delta), (float(edge), float(ys[i]))

    mm = sp.mm_x(best)
    if mm > LT_MAX_PLAUSIBLE_MM:
        # контур ушёл на шейку или головку — измерение недостоверно
        return 0.0, 0.0, None
    return mm, best / max(lm.shaft_width_px, 1e-6), best_pt


def measure_femur(arr: np.ndarray, spacing: Spacing | None = None, side: str = "") -> FemurMeasurements:
    sp = spacing or Spacing()
    a = to_float(arr)
    m = FemurMeasurements()
    body = body_mask(a)
    h, w_img = a.shape
    ys_f = np.flatnonzero(body.any(axis=1))
    xs_f = np.flatnonzero(body.any(axis=0))
    if len(ys_f) < 20 or len(xs_f) < 20:
        m.reason = "пустой кадр"
        return m
    fy0, fy1, fx0, fx1 = int(ys_f[0]), int(ys_f[-1]), int(xs_f[0]), int(xs_f[-1])
    m.field_h_cm = sp.cm_y(fy1 - fy0 + 1)
    m.field_w_cm = sp.cm_x(fx1 - fx0 + 1)

    bone = largest_component(bone_mask(a, body))
    m.bone_frac = float(bone.mean())
    if bone.sum() < 200:
        m.reason = "кость не найдена"
        return m

    track = track_femur(bone)
    if track is None:
        m.reason = "не удалось проследить бедренную кость"
        return m
    ys, left, right = track
    width = right - left + 1
    lm = _landmarks(ys, width, sp)
    if lm is None:
        m.reason = "не удалось разобрать бедро на диафиз и вертелы"
        return m

    sl_sh = slice(lm.i_shaft_top, len(ys))
    ys_sh, l_sh, r_sh = ys[sl_sh], left[sl_sh], right[sl_sh]
    c_sh = (l_sh + r_sh) / 2.0
    slope = theil_sen_slope(ys_sh, c_sh)
    m.shaft_deg = abs(sp.angle_from_vertical(slope))
    m.shaft_len_cm = sp.cm_y(float(ys_sh[-1] - ys_sh[0]))
    m.shaft_width_cm = sp.cm_x(lm.shaft_width_px)
    m.troch_width_cm = sp.cm_x(lm.troch_width_px)
    m.neck_width_cm = sp.cm_x(lm.neck_width_px)
    m.troch_ratio = float(lm.troch_width_px / max(lm.shaft_width_px, 1e-6))
    m.neck_ratio = float(lm.neck_width_px / max(lm.shaft_width_px, 1e-6))

    # медиальная сторона — та, куда смещён центр на уровне шейки относительно оси диафиза
    y_neck = float(ys[lm.i_neck])
    c_neck = float((left[lm.i_neck] + right[lm.i_neck]) / 2.0)
    shaft_x_at_neck = float(c_sh[0] + slope * (y_neck - ys_sh[0]))
    medial_is_right = c_neck > shaft_x_at_neck
    m.side = side or ("right" if medial_is_right else "left")
    m.head_offset_cm = sp.cm_x(abs(c_neck - shaft_x_at_neck))

    # шейка: угол линии «шейка -> верх диафиза» к горизонтали (шеечно-диафизарный)
    dy_mm = sp.mm_y(abs(float(ys_sh[0]) - y_neck))
    dx_mm = sp.mm_x(abs(float(c_sh[0]) - c_neck))
    m.neck_deg = float(np.degrees(np.arctan2(dy_mm, max(dx_mm, 1e-6))))

    m.lt_prominence_mm, m.lt_prominence_rel, lt_pt = _lesser_trochanter(ys, left, right, lm, medial_is_right, sp)
    m.lt_measured = int(lt_pt is not None)

    # видимая бедренная кость: от шейки до низа кадра
    # Область интереса по рис. 6 ТЗ — это проксимальный отдел бедра, а не всё,
    # что попало в кадр: таз и седалищная кость у края поля нормальны и в отсчёт
    # запасов не входят. Верхняя граница ROI — уровень вертелов (на рисунке
    # стрелка «3 см» упирается именно в большой вертел), нижняя — низ трека.
    # Область интереса тотального бедра: сверху — уровень вертелов/шейки,
    # снизу — подвертельный уровень (верх диафиза). Ниже идёт уже диафиз, который
    # в DXA штатно уходит за нижний край кадра и в отсчёт запаса не входит.
    i_top = min(lm.i_troch, lm.i_neck)
    i_bot = max(i_top + 1, min(len(ys) - 1, lm.i_shaft_top))
    seg = slice(i_top, i_bot + 1)
    by0, by1 = int(ys[i_top]), int(ys[i_bot])
    bx0, bx1 = int(left[seg].min()), int(right[seg].max())
    m.femur_len_cm = sp.cm_y(by1 - int(ys[lm.i_neck]))
    m.roi_h_cm = sp.cm_y(by1 - by0)
    m.roi_w_cm = sp.cm_x(bx1 - bx0)

    m.margin_top_cm = sp.cm_y(by0 - fy0)
    m.margin_bottom_cm = sp.cm_y(fy1 - by1)
    left_cm, right_cm = sp.cm_x(bx0 - fx0), sp.cm_x(fx1 - bx1)
    if medial_is_right:
        m.margin_medial_cm, m.margin_lateral_cm = right_cm, left_cm
    else:
        m.margin_medial_cm, m.margin_lateral_cm = left_cm, right_cm
    m.margin_min_vertical_cm = min(m.margin_top_cm, m.margin_bottom_cm)
    m.margin_min_horizontal_cm = min(left_cm, right_cm)

    ays, axs = np.nonzero(bone)
    m.margin_all_bone_top_cm = sp.cm_y(int(ays.min()) - fy0)
    m.margin_all_bone_bottom_cm = sp.cm_y(fy1 - int(ays.max()))

    # По ТЗ (рис. 4) в кадре должны быть большой вертел, шейка бедра и седалищная
    # кость. Вертел и шейка — это расширение и сужение самого трека бедра;
    # седалищная кость — отдельная кость с медиальной стороны выше вертелов.
    m.troch_visible = float(lm.troch_width_px / max(lm.shaft_width_px, 1e-6))
    m.neck_visible = float(1.0 - lm.neck_width_px / max(lm.troch_width_px, 1e-6))
    y_troch = int(ys[lm.i_troch])
    band = bone[fy0:y_troch, :] if y_troch > fy0 + 3 else bone[fy0 : fy0 + 4, :]
    if band.size:
        # медиальная половина поля, начиная от медиального контура бедра
        med_edge = int(right[lm.i_troch]) if medial_is_right else int(left[lm.i_troch])
        region = band[:, med_edge : fx1 + 1] if medial_is_right else band[:, fx0 : med_edge + 1]
        m.ischium_score = float(region.mean()) if region.size else 0.0

    m.metal_area = float(
        sum(b["area_px"] for b in saturated_blobs(a) if not bone[int(b["centroid"][0]), int(b["centroid"][1])])
    )

    step = max(1, len(ys) // 40)
    m.overlay = {
        "shaft_axis": [[float(c_sh[0]), float(ys_sh[0])], [float(c_sh[-1]), float(ys_sh[-1])]],
        "femur_contour": [[float(left[i]), float(right[i]), float(ys[i])] for i in range(0, len(ys), step)],
        "shaft_top": [float(c_sh[0]), float(ys_sh[0])],
        "trochanter": [float((left[lm.i_troch] + right[lm.i_troch]) / 2), float(ys[lm.i_troch])],
        "neck": [c_neck, y_neck],
        "lesser_trochanter": lt_pt,
        "bone_bbox": [by0, by1, bx0, bx1],
        "field_bbox": [fy0, fy1, fx0, fx1],
        "shape": [int(h), int(w_img)],
    }
    m.ok = True
    return m
