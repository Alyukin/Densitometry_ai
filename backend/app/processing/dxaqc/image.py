"""Базовые операции над снимком DXA: маски, единицы измерения, устойчивые примитивы.

Модуль намеренно не зависит ни от FastAPI, ни от torch: только numpy + scipy.
Это позволяет использовать его и в сервисе, и в офлайн-оценке на размеченной выгрузке.

Единицы. В DICOM этой выгрузки нет PixelSpacing; заказчик сообщил размер пикселя
сканера: 1.05 мм по оси Y и 0.60 мм по оси X. Пиксель анизотропный, поэтому любой
угол и любое расстояние переводятся в физические единицы явно — иначе наклон
завышается в 1.75 раза.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.ndimage as ndi

PIXEL_MM_Y = 1.05
PIXEL_MM_X = 0.60

AIR_LEVEL = 8.0  # ниже этого — воздух/вне поля (в выгрузке фон строго 0)
SATURATION = 250.0  # потолок яркости (в выгрузке максимум 252)


@dataclass(frozen=True)
class Spacing:
    """Размер пикселя в мм."""

    y: float = PIXEL_MM_Y
    x: float = PIXEL_MM_X

    def mm_y(self, px: float) -> float:
        return px * self.y

    def mm_x(self, px: float) -> float:
        return px * self.x

    def cm_y(self, px: float) -> float:
        return px * self.y / 10.0

    def cm_x(self, px: float) -> float:
        return px * self.x / 10.0

    def angle_from_vertical(self, slope_px: float) -> float:
        """Наклон dx/dy в пикселях -> угол к вертикали в градусах, с учётом анизотропии."""
        return float(np.degrees(np.arctan(slope_px * self.x / self.y)))


def to_float(arr: np.ndarray) -> np.ndarray:
    """Приводит кадр к float32 в шкале 0..255, не меняя относительных яркостей."""
    a = np.asarray(arr)
    if a.ndim != 2:
        raise ValueError(f"ожидался двумерный кадр, получено {a.shape}")
    a = a.astype(np.float32)
    if a.max() > 255.0:  # на случай 12/16-битных выгрузок
        lo, hi = np.percentile(a, (0.5, 99.5))
        a = np.clip((a - lo) / max(hi - lo, 1e-6), 0, 1) * 255.0
    return a


def body_mask(a: np.ndarray) -> np.ndarray:
    """Маска мягких тканей и кости: всё, что не воздух, одной связной областью."""
    m = a > AIR_LEVEL
    m = ndi.binary_closing(m, np.ones((5, 5)))
    m = ndi.binary_fill_holes(m)
    lab, n = ndi.label(m)
    if n > 1:
        sizes = ndi.sum(m, lab, index=range(1, n + 1))
        m = lab == (int(np.argmax(sizes)) + 1)
    return m


def otsu_threshold(values: np.ndarray) -> float:
    """Порог Оцу по гистограмме 0..255."""
    hist, _ = np.histogram(values, bins=256, range=(0, 256))
    p = hist.astype(np.float64) / max(hist.sum(), 1)
    w = np.cumsum(p)
    mu = np.cumsum(p * np.arange(256))
    mt = mu[-1]
    den = w * (1.0 - w)
    den[den <= 0] = 1e-12
    return float(np.argmax((mt * w - mu) ** 2 / den))


def bone_mask(a: np.ndarray, body: np.ndarray, smooth: tuple[float, float] = (2.0, 1.5)) -> np.ndarray:
    """Маска кости: яркая часть внутри тела (порог Оцу по гистограмме тела)."""
    if body.sum() < 100:
        return np.zeros_like(body)
    sm = ndi.gaussian_filter(a, smooth)
    thr = otsu_threshold(sm[body])
    return body & (sm > thr)


def largest_component(mask: np.ndarray) -> np.ndarray:
    lab, n = ndi.label(mask)
    if n <= 1:
        return mask
    sizes = ndi.sum(mask, lab, index=range(1, n + 1))
    return lab == (int(np.argmax(sizes)) + 1)


def theil_sen_slope(y: np.ndarray, x: np.ndarray, max_pairs: int = 60) -> float:
    """Медиана попарных наклонов dx/dy — устойчива к выбросам, без итераций."""
    n = len(y)
    if n < 8:
        return 0.0
    step = max(1, n // max_pairs)
    i = np.arange(0, n, step)
    Y, X = y[i], x[i]
    slopes = []
    for k in range(len(Y) - 1):
        dy = Y[k + 1 :] - Y[k]
        dx = X[k + 1 :] - X[k]
        ok = np.abs(dy) > 1e-6
        if ok.any():
            slopes.append(dx[ok] / dy[ok])
    return float(np.median(np.concatenate(slopes))) if slopes else 0.0


def field_bounds(body: np.ndarray, a: np.ndarray) -> tuple[int, int, int, int]:
    """Границы отсканированного поля (y0, y1, x0, x1), включительно.

    Фон и воздух в этой выгрузке одинаково равны нулю, поэтому «поле» оценивается
    как прямоугольник кадра за вычетом полностью пустых краевых строк и столбцов:
    сканер растрирует прямоугольник, а сплошная нулевая полоса по краю означает,
    что туда луч не доходил.
    """
    nz = a > AIR_LEVEL
    rows = np.flatnonzero(nz.any(axis=1))
    cols = np.flatnonzero(nz.any(axis=0))
    if len(rows) == 0 or len(cols) == 0:
        h, w = a.shape
        return 0, h - 1, 0, w - 1
    return int(rows[0]), int(rows[-1]), int(cols[0]), int(cols[-1])


def saturated_blobs(a: np.ndarray, min_area: int = 6) -> list[dict]:
    """Компактные насыщенные объекты (металл): яркость у потолка шкалы."""
    m = a >= SATURATION
    if m.sum() == 0:
        return []
    m = ndi.binary_opening(m, np.ones((2, 2)))
    lab, n = ndi.label(m)
    out = []
    for i in range(1, n + 1):
        sel = lab == i
        area = int(sel.sum())
        if area < min_area:
            continue
        ys, xs = np.nonzero(sel)
        out.append(
            {
                "area_px": area,
                "bbox": (int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())),
                "centroid": (float(ys.mean()), float(xs.mean())),
            }
        )
    return out


def ridge_response(a: np.ndarray, sigma: float = 1.2) -> np.ndarray:
    """Отклик на тонкие яркие структуры: исходник минус медианно сглаженный фон.

    Провода, застёжки, цепочки и края одежды дают узкие яркие линии поверх мягких
    тканей; кость такой отклик почти не даёт, потому что она широкая.
    """
    bg = ndi.median_filter(a, size=int(max(5, round(sigma * 7))))
    return np.clip(a - bg, 0, None)
