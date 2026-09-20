"""Метрики по ТЗ.

Классификация: sensitivity, specificity, balanced accuracy, F1, ROC AUC, PR AUC + 95% ДИ.
Локализация: Dice, IoU, расстояние между ключевыми точками (в миллиметрах) + 95% ДИ.
Пропускная способность: время обработки и доля успешно обработанных файлов.

Один и тот же модуль используют и baseline на правилах (`ml/baseline/`), и обучение
(`ml/train/train.py`), поэтому цифры сравнимы напрямую.
"""

from __future__ import annotations

import numpy as np


def _rates(y: np.ndarray, p: np.ndarray, thr: float) -> tuple[int, int, int, int]:
    pred = (p >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    return tp, tn, fp, fn


def roc_auc(y: np.ndarray, p: np.ndarray) -> float:
    pos, neg = p[y == 1], p[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(p, kind="mergesort")
    ranks = np.empty(len(p), dtype=float)
    ranks[order] = np.arange(1, len(p) + 1)
    # средние ранги для одинаковых значений
    sp = p[order]
    i = 0
    while i < len(sp):
        j = i
        while j + 1 < len(sp) and sp[j + 1] == sp[i]:
            j += 1
        if j > i:
            ranks[order[i : j + 1]] = ranks[order[i : j + 1]].mean()
        i = j + 1
    return float((ranks[y == 1].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def pr_auc(y: np.ndarray, p: np.ndarray) -> float:
    """Average precision."""
    if y.sum() == 0:
        return float("nan")
    order = np.argsort(-p, kind="mergesort")
    y = y[order]
    tp = np.cumsum(y)
    precision = tp / np.arange(1, len(y) + 1)
    return float((precision * y).sum() / y.sum())


def binary_metrics(y: np.ndarray, p: np.ndarray, thr: float = 0.5) -> dict[str, float]:
    tp, tn, fp, fn = _rates(y, p, thr)
    sens = tp / (tp + fn) if tp + fn else float("nan")
    spec = tn / (tn + fp) if tn + fp else float("nan")
    prec = tp / (tp + fp) if tp + fp else float("nan")
    f1 = 2 * prec * sens / (prec + sens) if prec and sens and not np.isnan(prec) and not np.isnan(sens) else 0.0
    return {
        "n": int(len(y)),
        "n_pos": int(y.sum()),
        "sensitivity": sens,
        "specificity": spec,
        "precision": prec,
        "f1": f1,
        "balanced_accuracy": (sens + spec) / 2 if not (np.isnan(sens) or np.isnan(spec)) else float("nan"),
        "roc_auc": roc_auc(y, p),
        "pr_auc": pr_auc(y, p),
    }


def bootstrap_ci(
    y: np.ndarray,
    p: np.ndarray,
    metric: str,
    thr: float = 0.5,
    n: int = 2000,
    seed: int = 0,
) -> tuple[float, float]:
    """95% доверительный интервал перцентильным бутстрэпом."""
    rng = np.random.default_rng(seed)
    vals = []
    idx = np.arange(len(y))
    for _ in range(n):
        b = rng.choice(idx, size=len(idx), replace=True)
        if y[b].sum() in (0, len(b)):
            continue
        vals.append(binary_metrics(y[b], p[b], thr)[metric])
    if not vals:
        return float("nan"), float("nan")
    lo, hi = np.percentile(np.asarray(vals, dtype=float), (2.5, 97.5))
    return float(lo), float(hi)


def best_threshold(y: np.ndarray, p: np.ndarray, objective: str = "balanced_accuracy") -> float:
    """Порог, максимизирующий метрику (по сетке из наблюдаемых вероятностей)."""
    cands = np.unique(np.concatenate([p, [0.5]]))
    best, best_v = 0.5, -1.0
    for t in cands:
        v = binary_metrics(y, p, float(t))[objective]
        if not np.isnan(v) and v > best_v:
            best, best_v = float(t), v
    return best


def summarize(y: np.ndarray, p: np.ndarray, thr: float = 0.5, ci: bool = True, seed: int = 0) -> dict:
    m = binary_metrics(y, p, thr)
    m["threshold"] = thr
    if ci and 0 < y.sum() < len(y):
        for key in (
            "sensitivity",
            "specificity",
            "balanced_accuracy",
            "f1",
            "roc_auc",
            "pr_auc",
        ):
            lo, hi = bootstrap_ci(y, p, key, thr, seed=seed)
            m[f"{key}_ci95"] = [round(lo, 4), round(hi, 4)]
    return m


# --- локализация: Dice / IoU / расстояние между ключевыми точками -------------
#
# ТЗ требует эти метрики для локализации. Экспертных масок и точек в выгрузке нет,
# поэтому истина берётся из двух независимых источников, и оба честно подписаны:
#
#   * синтетические фантомы, где геометрия задана формулой и известна точно;
#   * реальные снимки — повторяемость: к снимку применяется известное малое
#     преобразование (поворот, сдвиг), локализация считается заново и переводится
#     обратно; Dice/IoU и смещение точек показывают, насколько устойчив детектор.
#
# Расстояния между точками считаются в миллиметрах с учётом анизотропии пикселя.


def dice(a: np.ndarray, b: np.ndarray) -> float:
    """Коэффициент Дайса двух бинарных масок одинаковой формы."""
    a = np.asarray(a, dtype=bool)
    b = np.asarray(b, dtype=bool)
    if a.shape != b.shape:
        raise ValueError(f"формы масок не совпадают: {a.shape} и {b.shape}")
    total = int(a.sum()) + int(b.sum())
    if total == 0:
        return float("nan")  # обе пустые — метрика не определена
    return float(2 * int((a & b).sum()) / total)


def iou(a: np.ndarray, b: np.ndarray) -> float:
    """Intersection over Union двух бинарных масок одинаковой формы."""
    a = np.asarray(a, dtype=bool)
    b = np.asarray(b, dtype=bool)
    if a.shape != b.shape:
        raise ValueError(f"формы масок не совпадают: {a.shape} и {b.shape}")
    union = int((a | b).sum())
    if union == 0:
        return float("nan")
    return float(int((a & b).sum()) / union)


def keypoint_distance(
    pred: tuple[float, float] | None,
    true: tuple[float, float] | None,
    spacing_mm: tuple[float, float] = (1.05, 0.60),
) -> float:
    """Расстояние между точками в миллиметрах. Точка задаётся как (x, y) в пикселях.

    `spacing_mm` — физический размер пикселя (по Y, по X). Без этой поправки
    расстояние в пикселях не имеет физического смысла: пиксель прямоугольный.
    """
    if pred is None or true is None:
        return float("nan")  # точка не найдена — в среднее не попадает, считается отдельно
    sy, sx = spacing_mm
    dx = (float(pred[0]) - float(true[0])) * sx
    dy = (float(pred[1]) - float(true[1])) * sy
    return float(np.hypot(dx, dy))


def _percentile_ci(values: np.ndarray, stat: str = "median", n: int = 2000, seed: int = 0) -> tuple[float, float]:
    """95% ДИ бутстрэпом для медианы или среднего."""
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    if len(v) < 3:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    fn = np.median if stat == "median" else np.mean
    boot = [float(fn(rng.choice(v, size=len(v), replace=True))) for _ in range(n)]
    lo, hi = np.percentile(boot, (2.5, 97.5))
    return float(lo), float(hi)


def summarize_localization(values: np.ndarray | list[float], stat: str = "median", seed: int = 0) -> dict:
    """Сводка по набору значений одной метрики локализации: центр, разброс, 95% ДИ.

    `n_missing` — сколько раз величину не удалось посчитать (структура не найдена).
    Пропуски не усредняются молча: их доля выводится рядом с метрикой.
    """
    v = np.asarray(list(values), dtype=float)
    ok = v[~np.isnan(v)]
    if len(ok) == 0:
        return {"n": int(len(v)), "n_measured": 0, "n_missing": int(len(v))}
    lo, hi = _percentile_ci(ok, stat=stat, seed=seed)
    return {
        "n": int(len(v)),
        "n_measured": int(len(ok)),
        "n_missing": int(len(v) - len(ok)),
        "median": float(np.median(ok)),
        "mean": float(ok.mean()),
        "p10": float(np.percentile(ok, 10)),
        "p90": float(np.percentile(ok, 90)),
        f"{stat}_ci95": [round(lo, 4), round(hi, 4)],
    }


# --- пропускная способность ---------------------------------------------------


def throughput(times_sec: np.ndarray | list[float], n_total: int, images_per_study: int = 3) -> dict:
    """Время обработки и доля успешно обработанных файлов — обе метрики из ТЗ.

    `times_sec` — время на успешно обработанные снимки, `n_total` — сколько файлов
    подавалось на вход (включая те, что обработать не удалось).
    """
    t = np.asarray(list(times_sec), dtype=float)
    t = t[~np.isnan(t)]
    n_ok = int(len(t))
    out = {
        "файлов": int(n_total),
        "успешно_обработано": n_ok,
        "доля_успеха": round(n_ok / n_total, 4) if n_total else float("nan"),
    }
    if n_ok:
        out.update(
            {
                "время_мс_медиана": round(float(np.median(t)) * 1000, 1),
                "время_мс_p95": round(float(np.percentile(t, 95)) * 1000, 1),
                "время_мс_max": round(float(t.max()) * 1000, 1),
                "время_на_исследование_сек_p95": round(float(np.percentile(t, 95)) * images_per_study, 3),
                "лимит_ТЗ_сек": 180,
            }
        )
    return out
