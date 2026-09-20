"""Метрики локализации по ТЗ: Dice, IoU, расстояние между ключевыми точками.

Экспертных масок и ключевых точек в выгрузке нет — размечены только класс качества и
тип нарушения. Поэтому истина берётся из двух независимых источников, и каждый честно
подписан в отчёте:

1. **Точность на фантомах.** Синтетические снимки строятся по формулам, поэтому
   положение позвоночного столба, оси диафиза и малого вертела известно точно.
   Это настоящие Dice/IoU/keypoint distance относительно истины, но истина —
   идеализированная анатомия, а не живой пациент.

2. **Повторяемость на реальных снимках.** К снимку применяется известное малое
   изменение (сдвиг кадра, шум детектора, поворот на 1°), локализация считается
   заново и приводится к исходным координатам. Dice/IoU и смещение точек показывают,
   насколько устойчив детектор на настоящих данных. Это не точность, а
   воспроизводимость — и в отчёте она названа именно так.

Дополнительно считается точность определения анатомической области и стороны — это
локализация, для которой размеченная истина в выгрузке есть.

Запуск:

    python ml/baseline/localization.py                      # фантомы + реальные снимки
    python ml/baseline/localization.py --no-real            # только фантомы
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import scipy.ndimage as ndi
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "backend" / "app" / "processing"))
sys.path.insert(0, str(ROOT / "ml"))

from app.scripts.generate_samples import (  # noqa: E402
    _hip_image,
    _spine_image,
    hip_truth,
    spine_truth,
)
from dxaqc.femur import measure_femur  # noqa: E402
from dxaqc.image import PIXEL_MM_X, PIXEL_MM_Y  # noqa: E402
from dxaqc.region import REGION_SPINE, detect  # noqa: E402
from dxaqc.spine import measure_spine  # noqa: E402

from train.metrics import dice, iou, keypoint_distance, summarize_localization  # noqa: E402

SPACING = (PIXEL_MM_Y, PIXEL_MM_X)


# --- перевод overlay в маску --------------------------------------------------


def _interp_rows(ys: np.ndarray, xs: np.ndarray, y_lo: int, y_hi: int) -> tuple[np.ndarray, np.ndarray]:
    """Полилиния, заданная разреженными точками, -> значение в каждой строке."""
    rows = np.arange(y_lo, y_hi + 1)
    return rows, np.interp(rows, ys, xs)


def mask_from_overlay(overlay: dict, shape: tuple[int, int]) -> np.ndarray | None:
    """Маска найденной структуры: столб позвонков или огибающая бедренной кости."""
    h, w = shape
    mask = np.zeros((h, w), dtype=bool)
    if overlay.get("column_left") and overlay.get("column_right"):
        left = np.array(overlay["column_left"], dtype=float)  # [[x, y], ...]
        right = np.array(overlay["column_right"], dtype=float)
        ys, xs_l = left[:, 1], left[:, 0]
        xs_r = right[:, 0]
    elif overlay.get("femur_contour"):
        c = np.array(overlay["femur_contour"], dtype=float)  # [[left, right, y], ...]
        ys, xs_l, xs_r = c[:, 2], c[:, 0], c[:, 1]
    else:
        return None
    if len(ys) < 2:
        return None
    y_lo, y_hi = int(round(ys.min())), int(round(ys.max()))
    y_lo, y_hi = max(0, y_lo), min(h - 1, y_hi)
    rows, ll = _interp_rows(ys, xs_l, y_lo, y_hi)
    _, rr = _interp_rows(ys, xs_r, y_lo, y_hi)
    for y, a, b in zip(rows, ll, rr, strict=True):
        x0, x1 = int(round(min(a, b))), int(round(max(a, b)))
        mask[int(y), max(0, x0) : min(w, x1 + 1)] = True
    return mask


def _line_deviation_mm(seg: list | None, true_x, spacing=SPACING) -> float:
    """Среднее боковое отклонение отрезка оси от истинной оси, в миллиметрах."""
    if not seg or len(seg) != 2:
        return float("nan")
    devs = [abs(float(x) - float(true_x(y))) * spacing[1] for x, y in seg]
    return float(np.mean(devs))


# --- 1. точность на фантомах с известной истиной ------------------------------


def phantom_spine(n: int = 40, seed: int = 0) -> dict:
    rng_master = np.random.default_rng(seed)
    d, j, axis_err = [], [], []
    for k in range(n):
        tilt = float(rng_master.uniform(-0.12, 0.12))
        iliac = bool(k % 3)
        img = _spine_image(np.random.default_rng(1000 + k), tilt=tilt, iliac=iliac)
        truth = spine_truth(tilt=tilt)
        m = measure_spine(img)
        if not m.ok:
            d.append(np.nan), j.append(np.nan), axis_err.append(np.nan)
            continue
        pred = mask_from_overlay(m.overlay, img.shape)
        d.append(dice(pred, truth["column"]) if pred is not None else np.nan)
        j.append(iou(pred, truth["column"]) if pred is not None else np.nan)
        axis_err.append(_line_deviation_mm(m.overlay.get("axis"), truth["axis_x"]))
    return {
        "Dice, столб позвонков": summarize_localization(d),
        "IoU, столб позвонков": summarize_localization(j),
        "Отклонение оси позвоночника, мм": summarize_localization(axis_err),
    }


def phantom_femur(n: int = 40, seed: int = 0) -> dict:
    rng_master = np.random.default_rng(seed)
    d, j, shaft_err, troch_err, lt_err = [], [], [], [], []
    for k in range(n):
        tilt = float(rng_master.uniform(0.0, 0.30))
        lt = float(rng_master.uniform(0.25, 0.8))
        img = _hip_image(np.random.default_rng(2000 + k), shaft_tilt=tilt, lesser_troch=lt)
        truth = hip_truth(shaft_tilt=tilt, lesser_troch=lt)
        m = measure_femur(img, side="right")
        if not m.ok:
            for arr in (d, j, shaft_err, troch_err, lt_err):
                arr.append(np.nan)
            continue
        pred = mask_from_overlay(m.overlay, img.shape)
        d.append(dice(pred, truth["femur"]) if pred is not None else np.nan)
        j.append(iou(pred, truth["femur"]) if pred is not None else np.nan)
        shaft_err.append(_line_deviation_mm(m.overlay.get("shaft_axis"), truth["shaft_x"]))
        # «уровень вертелов» алгоритм определяет как самое широкое место кости;
        # эталон считается по тому же определению, но на истинной геометрии —
        # так сравниваются две величины с одинаковым смыслом, а не разные точки
        tp = m.overlay.get("trochanter")
        y_true = float(np.argmax(truth["femur"].sum(axis=1)))
        troch_err.append(abs(float(tp[1]) - y_true) * SPACING[0] if tp else np.nan)
        lt_err.append(keypoint_distance(m.overlay.get("lesser_trochanter"), truth["lesser_trochanter"], SPACING))
    out = {
        "Dice, огибающая бедра": summarize_localization(d),
        "IoU, огибающая бедра": summarize_localization(j),
        "Отклонение оси диафиза, мм": summarize_localization(shaft_err),
        "Ошибка уровня вертелов, мм": summarize_localization(troch_err),
        "Ошибка положения малого вертела, мм": summarize_localization(lt_err),
    }
    if out["Ошибка положения малого вертела, мм"].get("n_measured", 0) == 0:
        out["примечание"] = (
            "малый вертел на фантоме не измеряется: пропорции фантома сжаты, и в зоне "
            "поиска (3 см над верхом диафиза) оказывается шейка, из-за чего срабатывает "
            "защита от недостоверного измерения. Точность по этой точке на фантоме не "
            "определена; её повторяемость меряется на реальных снимках"
        )
    return out


# --- 2. повторяемость на реальных снимках -------------------------------------


def _perturb(img: np.ndarray, kind: str, rng: np.random.Generator) -> tuple[np.ndarray, tuple[float, float], float]:
    """Известное малое изменение снимка. Возвращает (кадр, сдвиг (dx, dy), угол°)."""
    a = img.astype(np.float32)
    if kind == "сдвиг кадра":
        # дробный сдвиг: целый пиксель детектор увидел бы как тот же самый снимок
        dx, dy = 2.5, -1.5
        out = ndi.shift(a, (dy, dx), order=1, mode="constant", cval=0.0)
        return np.clip(out, 0, 255).astype(np.uint8), (dx, dy), 0.0
    if kind == "шум детектора":
        noise = rng.normal(0.0, 2.0, a.shape).astype(np.float32)
        out = np.where(a > 0, a * 1.03 + noise, 0.0)
        return np.clip(out, 0, 252).astype(np.uint8), (0.0, 0.0), 0.0
    if kind == "поворот 1°":
        ang = 1.0
        out = ndi.rotate(a, ang, reshape=False, order=1, mode="constant", cval=0.0)
        return np.clip(out, 0, 255).astype(np.uint8), (0.0, 0.0), ang
    raise ValueError(kind)


def _unwarp_mask(mask: np.ndarray, shift: tuple[float, float], angle: float) -> np.ndarray:
    """Возвращает маску из координат изменённого кадра в исходные."""
    a = mask.astype(np.float32)
    if angle:
        a = ndi.rotate(a, -angle, reshape=False, order=0, mode="constant", cval=0.0)
    if shift != (0.0, 0.0):
        a = ndi.shift(a, (-shift[1], -shift[0]), order=0, mode="constant", cval=0.0)
    return a > 0.5


def _unwarp_point(pt, shift: tuple[float, float], angle: float, shape: tuple[int, int]):
    if pt is None:
        return None
    x, y = float(pt[0]) - shift[0], float(pt[1]) - shift[1]
    if angle:
        cy, cx = (shape[0] - 1) / 2.0, (shape[1] - 1) / 2.0
        t = np.deg2rad(angle)
        dx, dy = x - cx, y - cy
        # ndi.rotate на угол +ang поворачивает содержимое; возвращаем обратно
        x = cx + dx * np.cos(t) - dy * np.sin(t)
        y = cy + dx * np.sin(t) + dy * np.cos(t)
    return (x, y)


KEYPOINTS = ("shaft_top", "trochanter", "neck", "lesser_trochanter")


def repeatability(data_dir: Path, limit: int | None = None, seed: int = 0) -> dict:
    rows = [r for r in csv.DictReader(open(data_dir / "dataset.csv", encoding="utf-8")) if r["quality_class"] != ""]
    if limit:
        rows = rows[:limit]
    kinds = ("сдвиг кадра", "шум детектора", "поворот 1°")
    fields = ("dice", "iou", "axis_spine_mm", "axis_femur_mm", "keypoints_mm")
    acc: dict[str, dict[str, list]] = {k: {f: [] for f in fields} for k in kinds}
    rng = np.random.default_rng(seed)
    for r in rows:
        img = np.asarray(Image.open(data_dir / r["png_path"]).convert("L"))
        spine = r["region"] == REGION_SPINE
        base = measure_spine(img) if spine else measure_femur(img, side=r["side"])
        if not base.ok:
            continue
        m0 = mask_from_overlay(base.overlay, img.shape)
        axis_key = "axis" if spine else "shaft_axis"
        axis_field = "axis_spine_mm" if spine else "axis_femur_mm"
        for kind in kinds:
            img2, shift, angle = _perturb(img, kind, rng)
            m = measure_spine(img2) if spine else measure_femur(img2, side=r["side"])
            a = acc[kind]
            if not m.ok:
                # сорвавшееся измерение — тоже результат: считается как пропуск
                a["dice"].append(np.nan), a["iou"].append(np.nan), a[axis_field].append(np.nan)
                continue
            m1 = mask_from_overlay(m.overlay, img2.shape)
            if m0 is not None and m1 is not None:
                m1b = _unwarp_mask(m1, shift, angle)[: m0.shape[0], : m0.shape[1]]
                if m1b.shape == m0.shape:
                    a["dice"].append(dice(m0, m1b))
                    a["iou"].append(iou(m0, m1b))
            s0, s1 = base.overlay.get(axis_key), m.overlay.get(axis_key)
            if s0 and s1:
                back = [_unwarp_point(p, shift, angle, img.shape) for p in s1]
                a[axis_field].append(
                    float(np.mean([keypoint_distance(b, p, SPACING) for b, p in zip(back, s0, strict=True)]))
                )
            if not spine:
                for key in KEYPOINTS:
                    p0, p1 = base.overlay.get(key), m.overlay.get(key)
                    if p0 and p1:
                        a["keypoints_mm"].append(
                            keypoint_distance(_unwarp_point(p1, shift, angle, img.shape), p0, SPACING)
                        )
    out = {}
    for kind, a in acc.items():
        out[kind] = {
            "Dice": summarize_localization(a["dice"]),
            "IoU": summarize_localization(a["iou"]),
            "Смещение оси позвоночника, мм": summarize_localization(a["axis_spine_mm"]),
            "Смещение оси диафиза, мм": summarize_localization(a["axis_femur_mm"]),
            "Смещение ориентиров бедра, мм": summarize_localization(a["keypoints_mm"]),
        }
    return out


# --- 3. точность определения области и стороны (истина размечена) -------------


def region_accuracy(data_dir: Path) -> dict:
    rows = [r for r in csv.DictReader(open(data_dir / "dataset.csv", encoding="utf-8")) if r["quality_class"] != ""]
    n_reg = n_reg_ok = n_side = n_side_ok = 0
    for r in rows:
        img = np.asarray(Image.open(data_dir / r["png_path"]).convert("L"))
        region, side, _ = detect(img)
        n_reg += 1
        n_reg_ok += int(region == r["region"])
        if r["side"]:
            n_side += 1
            n_side_ok += int(side == r["side"])
    return {
        "анатомическая область": {"n": n_reg, "верно": n_reg_ok, "accuracy": round(n_reg_ok / max(n_reg, 1), 4)},
        "сторона (в выгрузку не входит)": {
            "n": n_side,
            "верно": n_side_ok,
            "accuracy": round(n_side_ok / max(n_side, 1), 4),
        },
    }


def _print_block(title: str, block: dict) -> None:
    print(f"=== {title} ===")
    for name, m in block.items():
        if isinstance(m, str):
            print(f"  примечание: {m}")
            continue
        if "median" not in m:
            print(f"  {name[:44]:44} нет измерений ({m.get('n_missing', 0)} пропусков)")
            continue
        lo, hi = next(v for k, v in m.items() if k.endswith("_ci95"))
        miss = f", пропусков {m['n_missing']}" if m["n_missing"] else ""
        print(f"  {name[:44]:44} медиана {m['median']:7.3f}  95% ДИ [{lo:.3f}; {hi:.3f}]  n={m['n_measured']}{miss}")
    print()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default=str(ROOT / "ml" / "data" / "processed"))
    ap.add_argument("--out", default=str(ROOT / "ml" / "baseline" / "localization_metrics.json"))
    ap.add_argument("--phantoms", type=int, default=40, help="сколько фантомов каждой области")
    ap.add_argument("--limit", type=int, default=0, help="ограничить число реальных снимков (0 — все)")
    ap.add_argument("--no-real", action="store_true", help="только фантомы, без реальных снимков")
    args = ap.parse_args()

    res: dict = {
        "точность на фантомах (истина известна точно)": {
            "Поясничный отдел позвоночника": phantom_spine(args.phantoms),
            "Проксимальный отдел бедра": phantom_femur(args.phantoms),
        }
    }
    _print_block(
        "фантомы: позвоночник", res["точность на фантомах (истина известна точно)"]["Поясничный отдел позвоночника"]
    )
    _print_block("фантомы: бедро", res["точность на фантомах (истина известна точно)"]["Проксимальный отдел бедра"])

    data_dir = Path(args.data)
    if not args.no_real and (data_dir / "dataset.csv").exists():
        rep = repeatability(data_dir, limit=args.limit or None)
        res["повторяемость на реальных снимках"] = rep
        for kind, block in rep.items():
            _print_block(f"повторяемость: {kind}", block)
        res["определение области и стороны (истина размечена)"] = region_accuracy(data_dir)
        print("=== определение области и стороны ===")
        for name, m in res["определение области и стороны (истина размечена)"].items():
            print(f"  {name[:44]:44} {m['верно']}/{m['n']}  accuracy {m['accuracy']:.4f}")
        print()
    elif not args.no_real:
        print(f"Реальной выгрузки нет ({data_dir}/dataset.csv) — посчитаны только фантомы.\n")

    Path(args.out).write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"метрики локализации -> {args.out}")


if __name__ == "__main__":
    main()
