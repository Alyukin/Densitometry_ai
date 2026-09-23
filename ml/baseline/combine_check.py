"""Как складывать оценки правил в quality_prob — проверка вложенной кросс-валидацией.

От способа сложения зависят только ROC и PR AUC класса качества: класс и нарушения
задают сработавшие решающие проверки, и они здесь не меняются.

Варианты объявлены заранее, их четыре:

* `max` — максимум по решающим проверкам;
* `mean` — среднее по решающим;
* `noisy_or` — 1 - prod(1 - s) по решающим;
* `mean_all` — среднее по всем проверкам, включая справочные.

На каждом внешнем фолде пороги подбираются на остальных четырёх, вариант выбирается
по AUC на внутренних OOF этих же четырёх фолдов и применяется к внешнему. Так оценка
учитывает, что вариант выбирали: это не лучший из четырёх задним числом.

Результат (23.09.2026) записан в `dxaqc.rules.QUALITY_SCORE`:

* позвоночник — `mean_all` выбран на всех пяти фолдах, AUC 0.690 против 0.618 у `max`,
  парный бутстрэп разницы +0.071 [+0.008; +0.136];
* бедро — выбор неустойчив (max, mean, mean_all), процедура проигрывает `max`
  (0.715 против 0.743), остаётся `max`.

    python ml/baseline/combine_check.py --features /tmp/features.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend" / "app" / "processing"))
sys.path.insert(0, str(ROOT / "ml"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from calibrate import fit_rules, load_features, predict_rows  # noqa: E402
from dxaqc.rules import REGION_FEMUR, REGION_SPINE  # noqa: E402

from train.metrics import bootstrap_ci, roc_auc  # noqa: E402

COMBS = ("max", "mean", "noisy_or", "mean_all")


def combine(verdict, name: str) -> float:  # noqa: ANN001
    decisive = [c.score for c in verdict.checks if c.decides]
    every = [c.score for c in verdict.checks]
    if name == "max":
        return max(decisive, default=0.0)
    if name == "mean":
        return float(np.mean(decisive)) if decisive else 0.0
    if name == "noisy_or":
        return 1.0 - float(np.prod([1.0 - s for s in decisive])) if decisive else 0.0
    return float(np.mean(every)) if every else 0.0


def as_output(verdict, name: str) -> float:  # noqa: ANN001
    """Как значение попадёт в выгрузку: больше 0.5 тогда и только тогда, когда класс 1."""
    s = combine(verdict, name)
    return 0.5 + 0.5 * s if verdict.quality_class else 0.5 * s


def _oof(rows: list[dict], folds: list[int], fit_args: tuple) -> dict:
    out = {}
    for f in folds:
        cfg = fit_rules([r for r in rows if r["_fold"] != f], *fit_args)
        test = [r for r in rows if r["_fold"] == f]
        for r, v in zip(test, predict_rows(test, cfg), strict=True):
            out[id(r)] = v
    return out


def check_region(rows: list[dict], region: str, fit_args: tuple) -> None:
    folds = sorted({r["_fold"] for r in rows})
    in_region = [r for r in rows if r["region"] == region]
    y = np.array([r["quality_class"] == "1" for r in in_region], dtype=int)

    oof = _oof(rows, folds, fit_args)
    fixed = {c: np.array([as_output(oof[id(r)], c) for r in in_region]) for c in COMBS}

    nested: dict[int, float] = {}
    chosen = []
    for f in folds:
        inner_rows = [r for r in rows if r["_fold"] != f]
        inner = _oof(inner_rows, [g for g in folds if g != f], fit_args)
        inner_region = [r for r in inner_rows if r["region"] == region]
        yi = np.array([r["quality_class"] == "1" for r in inner_region], dtype=int)
        best = max(COMBS, key=lambda c: roc_auc(yi, np.array([as_output(inner[id(r)], c) for r in inner_region])))
        chosen.append(best)
        cfg = fit_rules(inner_rows, *fit_args)
        test = [r for r in rows if r["_fold"] == f and r["region"] == region]
        for r, v in zip(test, predict_rows(test, cfg), strict=True):
            nested[id(r)] = as_output(v, best)
    p = np.array([nested[id(r)] for r in in_region])

    print(f"\n=== {region}: n={len(y)}, нарушений {int(y.sum())} ===")
    for c in COMBS:
        print(f"  {c:9} AUC {roc_auc(y, fixed[c]):.3f}  (фиксированный вариант)")
    lo, hi = bootstrap_ci(y, p, "roc_auc")
    print(f"  выбор по фолдам: {', '.join(chosen)}")
    print(f"  вложенная CV: AUC {roc_auc(y, p):.3f} [{lo:.3f}; {hi:.3f}]")

    rng = np.random.default_rng(0)
    diff = []
    for _ in range(2000):
        b = rng.integers(0, len(y), len(y))
        if y[b].sum() in (0, len(b)):
            continue
        diff.append(roc_auc(y[b], p[b]) - roc_auc(y[b], fixed["max"][b]))
    print(
        f"  вложенная CV минус max: {np.mean(diff):+.3f} "
        f"[{np.percentile(diff, 2.5):+.3f}; {np.percentile(diff, 97.5):+.3f}]"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features", default="/tmp/features.csv")
    ap.add_argument("--min-specificity", type=float, default=0.70)
    args = ap.parse_args()
    rows = [r for r in load_features(Path(args.features)) if r["_fold"] >= 0]
    fit_args = (args.min_specificity, 0.02, 2)
    for region in (REGION_SPINE, REGION_FEMUR):
        check_region(rows, region, fit_args)


if __name__ == "__main__":
    main()
