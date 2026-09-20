"""Ранжирование измеримых признаков по разделяющей способности (AUC с поправкой на связи)."""

from __future__ import annotations

import argparse
import csv

import numpy as np
from scipy.stats import rankdata

SPINE = "Поясничный отдел позвоночника"
TARGETS = {
    SPINE: [
        "Некорректная укладка",
        "Не выравнена ось позвоночника",
        "Присутствуют посторонние предметы",
        "__quality__",
    ],
    "Проксимальный отдел бедра": ["Некорректная укладка", "Некорректная область интереса", "__quality__"],
}
SKIP = {
    "image_id",
    "study_dir",
    "region",
    "side",
    "fold",
    "quality_class",
    "violation_type",
    "violations_known",
    "ok",
    "reason",
    "overlay",
}


def auc(ok: np.ndarray, bad: np.ndarray) -> float:
    if len(ok) == 0 or len(bad) == 0:
        return float("nan")
    rk = rankdata(np.concatenate([ok, bad]))
    return float((rk[len(ok) :].sum() - len(bad) * (len(bad) + 1) / 2) / (len(ok) * len(bad)))


def main(path: str, top: int) -> None:
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    for region, targets in TARGETS.items():
        sub = [r for r in rows if r["region"] == region]
        if not sub:
            continue
        keys = [k for k in sub[0] if k not in SKIP and all(_is_num(r[k]) for r in sub)]
        print(f"\n{'=' * 100}\n{region}  (n={len(sub)})")
        for t in targets:
            if t == "__quality__":
                y = np.array([r["quality_class"] == "1" for r in sub])
                name = "качество (любое нарушение)"
            else:
                y = np.array([t in r["violation_type"].split(";") for r in sub])
                name = t
            if y.sum() < 3:
                continue
            scores = []
            for k in keys:
                v = np.array([float(r[k]) for r in sub])
                if np.unique(v).size < 3:
                    continue
                a = auc(v[~y], v[y])
                scores.append((max(a, 1 - a), a, k))
            scores.sort(reverse=True)
            print(f"\n  {name}: позитивов {int(y.sum())} из {len(sub)}")
            for _s, a, k in scores[:top]:
                direction = "выше => нарушение" if a > 0.5 else "ниже => нарушение"
                v = np.array([float(r[k]) for r in sub])
                print(
                    f"    {k:24} AUC {a:5.3f} ({direction:18})  "
                    f"норма med {np.median(v[~y]):8.2f}  нарушение med {np.median(v[y]):8.2f}"
                )


def _is_num(x: str) -> bool:
    try:
        float(x)
        return True
    except (TypeError, ValueError):
        return False


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features", default="/tmp/features.csv")
    ap.add_argument("--top", type=int, default=8)
    a = ap.parse_args()
    main(a.features, a.top)
