"""Модель против baseline на правилах — на одних и тех же снимках и фолдах.

Обе стороны оцениваются out-of-fold: у модели — предсказания `train.train`
(`oof_predictions.csv`), у правил — пересчёт `baseline/calibrate.py` с подбором порогов
без отложенного фолда. Снимки сравниваются попарно, только там, где цель определена у
модели (маска R3 одна на обоих), поэтому разница не объясняется разными выборками.

Доверительный интервал разницы — парный бутстрэп по исследованиям: снимки одного
исследования (одного пациента) выпадают и попадают в выборку вместе, как в фолдах.

Критерий закрытия этапа 6 (roadmap.md): macro-F1 модели выше, чем у правил, хотя бы на
0.05 хотя бы в одной области, при специфичности классов не ниже 0.70.

    python -m train.compare --run runs/cnn --features /tmp/features.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

from train import metrics as M
from train.tasks import TASKS

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend" / "app" / "processing"))
sys.path.insert(0, str(ROOT / "ml" / "baseline"))

VIOLATION_TASKS = {t.key for t in TASKS if t.violation is not None}
REGION_TASKS: dict[str, list[str]] = {}
for _t in TASKS:
    if _t.violation is not None:
        REGION_TASKS.setdefault(_t.region, []).append(_t.key)


def rules_oof(features: Path, min_specificity: float = 0.70) -> dict[str, dict[str, float]]:
    """image_id -> {задача: оценка правил 0..1} по OOF, как в calibrate.report."""
    from calibrate import cross_validate, load_features

    rows = load_features(features)
    oof, _ = cross_validate(rows, min_specificity, 0.02, 2)
    out: dict[str, dict[str, float]] = {}
    for r in rows:
        v = oof.get(id(r))
        if v is None:
            continue
        scores: dict[str, float] = {}
        for t in TASKS:
            if t.region != r["region"]:
                continue
            if t.violation is None:
                scores[t.key] = float(v.quality_prob)
            else:
                rel = [c.score for c in v.checks if c.violation == t.violation and c.decides]
                scores[t.key] = max(rel) if rel else 0.0
        out[r["image_id"]] = scores
    return out


def model_oof(run: Path, name: str = "oof_predictions.csv") -> list[dict]:
    with (run / name).open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _auc_diff_ci(y, a, b, groups, n=2000, seed=0) -> tuple[float, float, float]:  # noqa: ANN001
    """Парный бутстрэп по группам (исследованиям): AUC(a) - AUC(b)."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    idx_by = {g: np.flatnonzero(groups == g) for g in uniq}
    diffs = []
    for _ in range(n):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by[g] for g in pick])
        if y[idx].sum() in (0, len(idx)):
            continue
        diffs.append(M.roc_auc(y[idx], a[idx]) - M.roc_auc(y[idx], b[idx]))
    d = np.asarray(diffs)
    return float(M.roc_auc(y, a) - M.roc_auc(y, b)), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def compare(
    run: Path,
    features: Path,
    min_specificity: float = 0.70,
    oof: str = "oof_predictions.csv",
    rules: dict[str, dict[str, float]] | None = None,
) -> dict:
    """`rules` — готовый `rules_oof(...)`, чтобы не пересчитывать правила для каждого прогона."""
    if rules is None:
        rules = rules_oof(features, min_specificity)
    rows = model_oof(run, oof)
    res: dict = {"oof": oof, "tasks": {}, "regions": {}}
    for t in TASKS:
        sel = [
            r for r in rows if r[f"y_{t.key}"] != "" and r[f"p_{t.key}"] != "" and t.key in rules.get(r["image_id"], {})
        ]
        if not sel:
            continue
        y = np.array([int(float(r[f"y_{t.key}"])) for r in sel])
        pm = np.array([float(r[f"p_{t.key}"]) for r in sel])
        pr = np.array([rules[r["image_id"]][t.key] for r in sel])
        groups = np.array([r["study_dir"] for r in sel])
        entry = {
            "label": t.label,
            "n": int(len(y)),
            "n_pos": int(y.sum()),
            "model": M.summarize(y, pm, thr=0.5),
            "rules": M.summarize(y, pr, thr=0.5),
        }
        if 0 < y.sum() < len(y):
            d, lo, hi = _auc_diff_ci(y, pm, pr, groups)
            entry["auc_diff"] = {"value": d, "ci95": [lo, hi]}
        res["tasks"][t.key] = entry

    closed = False
    for region, keys in REGION_TASKS.items():
        present = [k for k in keys if k in res["tasks"]]
        f1_m = float(np.mean([res["tasks"][k]["model"]["f1"] for k in present]))
        f1_r = float(np.mean([res["tasks"][k]["rules"]["f1"] for k in present]))
        spec_ok = all(res["tasks"][k]["model"]["specificity"] >= min_specificity for k in present)
        gain = f1_m - f1_r
        region_closed = gain >= 0.05 and spec_ok
        closed = closed or region_closed
        res["regions"][region] = {
            "macro_f1_model": f1_m,
            "macro_f1_rules": f1_r,
            "gain": gain,
            "model_specificity_ok": spec_ok,
            "criterion_met": region_closed,
        }
    res["stage6_criterion_met"] = closed
    return res


def _fmt_ci(m: dict, key: str) -> str:
    ci = m.get(f"{key}_ci95")
    v = m.get(key, float("nan"))
    return f"{v:.2f} [{ci[0]:.2f}; {ci[1]:.2f}]" if ci else f"{v:.2f}"


def print_report(res: dict) -> None:
    print(f"{'задача':42} {'n/n+':>7}  {'AUC модели':>20}  {'AUC правил':>20}  {'разница AUC':>22}")
    for t in res["tasks"].values():
        d = t.get("auc_diff")
        diff = f"{d['value']:+.2f} [{d['ci95'][0]:+.2f}; {d['ci95'][1]:+.2f}]" if d else "—"
        print(
            f"{t['label'][:42]:42} {t['n']:>3}/{t['n_pos']:<3}  {_fmt_ci(t['model'], 'roc_auc'):>20}  "
            f"{_fmt_ci(t['rules'], 'roc_auc'):>20}  {diff:>22}"
        )
    print(f"\n{'задача':42} {'модель Se/Sp/BA/F1 при 0.5':>30}  {'правила Se/Sp/BA/F1':>26}")
    for t in res["tasks"].values():
        m, r = t["model"], t["rules"]
        fm = f"{m['sensitivity']:.2f}/{m['specificity']:.2f}/{m['balanced_accuracy']:.2f}/{m['f1']:.2f}"
        fr = f"{r['sensitivity']:.2f}/{r['specificity']:.2f}/{r['balanced_accuracy']:.2f}/{r['f1']:.2f}"
        print(f"{t['label'][:42]:42} {fm:>30}  {fr:>26}")
    print()
    for region, g in res["regions"].items():
        print(
            f"{region}: macro-F1 модель {g['macro_f1_model']:.3f}, правила {g['macro_f1_rules']:.3f}, "
            f"разница {g['gain']:+.3f}; специфичность модели >= 0.70: {'да' if g['model_specificity_ok'] else 'нет'}"
        )
    print(f"\nкритерий закрытия этапа 6 выполнен: {'да' if res['stage6_criterion_met'] else 'нет'}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="runs/cnn", help="папка прогона train.train")
    ap.add_argument("--features", default="/tmp/features.csv", help="измерения правил (baseline/featdump.py)")
    ap.add_argument("--min-specificity", type=float, default=0.70)
    ap.add_argument(
        "--oof",
        default="oof_predictions.csv",
        help="файл предсказаний модели; oof_predictions_best_epoch.csv — чтобы увидеть завышение от выбора эпохи",
    )
    args = ap.parse_args()
    res = compare(Path(args.run), Path(args.features), args.min_specificity, args.oof)
    suffix = "" if args.oof == "oof_predictions.csv" else "_" + Path(args.oof).stem.removeprefix("oof_predictions_")
    out = Path(args.run) / f"compare_baseline{suffix}.json"
    out.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print_report(res)
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
