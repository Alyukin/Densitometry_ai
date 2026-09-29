"""Этап 7: гибрид правил и модели по К3, вложенная кросс-валидация.

Протокол объявлен до расчёта — OVERVIEW.md, К2, «Резервная конфигурация 4 и гибрид
(27.09)». Коротко:

* модель — конфигурация 4: замороженный xrv-densenet121 384×320 + логистическая
  регрессия на каждую задачу, ровно как в `train/probe.py` (C = 0.1, порог — вложенной
  CV внутри обучающих фолдов при специфичности ≥ 0.70, оценки приведены к порогу 0.5);
* правила — как в сервисе и в `baseline/calibrate.py`, пороги подбираются без
  отложенного фолда;
* варианты К3: только правила, среднее оценок, OR, только модель — выбираются по
  области, по macro-F1 видов нарушений, при равенстве — первый в этом списке;
* внешний фолд k: обе составляющие учатся на остальных фолдах, вариант выбирается по их
  внутренним OOF (составляющие переобучаются без внутреннего фолда) и применяется к k;
* задачи, где правила объективно сильнее модели (95% ДИ разницы AUC ниже нуля),
  во всех вариантах остаются за правилами;
* модель входит в вердикт области, только если выбор на всех пяти фолдах — не «только
  правила», macro-F1 вложенной CV выше, чем у правил, и специфичность каждого вида
  нарушения не ниже 0.70.

    python -m train.hybrid --data data/processed --features runs/v2/features.csv \\
        --probe-features runs/probe/features_xrv-densenet121_384x320.npy --out runs/hybrid
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

import numpy as np

from train import metrics as M
from train.dataset import read_dataset
from train.tasks import TASKS, targets_for

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend" / "app" / "processing"))
sys.path.insert(0, str(ROOT / "ml" / "baseline"))

logger = logging.getLogger("hybrid")

VARIANTS = ("rules", "mean", "or", "model")  # порядок — и приоритет при равенстве
VARIANT_RU = {"rules": "только правила", "mean": "среднее", "or": "OR", "model": "только модель"}
REGION_TASKS: dict[str, list[int]] = {}
for _i, _t in enumerate(TASKS):
    if _t.violation is not None:
        REGION_TASKS.setdefault(_t.region, []).append(_i)


# --- сложение и выбор ------------------------------------------------------------


def combine(variant: str, r: np.ndarray, m: np.ndarray) -> np.ndarray:
    """Оценки правил и модели на одной шкале: 0.5 — порог решения у обоих."""
    if variant == "rules":
        return r
    if variant == "model":
        return m
    if variant == "mean":
        return (r + m) / 2.0
    if variant == "or":
        return np.maximum(r, m)
    raise ValueError(variant)


def hybrid_scores(variant: str, R: np.ndarray, Mo: np.ndarray, rules_only: set[int]) -> np.ndarray:  # noqa: N803
    out = combine(variant, R, Mo).copy()
    for ti in rules_only:
        out[:, ti] = R[:, ti]
    return out


def macro_f1(S: np.ndarray, T: np.ndarray, Mk: np.ndarray, idx: np.ndarray, region: str) -> float:  # noqa: N803
    f1s = []
    for ti in REGION_TASKS[region]:
        sel = idx[Mk[idx, ti] > 0]
        f1s.append(M.binary_metrics(T[sel, ti].astype(int), S[sel, ti])["f1"] if len(sel) else 0.0)
    return float(np.mean(f1s))


def choose(
    R: np.ndarray,  # noqa: N803
    Mo: np.ndarray,  # noqa: N803
    T: np.ndarray,  # noqa: N803
    Mk: np.ndarray,  # noqa: N803
    idx: np.ndarray,
    region: str,
    rules_only: set[int],
) -> tuple[str, dict[str, float]]:
    """Вариант с лучшим macro-F1 области; при равенстве — первый в VARIANTS."""
    scores = {v: macro_f1(hybrid_scores(v, R, Mo, rules_only), T, Mk, idx, region) for v in VARIANTS}
    best = max(VARIANTS, key=lambda v: (round(scores[v], 10), -VARIANTS.index(v)))
    return best, scores


def accept(production: str, nested_f1: float, rules_f1: float, specificities: list[float]) -> tuple[bool, str]:
    """Условия К2 (27.09), при которых модель входит в вердикт области."""
    if production == "rules":
        return False, "выбор на всех пяти фолдах — только правила"
    if not nested_f1 > rules_f1:
        return False, f"macro-F1 вложенной CV {nested_f1:.3f} не выше, чем у правил ({rules_f1:.3f})"
    low = [s for s in specificities if not s >= 0.70]
    if low:
        return False, f"специфичность ниже 0.70 ({', '.join(f'{s:.2f}' for s in low)})"
    return True, "все три условия выполнены"


# --- составляющие ----------------------------------------------------------------


class Components:
    """Правила и модель, которые умеют учиться на любом наборе фолдов."""

    def __init__(self, samples: list, rows_by_id: dict, X: np.ndarray, C: float, min_spec: float) -> None:  # noqa: N803
        self.samples, self.rows_by_id, self.X, self.C, self.min_spec = samples, rows_by_id, X, C, min_spec
        self.folds = np.array([s.fold for s in samples])
        self.region = np.array([s.region for s in samples])
        tm = [targets_for(s.region, s.quality_class, s.violations, s.violations_known) for s in samples]
        self.T = np.array([t for t, _ in tm])
        self.Mk = np.array([m for _, m in tm])

    def rules(self, train: np.ndarray, test: np.ndarray) -> np.ndarray:
        from calibrate import fit_rules, predict_rows

        cfg = fit_rules([self.rows_by_id[self.samples[i].image_id] for i in train], self.min_spec, 0.02, 2)
        test_rows = [self.rows_by_id[self.samples[i].image_id] for i in test]
        out = np.full((len(test), len(TASKS)), np.nan)
        for k, v in enumerate(predict_rows(test_rows, cfg)):
            for ti, t in enumerate(TASKS):
                if t.region != test_rows[k]["region"]:
                    continue
                if t.violation is None:
                    out[k, ti] = float(v.quality_prob)
                else:
                    rel = [c.score for c in v.checks if c.violation == t.violation and c.decides]
                    out[k, ti] = max(rel) if rel else 0.0
        return out

    def model(self, train: np.ndarray, test: np.ndarray) -> np.ndarray:
        from baseline.calibrate import fit_threshold
        from train.probe import _aligned, _fit_predict

        out = np.full((len(test), len(TASKS)), np.nan)
        for ti, t in enumerate(TASKS):
            tr = train[self.Mk[train, ti] > 0]
            te_pos = np.flatnonzero(self.region[test] == t.region)
            if not len(tr) or not len(te_pos):
                continue
            y = self.T[:, ti].astype(int)
            inner = np.full(len(tr), np.nan)
            for g in sorted(set(self.folds[tr])):
                a, b = tr[self.folds[tr] != g], tr[self.folds[tr] == g]
                inner[np.isin(tr, b)] = _fit_predict(self.X[a], y[a], self.X[b], self.C)
            thr, _ = fit_threshold(inner, y[tr].astype(bool), ">", self.min_spec)
            p = _fit_predict(self.X[tr], y[tr], self.X[test[te_pos]], self.C)
            out[te_pos, ti] = _aligned(p, thr)
        return out

    def oof(self, fold_set: list[int]) -> tuple[np.ndarray, np.ndarray]:
        """OOF обеих составляющих по заданным фолдам: учатся на остальных из этого же набора."""
        R = np.full((len(self.samples), len(TASKS)), np.nan)  # noqa: N806
        Mo = np.full_like(R, np.nan)  # noqa: N806
        for f in fold_set:
            train = np.flatnonzero(np.isin(self.folds, [g for g in fold_set if g != f]))
            test = np.flatnonzero(self.folds == f)
            R[test], Mo[test] = self.rules(train, test), self.model(train, test)
        return R, Mo


# --- сравнение -------------------------------------------------------------------


def _groups_boot(groups: np.ndarray, n: int, seed: int = 0):  # noqa: ANN202
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    by = {g: np.flatnonzero(groups == g) for g in uniq}
    for _ in range(n):
        yield np.concatenate([by[g] for g in rng.choice(uniq, size=len(uniq), replace=True)])


def macro_f1_diff_ci(a, b, T, Mk, idx, region, groups, n=2000) -> tuple[float, float, float]:  # noqa: ANN001, N803
    """macro-F1(a) − macro-F1(b), парный бутстрэп по исследованиям."""
    diffs = []
    for boot in _groups_boot(groups[idx], n):
        sub = idx[boot]
        diffs.append(macro_f1(a, T, Mk, sub, region) - macro_f1(b, T, Mk, sub, region))
    d = macro_f1(a, T, Mk, idx, region) - macro_f1(b, T, Mk, idx, region)
    return float(d), float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def rules_stronger_tasks(R, Mo, T, Mk, groups) -> tuple[set[int], dict]:  # noqa: ANN001, N803
    """Задачи, где правила объективно сильнее модели: 95% ДИ разницы AUC целиком ниже нуля."""
    from train.compare import _auc_diff_ci

    stronger, diffs = set(), {}
    for ti, t in enumerate(TASKS):
        sel = np.flatnonzero((Mk[:, ti] > 0) & ~np.isnan(R[:, ti]) & ~np.isnan(Mo[:, ti]))
        y = T[sel, ti].astype(int)
        if not 0 < y.sum() < len(y):
            continue
        d, lo, hi = _auc_diff_ci(y, Mo[sel, ti], R[sel, ti], groups[sel])
        diffs[t.key] = {"model_minus_rules": d, "ci95": [lo, hi]}
        if hi < 0:
            stronger.add(ti)
    return stronger, diffs


# --- прогон ----------------------------------------------------------------------


def run(comp: Components) -> dict:
    T, Mk, folds = comp.T, comp.Mk, comp.folds  # noqa: N806
    all_folds = sorted(set(folds.tolist()))
    groups = np.array([s.study_dir for s in comp.samples])

    logger.info("OOF по всем пяти фолдам")
    R, Mo = comp.oof(all_folds)  # noqa: N806
    rules_only, auc_diffs = rules_stronger_tasks(R, Mo, T, Mk, groups)
    logger.info("за правилами во всех вариантах: %s", [TASKS[i].key for i in sorted(rules_only)] or "нет")

    nested = np.full_like(R, np.nan)
    chosen: dict[str, list[str]] = {region: [] for region in REGION_TASKS}
    for k in all_folds:
        inner = [f for f in all_folds if f != k]
        logger.info("внешний фолд %d: внутренние OOF по фолдам %s", k, inner)
        Ri, Mi = comp.oof(inner)  # noqa: N806
        train, test = np.flatnonzero(folds != k), np.flatnonzero(folds == k)
        Rk, Mok = comp.rules(train, test), comp.model(train, test)  # noqa: N806
        for region in REGION_TASKS:
            idx_in = np.flatnonzero(np.isin(folds, inner) & (comp.region == region))
            best, _ = choose(Ri, Mi, T, Mk, idx_in, region, rules_only)
            chosen[region].append(best)
            rows = np.flatnonzero(comp.region[test] == region)
            nested[test[rows]] = hybrid_scores(best, Rk, Mok, rules_only)[rows]

    result: dict = {"rules_only_tasks": [TASKS[i].key for i in sorted(rules_only)], "auc_model_minus_rules": auc_diffs}
    result["regions"] = {}
    for region in REGION_TASKS:
        idx = np.flatnonzero(comp.region == region)
        production, fixed = choose(R, Mo, T, Mk, idx, region, rules_only)
        f1_rules = macro_f1(R, T, Mk, idx, region)
        f1_nested = macro_f1(nested, T, Mk, idx, region)
        d, lo, hi = macro_f1_diff_ci(nested, R, T, Mk, idx, region, groups)
        tasks = {}
        for ti in [i for i, t in enumerate(TASKS) if t.region == region]:
            sel = idx[Mk[idx, ti] > 0]
            y = T[sel, ti].astype(int)
            tasks[TASKS[ti].key] = {
                "label": TASKS[ti].label,
                "n": int(len(y)),
                "n_pos": int(y.sum()),
                "rules": M.summarize(y, R[sel, ti]),
                "model": M.summarize(y, Mo[sel, ti]),
                "hybrid_nested": M.summarize(y, nested[sel, ti]),
            }
        specs = [tasks[TASKS[ti].key]["hybrid_nested"]["specificity"] for ti in REGION_TASKS[region]]
        ok, why = accept(production, f1_nested, f1_rules, specs)
        result["regions"][region] = {
            "chosen_by_outer_fold": chosen[region],
            "production_choice": production,
            "macro_f1_fixed_variants_optimistic": fixed,
            "macro_f1_rules": f1_rules,
            "macro_f1_model": macro_f1(Mo, T, Mk, idx, region),
            "macro_f1_hybrid_nested": f1_nested,
            "nested_minus_rules": {"value": d, "ci95": [lo, hi]},
            "model_in_verdict": ok,
            "why": why,
            "tasks": tasks,
        }
    result["_oof"] = (R, Mo, nested)
    return result


def _f(x: float) -> str:
    return f"{x:+.3f}".replace("-", "−")


def render(res: dict) -> str:
    L = ["# Этап 7: гибрид правил и модели (К3)", ""]  # noqa: N806
    L.append(
        "Модель — конфигурация 4: замороженный xrv-densenet121 384×320 + логистическая регрессия "
        "(линейный зонд). Протокол объявлен до расчёта: OVERVIEW.md, К2, 27.09. Вариант выбирается "
        "вложенной кросс-валидацией по macro-F1 видов нарушений отдельно по области. Цифры модели "
        "оптимистичны: результат зонда был известен до объявления."
    )
    L.append("")
    ro = ", ".join(res["rules_only_tasks"]) or "нет"
    L.append(f"Задачи, где правила объективно сильнее модели и во всех вариантах остаются за правилами: {ro}.")
    for region, r in res["regions"].items():
        L += ["", f"## {region}", ""]
        L.append("| | macro-F1 |")
        L.append("|---|---:|")
        L.append(f"| только правила | {r['macro_f1_rules']:.3f} |")
        L.append(f"| только модель | {r['macro_f1_model']:.3f} |")
        for v, f in r["macro_f1_fixed_variants_optimistic"].items():
            if v in ("mean", "or"):
                L.append(f"| {VARIANT_RU[v]} (фиксированный вариант, оптимистично) | {f:.3f} |")
        n = r["nested_minus_rules"]
        L.append(
            f"| **гибрид, вложенная CV** | **{r['macro_f1_hybrid_nested']:.3f}** "
            f"({_f(n['value'])} [{_f(n['ci95'][0])}; {_f(n['ci95'][1])}]) |"
        )
        L += ["", f"Выбор по внешним фолдам: {', '.join(VARIANT_RU[c] for c in r['chosen_by_outer_fold'])}."]
        L.append(f"Выбор на всех пяти фолдах: **{VARIANT_RU[r['production_choice']]}**.")
        L += [
            "",
            "| Задача | n / нарушений | Правила Se / Sp / F1 · AUC | Модель | Гибрид |",
            "|---|---:|---:|---:|---:|",
        ]
        for t in r["tasks"].values():
            cells = [
                f"{t[k]['sensitivity']:.2f} / {t[k]['specificity']:.2f} / {t[k]['f1']:.2f} · {t[k]['roc_auc']:.2f}"
                for k in ("rules", "model", "hybrid_nested")
            ]
            L.append(f"| {t['label']} | {t['n']} / {t['n_pos']} | " + " | ".join(cells) + " |")
        verdict = "**входит в вердикт**" if r["model_in_verdict"] else "**в вердикт не входит**"
        L += ["", f"Модель {verdict}: {r['why']}."]
    return "\n".join(L) + "\n"


def write_oof(path: Path, samples: list, R, Mo, H, T, Mk) -> None:  # noqa: ANN001, N803
    cols = ["image_id", "study_dir", "fold", "region"]
    for t in TASKS:
        cols += [f"y_{t.key}", f"rules_{t.key}", f"model_{t.key}", f"hybrid_{t.key}"]
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for k, s in enumerate(samples):
            row = [s.image_id, s.study_dir, s.fold, s.region]
            for ti in range(len(TASKS)):
                on = Mk[k, ti] > 0
                row += [int(T[k, ti]) if on else ""]
                row += [f"{A[k, ti]:.4f}" if on and not np.isnan(A[k, ti]) else "" for A in (R, Mo, H)]
            w.writerow(row)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="data/processed")
    ap.add_argument("--features", default="runs/v2/features.csv", help="измерения правил (baseline/featdump.py)")
    ap.add_argument("--probe-features", default="runs/probe/features_xrv-densenet121_384x320.npy")
    ap.add_argument("--out", default="runs/hybrid")
    ap.add_argument("--C", type=float, default=0.1, help="как в train/probe.py; задано заранее")
    ap.add_argument("--min-specificity", type=float, default=0.70)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    from calibrate import load_features

    data, out = Path(args.data), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    samples = read_dataset(data / "dataset.csv", data)
    X = np.load(args.probe_features)  # noqa: N806
    if len(X) != len(samples):
        raise SystemExit(f"признаков {len(X)}, снимков {len(samples)}: пересчитайте train.probe на этом датасете")
    rows_by_id = {r["image_id"]: r for r in load_features(Path(args.features)) if r["_fold"] >= 0}
    missing = [s.image_id for s in samples if s.image_id not in rows_by_id]
    if missing:
        raise SystemExit(f"нет измерений правил для {len(missing)} снимков, например {missing[:3]}")

    comp = Components(samples, rows_by_id, X, args.C, args.min_specificity)
    res = run(comp)
    R, Mo, H = res.pop("_oof")  # noqa: N806
    write_oof(out / "oof_hybrid.csv", samples, R, Mo, H, comp.T, comp.Mk)
    res["config"] = {k: str(v) for k, v in vars(args).items()}
    (out / "hybrid.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    md = render(res)
    (out / "HYBRID.md").write_text(md, encoding="utf-8")
    print(md)
    print(f"-> {out / 'HYBRID.md'}, {out / 'hybrid.json'}, {out / 'oof_hybrid.csv'}")


if __name__ == "__main__":
    main()
