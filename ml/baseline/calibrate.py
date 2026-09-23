"""Калибровка порогов правил и честная оценка baseline на отложенных фолдах.

Как считается «честно»:

* фолды — те же, что в датасете: группировка по исследованиям, один пациент целиком
  в одном фолде;
* для каждого фолда пороги подбираются ТОЛЬКО на остальных четырёх, предсказание
  делается на отложенном;
* итоговые метрики считаются по out-of-fold предсказаниям, то есть каждый снимок
  оценён моделью, которая его не видела;
* финальные пороги для сервиса подбираются на всех данных и записываются в
  thresholds.json — это рабочая точка, а метрики к ней берутся из OOF.

Пороги подбираются только там, где ТЗ не задаёт число. Там, где задаёт (наклон оси
5°, поля 3 см и 2 см), в отчёт выводятся обе версии: буквальная из ТЗ и подобранная.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend" / "app" / "processing"))
sys.path.insert(0, str(ROOT / "ml"))

from dxaqc.analyze import STRUCTURE_REASONS  # noqa: E402
from dxaqc.rules import (  # noqa: E402
    CLOSED_VIOLATIONS,
    RULES,
    TZ_THRESHOLDS,
    Verdict,
    evaluate,
    structures_not_found,
)

from train.metrics import summarize, throughput  # noqa: E402

VIOLATIONS = {region: list(viols) for region, viols in CLOSED_VIOLATIONS.items()}


def load_features(path: Path) -> list[dict]:
    rows = []
    for r in csv.DictReader(open(path, encoding="utf-8")):
        d = dict(r)
        d["_viol"] = {v for v in r["violation_type"].split(";") if v}
        d["_fold"] = int(r["fold"])
        rows.append(d)
    return rows


def _values(rows: list[dict], feature: str) -> np.ndarray:
    out = []
    for r in rows:
        try:
            out.append(float(r.get(feature, "") or 0.0))
        except ValueError:
            out.append(0.0)
    return np.array(out, dtype=float)


def balanced_accuracy(y: np.ndarray, pred: np.ndarray) -> float:
    tp = int((pred & y).sum())
    fn = int((~pred & y).sum())
    tn = int((~pred & ~y).sum())
    fp = int((pred & ~y).sum())
    se = tp / max(tp + fn, 1)
    spx = tn / max(tn + fp, 1)
    return (se + spx) / 2


def fit_threshold(values: np.ndarray, y: np.ndarray, op: str, min_specificity: float = 0.0) -> tuple[float, float]:
    """Порог, максимизирующий balanced accuracy.

    Берётся не точечный максимум, а центр лучшего плато: кривая BA по сетке порогов
    сглаживается, и выбирается вершина сглаженной кривой. На выборке в полсотни
    примеров точечный максимум смещается от фолда к фолду и не переносится на
    отложенные данные, а центр плато устойчив.
    """
    if y.sum() == 0 or (~y).sum() == 0:
        return (float("inf") if op == ">" else float("-inf")), 0.5
    grid = np.quantile(values, np.linspace(0.01, 0.99, 99))
    grid = np.unique(grid)
    if len(grid) < 3:
        return (float("inf") if op == ">" else float("-inf")), 0.5

    ba = np.zeros(len(grid))
    ok = np.ones(len(grid), dtype=bool)
    for i, t in enumerate(grid):
        pred = values > t if op == ">" else values < t
        tn = int((~pred & ~y).sum())
        fp = int((pred & ~y).sum())
        if (tn / max(tn + fp, 1)) < min_specificity:
            ok[i] = False
        ba[i] = balanced_accuracy(y, pred)

    k = max(3, len(grid) // 12)
    kern = np.ones(k) / k
    ba_s = np.convolve(ba, kern, mode="same")
    ba_s[~ok] = -1.0
    if ba_s.max() < 0:
        return (float("inf") if op == ">" else float("-inf")), 0.5
    i_best = int(np.argmax(ba_s))
    return float(grid[i_best]), float(ba[i_best])


def fit_rules(rows: list[dict], min_specificity: float, min_gain: float = 0.02, max_rules: int = 2) -> dict:
    """Подбор порогов и отбор правил (жадно, по OR-комбинации) на переданных данных.

    Отбор намеренно консервативный: правило включается, только если заметно
    улучшает balanced accuracy, и на одно нарушение берётся не больше `max_rules`.
    На выборке с 6-9 позитивами жадный отбор без этих ограничений подбирает
    комбинацию под обучающий фолд, и она не переносится на отложенный.
    """
    cfg: dict[str, dict] = {}
    for region, viols in VIOLATIONS.items():
        sub = [r for r in rows if r["region"] == region and r["ok"] == "True"]
        if not sub:
            continue
        for viol in viols:
            y = np.array([viol in r["_viol"] for r in sub])
            rule_ids = [
                rid
                for rid, s in RULES.items()
                if s["region"] == region and s["violation"] == viol and s.get("decides", True)
            ]
            fitted = {}
            for rid in rule_ids:
                spec = RULES[rid]
                v = _values(sub, spec["feature"])
                thr, ba = fit_threshold(v, y, spec["op"], min_specificity)
                fitted[rid] = {"threshold": thr, "ba": ba, "values": v}
            # жадный отбор: добавляем правило, если OR-комбинация улучшает BA
            chosen: list[str] = []
            cur = np.zeros(len(sub), dtype=bool)
            best = balanced_accuracy(y, cur)
            while True:
                gain = None
                for rid in rule_ids:
                    if rid in chosen:
                        continue
                    spec = RULES[rid]
                    v = fitted[rid]["values"]
                    t = fitted[rid]["threshold"]
                    pred = cur | ((v > t) if spec["op"] == ">" else (v < t))
                    ba = balanced_accuracy(y, pred)
                    if gain is None or ba > gain[0]:
                        gain = (ba, rid, pred)
                if gain is None or gain[0] <= best + min_gain or len(chosen) >= max_rules:
                    break
                best, rid, cur = gain[0], gain[1], gain[2]
                chosen.append(rid)
            for rid in rule_ids:
                v = fitted[rid]["values"]
                spread = float(np.percentile(v, 90) - np.percentile(v, 10))
                cfg[rid] = {
                    "threshold": fitted[rid]["threshold"],
                    "soft_width": max(spread / 6.0, 1e-3),
                    "enabled": rid in chosen,
                    "train_ba": round(fitted[rid]["ba"], 4),
                }
                # Проверку, которой требует ТЗ, отбор может не взять в вердикт — тогда
                # она остаётся справочной, с порогом ТЗ, если он там записан. На
                # вердикт и метрики это не влияет: у бедра quality_prob — максимум по
                # решающим проверкам.
                if rid not in chosen and RULES[rid].get("reference_when_off"):
                    cfg[rid].update(
                        enabled=True,
                        decides=False,
                        threshold=float(TZ_THRESHOLDS.get(rid, fitted[rid]["threshold"])),
                    )

    # Справочные проверки (decides=False) в вердикт не входят, но ТЗ требует их
    # выполнять и показывать, поэтому порог для них тоже нужен: берём записанный
    # в ТЗ, а если его там нет — подобранный.
    for rid, spec in RULES.items():
        if spec.get("decides", True) or rid in cfg:
            continue
        sub = [r for r in rows if r["region"] == spec["region"] and r["ok"] == "True"]
        if not sub:
            continue
        v = _values(sub, spec["feature"])
        y = np.array([spec["violation"] in r["_viol"] for r in sub])
        thr = TZ_THRESHOLDS.get(rid)
        if thr is None:
            thr, _ = fit_threshold(v, y, spec["op"], min_specificity)
        spread = float(np.percentile(v, 90) - np.percentile(v, 10))
        cfg[rid] = {
            "threshold": float(thr),
            "soft_width": max(spread / 6.0, 1e-3),
            "enabled": True,
            "decides": False,
        }
    return cfg


def predict_rows(rows: list[dict], cfg: dict) -> list[Verdict]:
    out = []
    for r in rows:
        if r.get("ok") != "True" and r.get("reason") in STRUCTURE_REASONS:
            # как в сервисе: структуры не найдены — это вердикт, а не отказ
            out.append(structures_not_found(r["region"], r["reason"]))
            continue
        meas = {}
        for k, v in r.items():
            if k.startswith("_") or k in ("image_id", "study_dir", "region", "side", "violation_type", "reason", "ok"):
                continue
            try:
                meas[k] = float(v)
            except (TypeError, ValueError):
                continue
        out.append(evaluate(r["region"], meas, cfg))
    return out


def cross_validate(rows: list[dict], min_specificity: float, min_gain: float, max_rules: int) -> tuple[dict, dict]:
    """OOF-предсказания: пороги подбираются без отложенного фолда."""
    folds = sorted({r["_fold"] for r in rows if r["_fold"] >= 0})
    oof: dict[int, Verdict] = {}
    for f in folds:
        train = [r for r in rows if r["_fold"] >= 0 and r["_fold"] != f]
        test = [r for r in rows if r["_fold"] == f]
        cfg = fit_rules(train, min_specificity, min_gain, max_rules)
        for r, v in zip(test, predict_rows(test, cfg), strict=True):
            oof[id(r)] = v
    final_cfg = fit_rules([r for r in rows if r["_fold"] >= 0], min_specificity, min_gain, max_rules)
    return oof, final_cfg


def report(rows: list[dict], oof: dict[int, Verdict]) -> dict:
    res: dict = {}
    for region, viols in VIOLATIONS.items():
        sub = [r for r in rows if r["region"] == region and r["_fold"] >= 0]
        if not sub:
            continue
        block: dict = {"n": len(sub)}
        y_q = np.array([r["quality_class"] == "1" for r in sub], dtype=int)
        p_q = np.array([oof[id(r)].quality_prob for r in sub])
        block["качество (есть нарушение)"] = summarize(y_q, p_q, thr=0.5)
        f1s = []
        for viol in viols:
            y = np.array([viol in r["_viol"] for r in sub], dtype=int)
            scores = []
            for r in sub:
                v = oof[id(r)]
                # только те проверки, что реально формируют вердикт
                rel = [c.score for c in v.checks if c.violation == viol and c.decides]
                scores.append(max(rel) if rel else 0.0)
            p = np.array(scores)
            m = summarize(y, p, thr=0.5)
            block[viol] = m
            f1s.append(m["f1"])
        block["macro_f1"] = float(np.mean(f1s))
        res[region] = block
    res["пропускная способность"] = _throughput_block(rows)
    return res


def _throughput_block(rows: list[dict]) -> dict:
    """Две метрики ТЗ, которые считаются не по классам, а по всей выгрузке:
    доля успешно обработанных файлов и время обработки."""
    times = []
    n_structures = 0
    for r in rows:
        if r.get("ok") != "True":
            if r.get("reason") not in STRUCTURE_REASONS:
                continue  # отказ: вердикта нет
            n_structures += 1
        try:
            times.append(float(r.get("measure_sec", "") or "nan"))
        except ValueError:
            continue
    block = throughput(times, n_total=len(rows))
    block["из_них_структуры_не_найдены"] = n_structures
    block["что_измерялось"] = (
        "измерения по подготовленному кадру на одном ядре CPU; чтение DICOM добавляет около 8 мс на снимок"
    )
    return block


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features", default="/tmp/features.csv")
    ap.add_argument("--out-thresholds", default=str(ROOT / "backend/app/processing/dxaqc/thresholds.json"))
    ap.add_argument("--out-metrics", default=str(ROOT / "ml/baseline/metrics.json"))
    ap.add_argument(
        "--min-specificity",
        type=float,
        default=0.70,
        help="нижняя граница специфичности при подборе порога: ложные срабатывания дороже пропусков",
    )
    ap.add_argument(
        "--min-gain", type=float, default=0.02, help="минимальный прирост BA, чтобы включить ещё одно правило"
    )
    ap.add_argument("--max-rules", type=int, default=2, help="сколько правил максимум на одно нарушение")
    args = ap.parse_args()

    rows = load_features(Path(args.features))
    oof, cfg = cross_validate(rows, args.min_specificity, args.min_gain, args.max_rules)
    metrics = report(rows, oof)

    for rid, c in cfg.items():
        spec = RULES[rid]
        c["source"] = spec["source"]
        c["feature"] = spec["feature"]
        c["op"] = spec["op"]
        if rid in TZ_THRESHOLDS:
            c["threshold_tz"] = TZ_THRESHOLDS[rid]
            c["note"] = (
                f"порог ТЗ = {TZ_THRESHOLDS[rid]}, справочно: в вердикт не входит"
                if c.get("decides") is False
                else f"порог ТЗ = {TZ_THRESHOLDS[rid]}, рабочий порог подобран по разметке"
            )

    Path(args.out_thresholds).write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    Path(args.out_metrics).write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"пороги -> {args.out_thresholds}\nметрики -> {args.out_metrics}\n")
    tp = metrics.get("пропускная способность", {})
    for region, block in metrics.items():
        if region == "пропускная способность":
            continue
        print(f"=== {region} (n={block['n']}) ===")
        for name, m in block.items():
            if not isinstance(m, dict):
                continue
            print(
                f"  {name[:42]:42} se {m['sensitivity']:.2f} sp {m['specificity']:.2f} "
                f"BA {m['balanced_accuracy']:.2f} F1 {m['f1']:.2f} "
                f"AUC {m['roc_auc']:.2f} PR {m['pr_auc']:.2f} (n+={m['n_pos']})"
            )
        print(f"  macro-F1: {block['macro_f1']:.3f}\n")

    if tp:
        print("=== пропускная способность ===")
        print(f"  доля успешно обработанных: {tp['успешно_обработано']}/{tp['файлов']} ({tp['доля_успеха']:.4f})")
        if "время_мс_медиана" in tp:
            print(
                f"  время измерений: медиана {tp['время_мс_медиана']} мс, "
                f"p95 {tp['время_мс_p95']} мс, max {tp['время_мс_max']} мс"
            )
            print(
                f"  исследование из 3 снимков по p95: {tp['время_на_исследование_сек_p95']} с "
                f"при лимите ТЗ {tp['лимит_ТЗ_сек']} с\n"
            )


if __name__ == "__main__":
    main()
