"""Сводка этапа 6 по протоколу v2 и следующий шаг по правилу, объявленному заранее (К2).

Для каждого прогона: сравнение с правилами на тех же снимках и фолдах (`train.compare`),
остановка по фолдам, попарные разницы AUC между конфигурациями. В конце правило из К2
(`questions.md`) применяется механически, без ручного выбора.

    python -m train.summary --runs runs/v2/resnet18 runs/v2/xrv-densenet121 runs/v2/xrv-densenet121_arak \\
        --features runs/v2/features.csv --reference runs/cnn --out runs/v2

    python -m train.summary --check-folds data/processed/dataset.csv   # те же ли фолды и метки
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np

from train.compare import _auc_diff_ci, compare, model_oof, rules_oof
from train.tasks import TASKS

# Отпечаток фолдов и меток, на которых посчитаны baseline и первый прогон ResNet18.
# Если сборка датасета на другой машине дала другие фолды, сравнение теряет смысл.
FOLDS_FINGERPRINT = "83c5b71b422355511364fd6c93eb4484c4182361c805e0e398df9ae1d4257547"
MIN_GAIN = 0.05
LABEL = {t.key: t.label for t in TASKS}


def folds_fingerprint(csv_path: Path) -> str:
    """Хеш того, от чего зависит сопоставимость: снимок, область, метки и фолд."""
    keys = ("image_id", "region", "quality_class", "violation_type", "violations_known", "fold")
    with csv_path.open(encoding="utf-8") as fh:
        rows = sorted("|".join(r[k] for k in keys) for r in csv.DictReader(fh))
    return hashlib.sha256("\n".join(rows).encode("utf-8")).hexdigest()


def _config(run: Path) -> dict:
    return json.loads((run / "config.json").read_text(encoding="utf-8"))


def _stopping(run: Path) -> list[dict]:
    out = []
    for f in sorted(run.glob("fold*_preds.npz")):
        z = np.load(f, allow_pickle=False)
        out.append({"fold": int(f.name[4]), **json.loads(str(z["info"]))})
    return out


def pairwise(runs: dict[str, Path], n_boot: int = 2000) -> list[dict]:
    """Разница AUC между конфигурациями на одних и тех же снимках, парный бутстрэп по исследованиям."""
    preds = {name: {r["image_id"]: r for r in model_oof(run)} for name, run in runs.items()}
    out = []
    for a, b in itertools.combinations(runs, 2):
        for t in TASKS:
            ids = [
                i
                for i, r in preds[a].items()
                if r[f"p_{t.key}"] != "" and i in preds[b] and preds[b][i][f"p_{t.key}"] != ""
            ]
            if not ids:
                continue
            y = np.array([int(preds[a][i][f"y_{t.key}"]) for i in ids])
            if not 0 < y.sum() < len(y):
                continue
            pa = np.array([float(preds[a][i][f"p_{t.key}"]) for i in ids])
            pb = np.array([float(preds[b][i][f"p_{t.key}"]) for i in ids])
            groups = np.array([preds[a][i]["study_dir"] for i in ids])
            d, lo, hi = _auc_diff_ci(y, pb, pa, groups, n=n_boot)
            out.append({"a": a, "b": b, "task": t.key, "diff_b_minus_a": d, "ci95": [lo, hi]})
    return out


def decide(results: dict[str, dict]) -> dict:
    """Правило К2: кто прошёл критерий, где кто объективно сильнее, какой следующий шаг."""
    passed = [n for n, r in results.items() if r["stage6_criterion_met"]]
    mean_gain = {n: float(np.mean([g["gain"] for g in r["regions"].values()])) for n, r in results.items()}
    mean_auc = {n: float(np.mean([t["model"]["roc_auc"] for t in r["tasks"].values()])) for n, r in results.items()}
    rules_stronger = {
        n: [k for k, t in r["tasks"].items() if "auc_diff" in t and t["auc_diff"]["ci95"][1] < 0]
        for n, r in results.items()
    }
    model_stronger = {
        n: [k for k, t in r["tasks"].items() if "auc_diff" in t and t["auc_diff"]["ci95"][0] > 0]
        for n, r in results.items()
    }
    if passed:
        chosen = max(passed, key=mean_gain.get)
        why = f"прошла критерий этапа 6 (macro-F1 +{MIN_GAIN} хотя бы в одной области при специфичности ≥ 0.70)" + (
            ", из прошедших — больший средний прирост macro-F1" if len(passed) > 1 else ""
        )
    else:
        chosen = max(results, key=mean_auc.get)
        why = "критерий не прошла ни одна конфигурация; выбрана с наибольшим средним ROC AUC по семи задачам"
    optimistic = len(results) > 1
    step = {
        "next_step": "гибрид по К3 (этап 7, вложенная кросс-валидация)",
        "model_for_hybrid": chosen,
        "why": why,
        "selection_is_optimistic": optimistic,
        "rules_only_tasks": rules_stronger[chosen],
        "model_stronger_tasks": model_stronger[chosen],
        "stage6_passed_by": passed,
        "mean_gain_macro_f1": mean_gain,
        "mean_roc_auc": mean_auc,
        "rules_stronger": rules_stronger,
        "model_stronger": model_stronger,
        "improve_ai": "только резервной конфигурацией с гипотезой из разбора расхождений, объявленной до запуска; "
        "иначе — только с новыми данными (З7)",
    }
    return step


def _f(x: float) -> str:
    return f"{x:+.2f}".replace("-", "−")


def render(results: dict, reference: dict | None, pairs: list[dict], decision: dict, meta: dict) -> str:
    names = list(results)
    L = []  # noqa: N806
    L.append("# Этап 6, протокол v2: сводка\n")
    L.append(
        "Out-of-fold на тех же 5 фолдах по исследованиям, что у baseline. Число эпох выбрано по loss на "
        "внутреннем фолде (k+1) mod 5 внутри train-части"
        + (", затем модель обучена заново на всей train-части столько эпох" if meta.get("refit") else "")
        + ". Внешний фолд — только для предсказаний. Порог 0.5 объявлен заранее. "
        "ДИ — парный бутстрэп по исследованиям.\n"
    )
    L.append(f"Отпечаток фолдов и меток: `{meta['folds_fingerprint'][:16]}…` ({meta['folds_check']}).")
    precision = "смешанная (AMP, fp16 на GPU)" if meta.get("amp") else "fp32"
    L.append(f"Точность вычислений: {precision}, одинаковая для всех конфигураций.\n")

    L.append("## ROC AUC: модель против правил\n")
    L.append("| Задача | n / нарушений | Правила | " + " | ".join(names) + " |")
    L.append("|---|---:|---:|" + "---:|" * len(names))
    first = results[names[0]]["tasks"]
    for key, t0 in first.items():
        cells = []
        for n in names:
            t = results[n]["tasks"].get(key)
            if not t:
                cells.append("—")
                continue
            d = t.get("auc_diff")
            mark = ""
            if d and d["ci95"][0] > 0:
                mark = " ▲"
            elif d and d["ci95"][1] < 0:
                mark = " ▼"
            diff = f" ({_f(d['value'])} [{_f(d['ci95'][0])}; {_f(d['ci95'][1])}])" if d else ""
            cells.append(f"{t['model']['roc_auc']:.2f}{diff}{mark}")
        L.append(
            f"| {LABEL[key]} | {t0['n']} / {t0['n_pos']} | {t0['rules']['roc_auc']:.2f} | " + " | ".join(cells) + " |"
        )
    L.append("\nВ скобках — разница с правилами и 95% ДИ. ▲/▼ — ДИ целиком выше/ниже нуля.\n")

    L.append("## macro-F1 по областям и критерий этапа 6\n")
    L.append("| Область | Правила | " + " | ".join(names) + " |")
    L.append("|---|---:|" + "---:|" * len(names))
    for region in results[names[0]]["regions"]:
        r0 = results[names[0]]["regions"][region]
        cells = []
        for n in names:
            g = results[n]["regions"][region]
            spec = "" if g["model_specificity_ok"] else ", Sp < 0.70"
            cells.append(f"{g['macro_f1_model']:.3f} ({_f(g['gain'])}{spec}){' ✔' if g['criterion_met'] else ''}")
        L.append(f"| {region} | {r0['macro_f1_rules']:.3f} | " + " | ".join(cells) + " |")
    L.append("\n✔ — критерий выполнен: прирост ≥ 0.05 при специфичности всех классов ≥ 0.70.\n")

    L.append("## Чувствительность / специфичность / F1 при пороге 0.5\n")
    L.append("| Задача | Правила | " + " | ".join(names) + " |")
    L.append("|---|---:|" + "---:|" * len(names))
    for key, t0 in first.items():
        r = t0["rules"]
        cells = []
        for n in names:
            m = results[n]["tasks"][key]["model"]
            cells.append(f"{m['sensitivity']:.2f} / {m['specificity']:.2f} / {m['f1']:.2f}")
        L.append(
            f"| {LABEL[key]} | {r['sensitivity']:.2f} / {r['specificity']:.2f} / {r['f1']:.2f} | "
            + " | ".join(cells)
            + " |"
        )

    L.append("\n## Остановка по внутренней validation\n")
    L.append(
        "| Конфигурация | Эпох на всей train-части по фолдам | Ранняя остановка | AUC качества на внешнем фолде "
        "| То же у модели на трёх фолдах (диагностика) |"
    )
    L.append("|---|---|---|---|---|")
    for n in names:
        st = results[n]["stopping"]
        ep = ", ".join(str(s.get("refit_epochs", s.get("best_inner_epoch", -1) + 1)) for s in st)
        stopped = sum(bool(s.get("early_stopped")) for s in st)
        auc = ", ".join(f"{s['external_auc']:.2f}" for s in st if "external_auc" in s)
        inner = ", ".join(f"{s['external_auc_inner_model']:.2f}" for s in st if "external_auc_inner_model" in s)
        L.append(f"| {n} | {ep} | {stopped} из {len(st)} | {auc} | {inner or '—'} |")
    L.append(
        "\nДиагностика — модель, которая училась на трёх фолдах до повторного обучения на всей train-части. "
        "В решении не участвует, показывает, что дали дополнительные данные."
    )

    if pairs:
        L.append("\n## Конфигурации между собой (разница ROC AUC, 95% ДИ)\n")
        L.append(
            "| Задача | " + " | ".join(f"{p['b']} − {p['a']}" for p in pairs if p["task"] == pairs[0]["task"]) + " |"
        )
        combos = [(p["a"], p["b"]) for p in pairs if p["task"] == pairs[0]["task"]]
        L.append("|---|" + "---:|" * len(combos))
        for key in first:
            cells = []
            for a, b in combos:
                p = next((x for x in pairs if x["a"] == a and x["b"] == b and x["task"] == key), None)
                cells.append(f"{_f(p['diff_b_minus_a'])} [{_f(p['ci95'][0])}; {_f(p['ci95'][1])}]" if p else "—")
            L.append(f"| {LABEL[key]} | " + " | ".join(cells) + " |")

    if reference:
        L.append("\n## Для сравнения: ResNet18, протокол v1 (в решении не участвует)\n")
        for region, g in reference["regions"].items():
            L.append(
                f"- {region}: macro-F1 {g['macro_f1_model']:.3f} (правила {g['macro_f1_rules']:.3f}, {_f(g['gain'])})"
            )

    L.append("\n## Следующий шаг по правилу К2\n")
    d = decision
    passed = ", ".join(d["stage6_passed_by"]) or "ни одна"
    L.append(f"- Критерий этапа 6 прошли: **{passed}**.")
    L.append(f"- Следующий шаг: **{d['next_step']}** с моделью **{d['model_for_hybrid']}** — {d['why']}.")
    if d["selection_is_optimistic"]:
        L.append("- Модель выбрана из нескольких по тем же OOF, поэтому её цифры немного оптимистичны.")
    ro = ", ".join(LABEL[k] for k in d["rules_only_tasks"]) or "нет"
    ms = ", ".join(LABEL[k] for k in d["model_stronger_tasks"]) or "нет"
    L.append(f"- Задачи, где правила объективно сильнее (в гибриде остаются за правилами): {ro}.")
    L.append(f"- Задачи, где модель объективно сильнее: {ms}.")
    L.append(
        "- Во вложенной кросс-валидации К3 есть вариант «только правила»: если выберет его, "
        "правила остаются, модель в сервис не подключается."
    )
    L.append(f"- Улучшать AI: {d['improve_ai']}.")
    return "\n".join(L) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", help="папки прогонов train.train по протоколу v2")
    ap.add_argument("--features", help="измерения правил (baseline/featdump.py)")
    ap.add_argument("--reference", help="прогон v1 для справки (runs/cnn)")
    ap.add_argument("--data", default="data/processed", help="датасет, на котором шли прогоны")
    ap.add_argument("--out", help="куда положить summary.json и SUMMARY.md")
    ap.add_argument("--min-specificity", type=float, default=0.70)
    ap.add_argument("--check-folds", metavar="DATASET_CSV", help="только проверить отпечаток фолдов и меток")
    args = ap.parse_args()

    if args.check_folds:
        fp = folds_fingerprint(Path(args.check_folds))
        ok = fp == FOLDS_FINGERPRINT
        print(f"отпечаток фолдов и меток: {fp}")
        print("совпадает с baseline и первым прогоном" if ok else f"НЕ совпадает, ожидался {FOLDS_FINGERPRINT}")
        raise SystemExit(0 if ok else 1)
    if not (args.runs and args.features and args.out):
        ap.error("нужны --runs, --features и --out (или --check-folds)")

    runs = {Path(r).name: Path(r) for r in args.runs}
    configs = {n: _config(r) for n, r in runs.items()}
    shas = {c.get("dataset_sha256") for c in configs.values()}
    if len(shas) != 1:
        raise SystemExit(f"прогоны сделаны на разных датасетах: {shas}")
    not_v2 = [n for n, c in configs.items() if c.get("early_stopping") != "inner"]
    if not_v2:
        raise SystemExit(f"не протокол v2 (--early-stopping inner): {not_v2}")
    amps = {bool(c.get("amp", False)) for c in configs.values()}
    if len(amps) != 1:
        raise SystemExit("часть прогонов в fp32, часть в смешанной точности — сравнение не на равных")
    refits = {bool(c.get("refit", False)) for c in configs.values()}
    if len(refits) != 1:
        raise SystemExit("часть прогонов с --refit, часть без — сравнение не на равных")

    rules = rules_oof(Path(args.features), args.min_specificity)
    results = {}
    for n, run in runs.items():
        res = compare(run, Path(args.features), args.min_specificity, rules=rules)
        (run / "compare_baseline.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
        res["stopping"] = _stopping(run)
        res["config"] = {
            k: configs[n].get(k)
            for k in ("backbone", "init_backbone", "early_stopping", "patience", "epochs", "amp", "refit")
        }
        results[n] = res
    reference = None
    if args.reference and (Path(args.reference) / "oof_predictions.csv").exists():
        reference = compare(Path(args.reference), Path(args.features), args.min_specificity, rules=rules)
    pairs = pairwise(runs)
    decision = decide(results)
    fp = folds_fingerprint(Path(args.data) / "dataset.csv")
    meta = {
        "folds_fingerprint": fp,
        "folds_check": "совпадает с baseline" if fp == FOLDS_FINGERPRINT else "НЕ совпадает с baseline",
        "dataset_sha256": shas.pop(),
        "amp": amps.pop(),
        "refit": refits.pop(),
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    summary = {"meta": meta, "results": results, "reference_v1": reference, "pairwise": pairs, "decision": decision}
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    md = render(results, reference, pairs, decision, meta)
    (out / "SUMMARY.md").write_text(md, encoding="utf-8")
    print(md)
    print(f"-> {out / 'SUMMARY.md'}, {out / 'summary.json'}")


if __name__ == "__main__":
    main()
