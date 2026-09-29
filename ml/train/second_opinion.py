"""Этап 8: модель «второго мнения» для сервиса — справочно, в вердикт не входит.

Гибрид (этап 7) у правил не выиграл, поэтому модель в вердикт не идёт, а показывается
рядом с проверками правил — только для бедра, где она вровень с правилами. Модель —
конфигурация 4 без изменений (OVERVIEW.md, К2 от 27.09): замороженный xrv-densenet121
384×320 + логистическая регрессия (C = 0.1, `class_weight="balanced"`).

Для сервиса регрессия учится на всех 148 снимках бедра. Порог выбирается тем же правилом,
что в зонде (`fit_threshold`, специфичность ≥ 0.70), по OOF на пяти фолдах. Точность,
которую сервис показывает рядом с оценкой, — OOF ROC AUC зонда из `runs/probe`, а не
оценка на обучающих данных.

    python -m train.second_opinion --data data/processed \\
        --features runs/probe/features_xrv-densenet121_384x320.npy \\
        --out ../backend/app/processing/second_opinion_model/xrv_probe.json

`--check DATA_ROOT` — сквозная проверка: исходные DICOM через предобработку сервиса и ONNX
(`backend/app/processing/second_opinion.py`) против признаков, на которых училась
регрессия. Расхождение должно быть на уровне погрешности float32.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

from train.dataset import read_dataset
from train.tasks import TASKS, targets_for

ROOT = Path(__file__).resolve().parents[2]
REGION_FEMUR = "Проксимальный отдел бедра"
PROBE_CONFIG = "xrv-densenet121 384x320"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _r(values) -> list[float]:  # noqa: ANN001
    return [float(f"{v:.7g}") for v in np.asarray(values).ravel()]


def fit(samples: list, X: np.ndarray, C: float, min_spec: float, probe_metrics: dict) -> dict:  # noqa: N803
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    from baseline.calibrate import fit_threshold
    from train.probe import _fit_predict

    folds = np.array([s.fold for s in samples])
    tasks = {}
    for ti, t in enumerate(TASKS):
        if t.region != REGION_FEMUR:
            continue
        tm = [targets_for(s.region, s.quality_class, s.violations, s.violations_known) for s in samples]
        use = np.array([m[ti] > 0 for _, m in tm])
        y_all = np.array([tt[ti] for tt, _ in tm], dtype=int)
        idx = np.flatnonzero(use)
        oof = np.full(len(samples), np.nan)
        for f in sorted(set(folds[idx])):
            tr, te = idx[folds[idx] != f], idx[folds[idx] == f]
            oof[te] = _fit_predict(X[tr], y_all[tr], X[te], C)
        thr, _ = fit_threshold(oof[idx], y_all[idx].astype(bool), ">", min_spec)
        scaler = StandardScaler().fit(X[idx])
        clf = LogisticRegression(C=C, class_weight="balanced", max_iter=5000).fit(scaler.transform(X[idx]), y_all[idx])
        m = probe_metrics["results"][PROBE_CONFIG][t.key]
        tasks[t.key] = {
            "label": t.label.split(": ", 1)[-1],
            "region": t.region,
            "violation": t.violation or "",
            "n": int(len(idx)),
            "n_pos": int(y_all[idx].sum()),
            "oof_roc_auc": round(float(m["roc_auc"]), 4),
            "threshold": float(thr),
            "mean": _r(scaler.mean_),
            "scale": _r(scaler.scale_),
            "coef": _r(clf.coef_[0]),
            "intercept": float(clf.intercept_[0]),
        }
    return tasks


def check(samples: list, X: np.ndarray, probe: dict, data_root: Path, model_dir: Path) -> None:  # noqa: N803
    """Исходные DICOM -> код сервиса -> ONNX против признаков зонда."""
    sys.path.insert(0, str(ROOT / "backend"))
    from app.processing.rulebased import read_pixels
    from app.processing.second_opinion import SecondOpinion, score

    so = SecondOpinion(model_dir)
    if not so.load():
        raise SystemExit(f"модель не загрузилась: {so.error}")
    import csv

    with (Path(probe["_data"]) / "dataset.csv").open(encoding="utf-8") as fh:
        dicom = {r["image_id"]: r["dicom_path"] for r in csv.DictReader(fh)}
    base = data_root / "НД_для_обучения" / "Исследования"
    feat_diff, cos, prob_diff = [], [], []
    for k, s in enumerate(samples):
        arr, _ = read_pixels(base / dicom[s.image_id])
        f = so.features(arr)
        feat_diff.append(float(np.max(np.abs(f - X[k]))))
        cos.append(float(f @ X[k] / (np.linalg.norm(f) * np.linalg.norm(X[k]) + 1e-12)))
        if s.region == REGION_FEMUR:
            for t in probe["tasks"].values():
                prob_diff.append(abs(score(f, t) - score(X[k], t)))
    print(f"снимков: {len(samples)}")
    print(f"признаки: макс. |разница| {max(feat_diff):.2e}, мин. косинус {min(cos):.6f}")
    print(f"оценки бедра: макс. |разница| {max(prob_diff):.2e}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="data/processed")
    ap.add_argument("--features", default="runs/probe/features_xrv-densenet121_384x320.npy")
    ap.add_argument("--probe-metrics", default="runs/probe/probe_metrics.json")
    ap.add_argument("--out", default=str(ROOT / "backend/app/processing/second_opinion_model/xrv_probe.json"))
    ap.add_argument("--C", type=float, default=0.1, help="как в train/probe.py; задано заранее")
    ap.add_argument("--min-specificity", type=float, default=0.70)
    ap.add_argument("--check", type=Path, metavar="DATA_ROOT", help="сквозная проверка против DICOM из выгрузки")
    args = ap.parse_args()

    data = Path(args.data)
    samples = read_dataset(data / "dataset.csv", data)
    X = np.load(args.features)  # noqa: N806
    if len(X) != len(samples):
        raise SystemExit(f"признаков {len(X)}, снимков {len(samples)}: пересчитайте train.probe на этом датасете")
    out = Path(args.out)

    if args.check:
        probe = json.loads(out.read_text(encoding="utf-8"))
        probe["_data"] = str(data)
        check(samples, X, probe, args.check, out.parent)
        return

    import sklearn

    probe_metrics = json.loads(Path(args.probe_metrics).read_text(encoding="utf-8"))
    payload = {
        "model": "замороженный DenseNet121 (TorchXRayVision) + логистическая регрессия, конфигурация 4 (К2)",
        "backbone": {"weights": "densenet121-res224-all", "height": 384, "width": 320, "onnx": "xrv_densenet121.onnx"},
        "trained_on": {
            "images": int(sum(s.region == REGION_FEMUR for s in samples)),
            "dataset_sha256": _sha(data / "dataset.csv"),
            "features_sha256": _sha(Path(args.features)),
            "C": args.C,
            "min_specificity": args.min_specificity,
            "sklearn": sklearn.__version__,
        },
        "tasks": fit(samples, X, args.C, args.min_specificity, probe_metrics),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=1, allow_nan=False), encoding="utf-8")
    for k, t in payload["tasks"].items():
        print(
            f"{k}: n={t['n']}, нарушений {t['n_pos']}, порог {t['threshold']:.3f}, OOF ROC AUC {t['oof_roc_auc']:.3f}"
        )
    print(f"-> {out} ({out.stat().st_size / 1024:.0f} КБ)")


if __name__ == "__main__":
    main()
