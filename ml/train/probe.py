"""Линейный зонд: что предобученный бэкбон «видит» в наших задачах сам по себе.

Бэкбон заморожен, признаки снимаются один раз, поверх — логистическая регрессия на
каждую из семи задач. Это отвечает на вопрос, какой бэкбон брать в полноценное
обучение, за минуты на CPU и без риска переобучить всю сеть на 246 снимках.

Сравнимость с baseline на правилах — полная:

* фолды те же (из dataset.csv), пациент целиком в одном фолде;
* порог для каждого отложенного фолда выбирается вложенной кросс-валидацией внутри
  обучающих фолдов и тем же правилом, что у правил: центр плато balanced accuracy при
  специфичности не ниже 0.70 (`ml/baseline/calibrate.fit_threshold`);
* сила регуляризации задана заранее и не подбирается: подбор по OOF — та же утечка.

    python -m train.probe --data data/processed
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from dxa.labels import REGION_FEMUR, REGION_SPINE, VIOL_FEMUR_ROI, VIOL_FOREIGN, VIOL_POSITION, VIOL_SPINE_AXIS
from train import metrics as M
from train.dataset import DxaDataset, read_dataset
from train.model import build_backbone, input_spec
from train.tasks import TASKS, targets_for

logger = logging.getLogger("probe")

# Конфигурации объявлены заранее, чтобы не выбирать лучшую из десятка задним числом:
# (бэкбон, высота, ширина, предобучен ли).
DEFAULT_CONFIGS = [
    ("resnet18", 384, 320, True),  # текущий бэкбон обучения, ImageNet
    ("xrv-densenet121", 384, 320, True),  # рентгенограммы, наше разрешение
    ("xrv-densenet121", 224, 224, True),  # рентгенограммы, родное разрешение весов
    # Контроль: та же архитектура со случайными весами. Разница с предобученной — это
    # и есть вклад предобучения; скачивать для него ничего не нужно.
    ("xrv-densenet121", 384, 320, False),
]

# Названия задач так, как они называются в метриках baseline на правилах
RULES_NAMES = {
    "spine_quality": (REGION_SPINE, "качество (есть нарушение)"),
    "spine_position": (REGION_SPINE, VIOL_POSITION),
    "spine_axis": (REGION_SPINE, VIOL_SPINE_AXIS),
    "spine_foreign": (REGION_SPINE, VIOL_FOREIGN),
    "femur_quality": (REGION_FEMUR, "качество (есть нарушение)"),
    "femur_position": (REGION_FEMUR, VIOL_POSITION),
    "femur_roi": (REGION_FEMUR, VIOL_FEMUR_ROI),
}


@torch.no_grad()
def extract(  # noqa: ANN201
    samples,  # noqa: ANN001
    backbone: str,
    size: tuple[int, int],
    device: torch.device,
    pretrained: bool = True,
    batch: int = 16,
):
    torch.manual_seed(0)  # случайные веса контроля тоже должны воспроизводиться
    net, _ = build_backbone(backbone, pretrained=pretrained)
    net.eval().to(device)
    ds = DxaDataset(samples, size=size, train=False, norm=input_spec(backbone))
    feats = []
    for i in range(0, len(ds), batch):
        x = torch.stack([ds[j][0] for j in range(i, min(i + batch, len(ds)))]).to(device)
        feats.append(net(x).cpu().numpy())
    return np.concatenate(feats)


def _fit_predict(X_tr: np.ndarray, y_tr: np.ndarray, X_te: np.ndarray, C: float) -> np.ndarray:
    if len(np.unique(y_tr)) < 2:  # в обучающей части нет одного из классов
        return np.full(len(X_te), float(y_tr.mean() if len(y_tr) else 0.0))
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=C, class_weight="balanced", max_iter=5000),
    )
    clf.fit(X_tr, y_tr)
    return clf.predict_proba(X_te)[:, 1]


def _aligned(p: np.ndarray, thr: float) -> np.ndarray:
    """Сдвиг вероятностей так, чтобы порог фолда пришёлся на 0.5.

    Порог у каждого фолда свой; после сдвига все фолды оцениваются с одним порогом 0.5,
    а порядок внутри фолда не меняется. Так же устроены мягкие оценки у правил.
    """
    if np.isposinf(thr):
        return np.zeros_like(p)
    if np.isneginf(thr):
        return np.ones_like(p)
    eps = 1e-6
    logit = lambda v: np.log(np.clip(v, eps, 1 - eps) / (1 - np.clip(v, eps, 1 - eps)))  # noqa: E731
    return 1.0 / (1.0 + np.exp(-(logit(p) - logit(np.clip(thr, eps, 1 - eps)))))


def evaluate(X: np.ndarray, samples, C: float, min_specificity: float) -> dict:  # noqa: ANN001
    from baseline.calibrate import fit_threshold  # то же правило выбора порога, что у правил

    folds = np.array([s.fold for s in samples])
    out: dict = {}
    for ti, task in enumerate(TASKS):
        tm = [targets_for(s.region, s.quality_class, s.violations, s.violations_known) for s in samples]
        use = np.array([m[ti] > 0 for _, m in tm])
        y_all = np.array([t[ti] for t, _ in tm], dtype=int)
        idx = np.flatnonzero(use)
        scores = np.full(len(samples), np.nan)
        for f in sorted(set(folds[idx])):
            tr = idx[folds[idx] != f]
            te = idx[folds[idx] == f]
            # вложенная CV по обучающим фолдам — оценки для выбора порога
            inner = np.full(len(tr), np.nan)
            for g in sorted(set(folds[tr])):
                a, b = tr[folds[tr] != g], tr[folds[tr] == g]
                inner[np.isin(tr, b)] = _fit_predict(X[a], y_all[a], X[b], C)
            thr, _ = fit_threshold(inner, y_all[tr].astype(bool), ">", min_specificity)
            p = _fit_predict(X[tr], y_all[tr], X[te], C)
            scores[te] = _aligned(p, thr)
        y, s = y_all[idx], scores[idx]
        out[task.key] = M.summarize(y, s, thr=0.5) if 0 < y.sum() < len(y) else {"n": int(len(y))}
    for region, keys in (
        (REGION_SPINE, ["spine_position", "spine_axis", "spine_foreign"]),
        (REGION_FEMUR, ["femur_position", "femur_roi"]),
    ):
        out[f"macro_f1 {region}"] = float(np.mean([out[k].get("f1", 0.0) for k in keys]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="data/processed")
    ap.add_argument("--out", default="runs/probe")
    ap.add_argument("--rules", default="baseline/metrics.json", help="метрики baseline на правилах для сравнения")
    ap.add_argument("--C", type=float, default=0.1, help="сила L2-регуляризации, задана заранее")
    ap.add_argument("--min-specificity", type=float, default=0.70)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    data, out = Path(args.data), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    samples = read_dataset(data / "dataset.csv", data)
    device = torch.device(args.device)
    torch.manual_seed(0)

    results: dict = {}
    skipped: dict[str, str] = {}
    for backbone, h, w, pretrained in DEFAULT_CONFIGS:
        name = f"{backbone} {h}x{w}" + ("" if pretrained else " случайные веса")
        cache = out / f"features_{backbone}_{h}x{w}{'' if pretrained else '_random'}.npy"
        t0 = time.time()
        if cache.exists():
            X = np.load(cache)
        else:
            try:
                X = extract(samples, backbone, (h, w), device, pretrained=pretrained)
            except Exception as exc:  # noqa: BLE001 — нет сети до весов: пропускаем, а не падаем
                skipped[name] = f"{type(exc).__name__}: {exc}"[:200]
                logger.warning("%s пропущен: веса недоступны (%s)", name, skipped[name])
                continue
            np.save(cache, X)
        logger.info("%s: признаки %s за %.0f с", name, X.shape, time.time() - t0)
        results[name] = evaluate(X, samples, args.C, args.min_specificity)

    rules = json.loads(Path(args.rules).read_text(encoding="utf-8")) if Path(args.rules).exists() else {}
    payload = {"results": results, "skipped": skipped, "C": args.C, "min_specificity": args.min_specificity}
    (out / "probe_metrics.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    names = list(results)
    header = f"{'задача':34}" + "".join(f"{('правила'):>16}") + "".join(f"{n[:22]:>24}" for n in names)
    for metric in ("roc_auc", "balanced_accuracy", "f1"):
        print(f"\n=== {metric} (out-of-fold) ===")
        print(header)
        for task in TASKS:
            region, rname = RULES_NAMES[task.key]
            r = rules.get(region, {}).get(rname, {}).get(metric, float("nan"))
            row = f"{task.label[:34]:34}{r:16.3f}"
            row += "".join(f"{results[n][task.key].get(metric, float('nan')):24.3f}" for n in names)
            print(row)
    print("\n=== macro-F1 по видам нарушений ===")
    for region in (REGION_SPINE, REGION_FEMUR):
        r = rules.get(region, {}).get("macro_f1", float("nan"))
        print(f"{region[:34]:34}{r:16.3f}" + "".join(f"{results[n][f'macro_f1 {region}']:24.3f}" for n in names))
    for name, why in skipped.items():
        print(f"\nпропущено: {name} — {why}")
    print(f"\nметрики -> {out / 'probe_metrics.json'}")


if __name__ == "__main__":
    main()
