"""Обучение с кросс-валидацией по исследованиям (group k-fold) и OOF-метриками.

Пример:
    python -m train.train --data data/processed --out runs/cnn --backbone resnet18 --epochs 40

Результат в `--out`:
    fold{k}.pt          — веса лучшей эпохи каждого фолда
    oof_predictions.csv — предсказания на отложенных фолдах для всех снимков
    metrics.json        — метрики по задачам с 95% ДИ (бутстрэп)
    config.json         — все параметры запуска и хеш датасета (для воспроизводимости)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from dxa.labels import REGION_FEMUR, REGION_SPINE
from train import metrics as M
from train.dataset import DxaDataset, read_dataset
from train.model import XRV_BACKBONES, DxaQualityNet, input_spec, masked_bce
from train.tasks import TASK_INDEX, TASKS

logger = logging.getLogger("train")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def pos_weights(samples, device) -> torch.Tensor:  # noqa: ANN001
    """pos_weight по каждой задаче — компенсация дисбаланса классов."""
    from train.tasks import targets_for

    pos = np.zeros(len(TASKS))
    neg = np.zeros(len(TASKS))
    for s in samples:
        t, m = targets_for(s.region, s.quality_class, s.violations)
        t, m = np.array(t), np.array(m)
        pos += (t == 1) * m
        neg += (t == 0) * m
    w = np.where(pos > 0, neg / np.maximum(pos, 1), 1.0)
    return torch.tensor(np.clip(w, 0.5, 20.0), dtype=torch.float32, device=device)


def run_epoch(model, loader, device, pw, optimizer=None) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, list[int]]:  # noqa: ANN001
    train = optimizer is not None
    model.train(train)
    total, n = 0.0, 0
    P, T, Mk, IDX = [], [], [], []
    with torch.set_grad_enabled(train):
        for x, t, m, idx in loader:
            x, t, m = x.to(device), t.to(device), m.to(device)
            logits = model(x)
            loss = masked_bce(logits, t, m, pw)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            total += float(loss.detach()) * len(x)
            n += len(x)
            P.append(torch.sigmoid(logits).detach().cpu().numpy())
            T.append(t.cpu().numpy())
            Mk.append(m.cpu().numpy())
            IDX.extend(int(i) for i in idx)
    return (
        total / max(n, 1),
        np.concatenate(P),
        np.concatenate(T),
        np.concatenate(Mk),
        IDX,
    )


def val_score(P: np.ndarray, T: np.ndarray, Mk: np.ndarray) -> float:
    """Средний ROC AUC по двум задачам качества — критерий выбора эпохи."""
    vals = []
    for key in ("spine_quality", "femur_quality"):
        i = TASK_INDEX[key]
        sel = Mk[:, i] > 0
        y, p = T[sel, i], P[sel, i]
        if sel.sum() and 0 < y.sum() < len(y):
            vals.append(M.roc_auc(y, p))
    return float(np.mean(vals)) if vals else 0.0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="data/processed", help="папка с dataset.csv и images/")
    ap.add_argument("--out", default="runs/cnn")
    ap.add_argument(
        "--backbone",
        default="resnet18",
        help="resnet18 | resnet34 | resnet50 | efficientnet_b0 (ImageNet) | "
        + " | ".join(XRV_BACKBONES)
        + " (веса TorchXRayVision, обучены на рентгенограммах)",
    )
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument(
        "--init-backbone",
        help="веса бэкбона после предобучения (runs/arak/backbone.pt из train.pretrain); архитектура та же",
    )
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--height", type=int, default=384)
    ap.add_argument("--width", type=int, default=320)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--only-fold", type=int, default=-1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument(
        "--hflip-femur",
        action="store_true",
        help="отражать снимки бедра (левое<->правое)",
    )
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    set_seed(args.seed)
    data = Path(args.data)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    # Хеш таблицы датасета: по нему через полгода видно, на тех ли данных и фолдах
    # получена метрика. Веса и данные в git не кладутся, поэтому это единственная связь.
    config = {**vars(args), "dataset_sha256": hashlib.sha256((data / "dataset.csv").read_bytes()).hexdigest()}
    (out / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    samples = read_dataset(data / "dataset.csv", data)
    logger.info(
        "снимков: %d (позвоночник %d, бедро %d), исследований: %d",
        len(samples),
        sum(s.region == REGION_SPINE for s in samples),
        sum(s.region == REGION_FEMUR for s in samples),
        len({s.study_dir for s in samples}),
    )
    device = torch.device(args.device)
    pw = pos_weights(samples, device)
    size = (args.height, args.width)

    oof_p = np.full((len(samples), len(TASKS)), np.nan)
    oof_t = np.zeros((len(samples), len(TASKS)))
    oof_m = np.zeros((len(samples), len(TASKS)))

    for fold in range(args.folds):
        if args.only_fold >= 0 and fold != args.only_fold:
            continue
        tr = [s for s in samples if s.fold != fold]
        va = [s for s in samples if s.fold == fold]
        if not va:
            continue
        norm = input_spec(args.backbone)
        ds_tr = DxaDataset(tr, size=size, train=True, hflip_femur=args.hflip_femur, norm=norm)
        ds_va = DxaDataset(va, size=size, train=False, norm=norm)
        dl_tr = DataLoader(
            ds_tr,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.workers,
            drop_last=False,
        )
        dl_va = DataLoader(ds_va, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

        model = DxaQualityNet(args.backbone, pretrained=not args.no_pretrained, init_backbone=args.init_backbone).to(
            device
        )
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
        best, best_state, best_epoch = -1.0, None, -1
        t0 = time.time()
        for epoch in range(args.epochs):
            tr_loss, *_ = run_epoch(model, dl_tr, device, pw, opt)
            va_loss, P, T, Mk, idx = run_epoch(model, dl_va, device, pw)
            sched.step()
            score = val_score(P, T, Mk)
            logger.info(
                "fold %d epoch %2d train %.4f val %.4f auc %.3f",
                fold,
                epoch,
                tr_loss,
                va_loss,
                score,
            )
            if score > best:
                best, best_epoch = score, epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                best_pred = (P, T, Mk, [va[i] for i in idx])
        if best_state is not None:
            torch.save(
                {"model": best_state, "config": vars(args), "backbone": args.backbone},
                out / f"fold{fold}.pt",
            )
        logger.info(
            "fold %d: лучшая эпоха %d, AUC %.3f, %.1f мин",
            fold,
            best_epoch,
            best,
            (time.time() - t0) / 60,
        )

        P, T, Mk, order = best_pred
        pos = {s.image_id: k for k, s in enumerate(samples)}
        for row, s in enumerate(order):
            k = pos[s.image_id]
            oof_p[k] = P[row]
            oof_t[k] = T[row]
            oof_m[k] = Mk[row]

    # --- метрики по OOF ---
    result = {}
    for t in TASKS:
        i = TASK_INDEX[t.key]
        sel = (oof_m[:, i] > 0) & ~np.isnan(oof_p[:, i])
        y, p = oof_t[sel, i], oof_p[sel, i]
        if sel.sum() == 0 or y.sum() == 0:
            result[t.key] = {"n": int(sel.sum()), "note": "нет положительных примеров"}
            continue
        # Порог объявлен заранее, а не подобран по этим же OOF-предсказаниям: подбор по
        # тем данным, на которых потом считаются BA и F1, завышает их — на 6 позитивах
        # заметно. pos_weight уравновешивает классы, поэтому 0.5 — естественная рабочая
        # точка. ROC AUC и PR AUC от порога не зависят вовсе.
        thr = 0.5
        result[t.key] = {"label": t.label, **M.summarize(y, p, thr, seed=args.seed)}
    (out / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    with (out / "oof_predictions.csv").open("w", encoding="utf-8") as fh:
        cols = (
            ["image_id", "study_dir", "region", "side", "fold"]
            + [f"p_{t.key}" for t in TASKS]
            + [f"y_{t.key}" for t in TASKS]
        )
        fh.write(",".join(cols) + "\n")
        for k, s in enumerate(samples):
            vals = [
                f"{oof_p[k, i]:.4f}" if oof_m[k, i] > 0 and not np.isnan(oof_p[k, i]) else "" for i in range(len(TASKS))
            ]
            ys = [f"{int(oof_t[k, i])}" if oof_m[k, i] > 0 else "" for i in range(len(TASKS))]
            fh.write(",".join([s.image_id, s.study_dir, s.region, s.side, str(s.fold), *vals, *ys]) + "\n")

    print(
        json.dumps(
            {k: {kk: vv for kk, vv in v.items() if not kk.endswith("_ci95")} for k, v in result.items()},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":  # pragma: no cover
    main()
