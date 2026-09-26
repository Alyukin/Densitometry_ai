"""Обучение с кросс-валидацией по исследованиям (group k-fold) и OOF-метриками.

Пример:
    python -m train.train --data data/processed --out runs/cnn --backbone resnet18 --epochs 40

Выбор эпохи (`--early-stopping`):
    none  — протокол v1: учим все `--epochs` эпох, в OOF идёт последняя эпоха.
    inner — протокол v2: эпоха выбирается по внутренней validation внутри train-части.
            Для внешнего фолда k внутренний — фолд (k+1) mod 5 (те же фолды по
            исследованиям), сеть учится на трёх оставшихся. Критерий — loss на внутреннем
            фолде, остановка после `--patience` эпох без улучшения. pos_weight считается
            только по снимкам, на которых сеть учится.
            С `--refit` (основной вариант этапа 6) найденное число эпох (лучшая
            внутренняя + 1) переносится на всю train-часть: модель учится заново на всех
            четырёх фолдах столько эпох, с тем же расписанием lr, и предсказывает внешний
            фолд. Без `--refit` итоговая модель — веса лучшей внутренней эпохи.
            Внешний фолд нужен только для предсказаний, ни для какого выбора.
            Порог 0.5 объявлен заранее в обоих протоколах.

Результат в `--out`:
    fold{k}.pt          — веса модели фолда (v1: последняя эпоха, v2: лучшая внутренняя
                          или, с --refit, обученная заново на всей train-части)
    fold{k}_preds.npz   — предсказания фолда (для --resume: готовый фолд не пересчитывается)
    fold{k}_search.pt   — v2 с --refit: итог поиска эпохи (для --resume)
    oof_predictions.csv — предсказания на отложенных фолдах для всех снимков
    oof_predictions_best_epoch.csv — только v1: то же с эпохи, лучшей на отложенном фолде
                          (только чтобы видеть, насколько такой выбор завышает метрики)
    oof_predictions_inner_model.csv — только v2 с --refit: предсказания модели, учившейся
                          на трёх фолдах, до повторного обучения на всей train-части
                          (диагностика, в решении не участвует)
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
from train.model import XRV_BACKBONES, DxaQualityNet, check_finite, input_spec, masked_bce, setup_amp
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


def amp_dtype(device: torch.device) -> torch.dtype:
    """fp16 на GPU; на CPU autocast умеет только bfloat16 — это режим для проверки кода."""
    return torch.float16 if device.type == "cuda" else torch.bfloat16


def run_epoch(  # noqa: ANN201
    model,
    loader,
    device,
    pw,
    optimizer=None,
    scaler=None,
    amp: bool = False,  # noqa: ANN001
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, list[int]]:
    """Одна эпоха. `amp` — смешанная точность: прямой проход в fp16, веса и лосс в fp32.

    Нужна, когда активации не помещаются в память GPU (DenseNet121 на 384×320 при
    batch 16 требует ~5 ГБ только под активации). Без `amp` всё считается в fp32.
    """
    train = optimizer is not None
    model.train(train)
    total, n = 0.0, 0
    P, T, Mk, IDX = [], [], [], []
    with torch.set_grad_enabled(train):
        for x, t, m, idx in loader:
            x, t, m = x.to(device), t.to(device), m.to(device)
            with torch.autocast(device.type, dtype=amp_dtype(device), enabled=amp):
                logits = model(x)
            logits = logits.float()
            loss = masked_bce(logits, t, m, pw)
            check_finite(loss)
            if train:
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None and scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
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


def write_oof(path: Path, samples: list, P: np.ndarray, T: np.ndarray, Mk: np.ndarray) -> None:  # noqa: N803
    cols = (
        ["image_id", "study_dir", "region", "side", "fold"]
        + [f"p_{t.key}" for t in TASKS]
        + [f"y_{t.key}" for t in TASKS]
    )
    with path.open("w", encoding="utf-8") as fh:
        fh.write(",".join(cols) + "\n")
        for k, s in enumerate(samples):
            vals = [f"{P[k, i]:.4f}" if Mk[k, i] > 0 and not np.isnan(P[k, i]) else "" for i in range(len(TASKS))]
            ys = [f"{int(T[k, i])}" if Mk[k, i] > 0 else "" for i in range(len(TASKS))]
            fh.write(",".join([s.image_id, s.study_dir, s.region, s.side, str(s.fold), *vals, *ys]) + "\n")


def _rng_state() -> dict:
    return {"py": random.getstate(), "np": np.random.get_state(), "torch": torch.get_rng_state()}


def _set_rng_state(st: dict) -> None:
    random.setstate(st["py"])
    np.random.set_state(st["np"])
    torch.set_rng_state(st["torch"])


def save_fold_preds(path: Path, last: tuple, best: tuple, info: dict) -> None:
    """Предсказания готового фолда: по ним --resume собирает OOF без переобучения."""
    np.savez(
        path,
        ids_last=np.array([s.image_id for s in last[3]]),
        p_last=last[0],
        t_last=last[1],
        m_last=last[2],
        ids_best=np.array([s.image_id for s in best[3]]),
        p_best=best[0],
        t_best=best[1],
        m_best=best[2],
        info=json.dumps(info, ensure_ascii=False),
    )


def load_fold_preds(path: Path, by_id: dict) -> tuple[tuple, tuple, dict]:
    z = np.load(path, allow_pickle=False)
    last = (z["p_last"], z["t_last"], z["m_last"], [by_id[i] for i in z["ids_last"]])
    best = (z["p_best"], z["t_best"], z["m_best"], [by_id[i] for i in z["ids_best"]])
    return last, best, json.loads(str(z["info"]))


def eval_loader(part: list, size: tuple[int, int], norm: str, args: argparse.Namespace) -> DataLoader:
    ds = DxaDataset(part, size=size, train=False, norm=norm)
    return DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)


def inner_split(train_part: list, fold: int, folds: int) -> tuple[list, list, int]:
    """Внутренняя validation для внешнего фолда `fold`: соседний фолд (fold+1) mod folds.

    Фолды уже разбиты по исследованиям и стратифицированы, поэтому снимки одного
    исследования не попадают и в обучение, и во внутреннюю validation. Внешний фолд сюда
    не передаётся вовсе.
    """
    inner = (fold + 1) % folds
    fit = [s for s in train_part if s.fold != inner]
    val = [s for s in train_part if s.fold == inner]
    return fit, val, inner


class EarlyStopping:
    """Остановка по loss на внутренней validation: меньше — лучше, улучшение строгое."""

    def __init__(self, patience: int) -> None:
        self.patience = patience
        self.best = float("inf")
        self.best_epoch = -1
        self.bad = 0

    def step(self, loss: float, epoch: int) -> bool:
        """True, если эпоха стала лучшей."""
        if loss < self.best:
            self.best, self.best_epoch, self.bad = loss, epoch, 0
            return True
        self.bad += 1
        return False

    @property
    def stop(self) -> bool:
        return self.bad >= self.patience

    def state(self) -> dict:
        return {"best": self.best, "best_epoch": self.best_epoch, "bad": self.bad}

    def load(self, st: dict) -> None:
        self.best, self.best_epoch, self.bad = st["best"], st["best_epoch"], st["bad"]


# Параметры, от которых зависит результат: при --resume они должны совпасть с прошлым запуском.
RESULT_KEYS = (
    "backbone", "no_pretrained", "init_backbone", "epochs", "batch_size", "lr", "weight_decay",
    "height", "width", "folds", "seed", "hflip_femur", "dataset_sha256", "early_stopping", "patience",
    "amp", "refit",
)  # fmt: skip
# Значения ключей, которых не было в config.json старых запусков.
KEY_DEFAULTS = {"early_stopping": "none", "patience": 8, "amp": False, "refit": False}


def refit_fold(args, fold, tr, va, device, size, norm, n_ep, ckpt_path):  # noqa: ANN001, ANN201
    """v2 с --refit: модель заново на всей train-части ровно `n_ep` эпох, затем внешний фолд.

    Всё как при поиске эпохи — та же инициализация, оптимизатор и косинусное расписание
    на `--epochs`, так что lr проходит тот же путь, что и за найденные эпохи. Внутренней
    validation здесь нет: число эпох уже выбрано. pos_weight — по всей train-части.
    """
    set_seed(args.seed + fold)
    pw = pos_weights(tr, device)
    dl_tr = DataLoader(
        DxaDataset(tr, size=size, train=True, hflip_femur=args.hflip_femur, norm=norm),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        drop_last=False,
    )
    model = DxaQualityNet(args.backbone, pretrained=not args.no_pretrained, init_backbone=args.init_backbone).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    start = 0
    ck = torch.load(ckpt_path, map_location=device, weights_only=False) if args.resume and ckpt_path.exists() else None
    if ck is not None and ck.get("phase") == "refit":
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        scaler.load_state_dict(ck["scaler"])
        _set_rng_state(ck["rng"])
        start = ck["epoch"] + 1
        logger.info("fold %d: продолжение обучения на всей train-части с эпохи %d", fold, start)
    for epoch in range(start, n_ep):
        tr_loss, *_ = run_epoch(model, dl_tr, device, pw, opt, scaler, args.amp)
        sched.step()
        logger.info("fold %d refit epoch %2d/%d train %.4f", fold, epoch, n_ep, tr_loss)
        ck = {
            "model": model.state_dict(),
            "opt": opt.state_dict(),
            "sched": sched.state_dict(),
            "scaler": scaler.state_dict(),
            "rng": _rng_state(),
            "epoch": epoch,
            "phase": "refit",
        }
        torch.save(ck, ckpt_path.with_suffix(".tmp"))
        ckpt_path.with_suffix(".tmp").replace(ckpt_path)
    _, P, T, Mk, idx = run_epoch(model, eval_loader(va, size, norm, args), device, pw, amp=args.amp)  # noqa: N806
    return model, (P, T, Mk, [va[i] for i in idx])


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
    ap.add_argument(
        "--early-stopping",
        choices=("none", "inner"),
        default="none",
        help="none — протокол v1 (последняя эпоха); inner — протокол v2 (внутренняя validation)",
    )
    ap.add_argument("--patience", type=int, default=8, help="v2: эпох без улучшения внутреннего loss до остановки")
    ap.add_argument(
        "--refit",
        action="store_true",
        help="v2: после поиска эпохи обучить модель заново на всей train-части найденное число эпох",
    )
    ap.add_argument(
        "--amp",
        action="store_true",
        help="смешанная точность (fp16 на GPU): для видеокарт, где активации не помещаются в память",
    )
    ap.add_argument(
        "--resume",
        action="store_true",
        help="продолжить прерванный запуск в той же --out: готовые фолды берутся из fold{k}_preds.npz, "
        "недоученный фолд — с последней сохранённой эпохи",
    )
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    set_seed(args.seed)
    data = Path(args.data)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    kernels = setup_amp(device, args.amp)
    logger.info("вычисления: %s", kernels)
    # Хеш таблицы датасета: по нему через полгода видно, на тех ли данных и фолдах
    # получена метрика. Веса и данные в git не кладутся, поэтому это единственная связь.
    config = {
        **vars(args),
        "dataset_sha256": hashlib.sha256((data / "dataset.csv").read_bytes()).hexdigest(),
        "compute": kernels,  # свойство видеокарты, а не параметр протокола: в RESULT_KEYS не входит
    }
    if args.resume and (out / "config.json").exists():
        prev = json.loads((out / "config.json").read_text(encoding="utf-8"))
        diff = {
            k: (prev.get(k, KEY_DEFAULTS.get(k)), config[k])
            for k in RESULT_KEYS
            if prev.get(k, KEY_DEFAULTS.get(k)) != config[k]
        }
        if diff:
            raise SystemExit(f"--resume: параметры не совпадают с прошлым запуском в {out}: {diff}")
    (out / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    samples = read_dataset(data / "dataset.csv", data)
    logger.info(
        "снимков: %d (позвоночник %d, бедро %d), исследований: %d",
        len(samples),
        sum(s.region == REGION_SPINE for s in samples),
        sum(s.region == REGION_FEMUR for s in samples),
        len({s.study_dir for s in samples}),
    )
    es = args.early_stopping == "inner"
    refit = es and args.refit
    if args.refit and not es:
        raise SystemExit("--refit работает только с --early-stopping inner")
    # v1: pos_weight по всем снимкам (как в первом прогоне); v2 — по обучающей части фолда
    pw_all = pos_weights(samples, device)
    size = (args.height, args.width)

    oof_p = np.full((len(samples), len(TASKS)), np.nan)
    oof_p_best = np.full((len(samples), len(TASKS)), np.nan)
    oof_t = np.zeros((len(samples), len(TASKS)))
    oof_m = np.zeros((len(samples), len(TASKS)))

    by_id = {s.image_id: s for s in samples}
    pos = {s.image_id: k for k, s in enumerate(samples)}

    def put(last: tuple, best: tuple) -> None:
        for target, pred in ((oof_p, last), (oof_p_best, best)):
            P, T, Mk, order = pred  # noqa: N806
            for row, s in enumerate(order):
                k = pos[s.image_id]
                target[k] = P[row]
                oof_t[k] = T[row]
                oof_m[k] = Mk[row]

    for fold in range(args.folds):
        if args.only_fold >= 0 and fold != args.only_fold:
            continue
        tr = [s for s in samples if s.fold != fold]
        va = [s for s in samples if s.fold == fold]
        if not va:
            continue
        preds_path, ckpt_path = out / f"fold{fold}_preds.npz", out / f"fold{fold}_ckpt.pt"
        if args.resume and preds_path.exists() and (out / f"fold{fold}.pt").exists():
            last_pred, best_pred, info = load_fold_preds(preds_path, by_id)
            put(last_pred, best_pred)
            logger.info("fold %d: готов, взят из %s (%s)", fold, preds_path.name, info)
            continue
        search_path = out / f"fold{fold}_search.pt"
        norm = input_spec(args.backbone)
        t0 = time.time()
        search = None
        if refit and args.resume and search_path.exists():
            search = torch.load(search_path, map_location="cpu", weights_only=False)
            logger.info(
                "fold %d: поиск эпохи уже сделан — лучшая внутренняя эпоха %d", fold, search["best_inner_epoch"]
            )
        if search is None:
            # Своё зерно на фолд: результат фолда не зависит от того, считались ли
            # предыдущие фолды в этом же процессе или взяты из --resume.
            set_seed(args.seed + fold)
            if es:
                fit, mon, inner = inner_split(tr, fold, args.folds)
                pw = pos_weights(fit, device)
            else:
                fit, mon, inner = tr, va, None
                pw = pw_all
            dl_tr = DataLoader(
                DxaDataset(fit, size=size, train=True, hflip_femur=args.hflip_femur, norm=norm),
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.workers,
                drop_last=False,
            )

            dl_mon = eval_loader(mon, size, norm, args)  # по нему следим за эпохами: v1 — внешний фолд, v2 — внутренний
            dl_va = eval_loader(va, size, norm, args) if es else dl_mon
            if es:
                logger.info(
                    "fold %d: обучение %d снимков, внутренняя validation — фолд %d (%d снимков), внешний %d снимков",
                    fold,
                    len(fit),
                    inner,
                    len(mon),
                    len(va),
                )

            model = DxaQualityNet(
                args.backbone, pretrained=not args.no_pretrained, init_backbone=args.init_backbone
            ).to(device)
            opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
            # Эпоха никогда не выбирается по внешнему фолду: это подгонка под тот же фолд,
            # на котором считается метрика.
            # v1: best/best_pred — лучшая эпоха по AUC на внешнем фолде, только как мера
            # завышения; в OOF идёт последняя эпоха.
            # v2: stopper следит за loss на внутреннем фолде, best_state — веса лучшей эпохи.
            best, best_epoch, best_pred = -1.0, -1, None
            stopper, best_state, last_epoch = EarlyStopping(args.patience), None, -1
            start = 0
            ck = (
                torch.load(ckpt_path, map_location=device, weights_only=False)
                if args.resume and ckpt_path.exists()
                else None
            )
            if ck is not None and ck.get("phase", "search") == "search":
                model.load_state_dict(ck["model"])
                opt.load_state_dict(ck["opt"])
                sched.load_state_dict(ck["sched"])
                if "scaler" in ck:
                    scaler.load_state_dict(ck["scaler"])
                _set_rng_state(ck["rng"])
                start = last_epoch = ck["epoch"]
                start += 1
                if es:
                    stopper.load(ck["stopper"])
                    best_state = ck["best_state"]
                else:
                    best, best_epoch = ck["best"], ck["best_epoch"]
                    bp = ck["best_pred"]
                    best_pred = (bp[0], bp[1], bp[2], [by_id[i] for i in bp[3]])
                logger.info("fold %d: продолжение с эпохи %d", fold, start)
            for epoch in range(start, args.epochs):
                if es and stopper.stop:
                    break
                tr_loss, *_ = run_epoch(model, dl_tr, device, pw, opt, scaler, args.amp)
                mon_loss, P, T, Mk, idx = run_epoch(model, dl_mon, device, pw, amp=args.amp)  # noqa: N806
                sched.step()
                score = val_score(P, T, Mk)
                last_epoch = epoch
                logger.info(
                    "fold %d epoch %2d train %.4f %s %.4f auc %.3f",
                    fold,
                    epoch,
                    tr_loss,
                    "inner" if es else "val",
                    mon_loss,
                    score,
                )
                if es:
                    if stopper.step(mon_loss, epoch):
                        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                elif score > best:
                    best, best_epoch = score, epoch
                    best_pred = (P, T, Mk, [mon[i] for i in idx])
                # Точка продолжения после каждой эпохи: прерванный процесс теряет не больше эпохи.
                ck = {
                    "model": model.state_dict(),
                    "opt": opt.state_dict(),
                    "sched": sched.state_dict(),
                    "scaler": scaler.state_dict(),
                    "rng": _rng_state(),
                    "epoch": epoch,
                    "phase": "search",
                }
                if es:
                    ck.update(stopper=stopper.state(), best_state=best_state)
                else:
                    ck.update(best=best, best_epoch=best_epoch)
                    ck["best_pred"] = (*best_pred[:3], [s.image_id for s in best_pred[3]])
                torch.save(ck, ckpt_path.with_suffix(".tmp"))
                ckpt_path.with_suffix(".tmp").replace(ckpt_path)

            if es:
                # Внешний фолд — один раз, моделью лучшей внутренней эпохи.
                model.load_state_dict(best_state)
                _, P, T, Mk, idx = run_epoch(model, dl_va, device, pw, amp=args.amp)  # noqa: N806
                score = val_score(P, T, Mk)
                last_pred = best_pred = (P, T, Mk, [va[i] for i in idx])
                saved_epoch = stopper.best_epoch
                info = {
                    "inner_fold": inner,
                    "best_inner_epoch": stopper.best_epoch,
                    "inner_loss": round(stopper.best, 4),
                    "last_epoch": last_epoch,
                    "early_stopped": stopper.stop,
                    "external_auc": round(score, 4),
                }
                msg = (
                    f"внутренний фолд {inner}, лучшая эпоха {stopper.best_epoch} (loss {stopper.best:.4f}), "
                    f"последняя {last_epoch}{', ранняя остановка' if stopper.stop else ''}; "
                    f"внешний фолд AUC {score:.3f}{' (модель на трёх фолдах)' if refit else ''}"
                )
            else:
                if start >= args.epochs:  # checkpoint уже на последней эпохе, предсказаний нет
                    _, P, T, Mk, idx = run_epoch(model, dl_va, device, pw, amp=args.amp)  # noqa: N806
                    score = val_score(P, T, Mk)
                last_pred = (P, T, Mk, [va[i] for i in idx])
                saved_epoch = args.epochs - 1
                info = {"last_auc": round(score, 4), "best_epoch": best_epoch, "best_auc": round(best, 4)}
                msg = f"последняя эпоха AUC {score:.3f} (лучшая на фолде — эпоха {best_epoch}, AUC {best:.3f})"
            if refit:
                search = {
                    "best_inner_epoch": stopper.best_epoch,
                    "info": info,
                    "msg": msg,
                    "pred": (*best_pred[:3], [s.image_id for s in best_pred[3]]),
                }
                torch.save(search, search_path)
                ckpt_path.unlink(missing_ok=True)
        if refit:
            # Основной результат: заново на всей train-части, найденное число эпох.
            n_ep = search["best_inner_epoch"] + 1
            model, last_pred = refit_fold(args, fold, tr, va, device, size, norm, n_ep, ckpt_path)
            bp = search["pred"]
            best_pred = (bp[0], bp[1], bp[2], [by_id[i] for i in bp[3]])
            score = val_score(*last_pred[:3])
            saved_epoch = n_ep - 1
            info = {**search["info"], "refit_epochs": n_ep, "refit_images": len(tr)}
            info["external_auc_inner_model"] = info.pop("external_auc")
            info["external_auc"] = round(score, 4)
            msg = (
                f"{search['msg']}; заново на всей train-части ({len(tr)} снимков), эпох: {n_ep}, "
                f"внешний фолд AUC {score:.3f}"
            )
        torch.save(
            {
                "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "config": vars(args),
                "backbone": args.backbone,
                "epoch": saved_epoch,
            },
            out / f"fold{fold}.pt",
        )
        save_fold_preds(preds_path, last_pred, best_pred, info)
        ckpt_path.unlink(missing_ok=True)
        search_path.unlink(missing_ok=True)
        logger.info("fold %d: %s, %.1f мин", fold, msg, (time.time() - t0) / 60)
        put(last_pred, best_pred)

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

    write_oof(out / "oof_predictions.csv", samples, oof_p, oof_t, oof_m)
    if not es:
        write_oof(out / "oof_predictions_best_epoch.csv", samples, oof_p_best, oof_t, oof_m)
    elif refit:
        write_oof(out / "oof_predictions_inner_model.csv", samples, oof_p_best, oof_t, oof_m)

    print(
        json.dumps(
            {k: {kk: vv for kk, vv in v.items() if not kk.endswith("_ci95")} for k, v in result.items()},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":  # pragma: no cover
    main()
