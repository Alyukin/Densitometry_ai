"""Предобучение бэкбона на открытом наборе Arak перед обучением на разметке эксперта.

Меток качества укладки в наборе нет, поэтому задача вспомогательная: по снимку узнать
область (позвоночник или бедро) и минеральную плотность кости (BMD, для бедра ещё и
шейки). Смысл — чтобы бэкбон привык к виду снимков DXA: кость, мягкие ткани, шум
сканирования. Дальше веса бэкбона подгружаются в обучение (`train.train --init-backbone`)
или в зонд (`train.probe --init-backbone`), а головы выбрасываются.

Кадры готовит `dxa.arak`: разметка сканера снята, яркость инвертирована, анизотропия
пикселя приведена к нашему сканеру. Масштаб у набора крупнее нашего, поэтому
аугментация в основном уменьшает кадр — как будто поле сканирования шире.

Проверка, что предобучение что-то дало, — только на нашей разметке, линейным зондом
или обучением на тех же фолдах. Метрики самого предобучения (точность области, R² по
BMD на отложенных пациентах Arak) показывают лишь, что сеть учится.

    python -m train.pretrain --data data/arak --out runs/arak --backbone xrv-densenet121
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset

from dxa.labels import REGION_FEMUR, REGION_SPINE
from train.dataset import to_tensor
from train.model import XRV_BACKBONES, build_backbone, input_spec

logger = logging.getLogger("pretrain")


def read_rows(csv_path: Path) -> list[dict]:
    with csv_path.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


class ArakDataset(Dataset):
    def __init__(
        self,
        rows: list[dict],
        root: Path,
        stats: dict,
        size: tuple[int, int],
        norm: str,
        train: bool,
        min_scale: float = 0.55,
    ) -> None:
        self.rows, self.root, self.stats = rows, root, stats
        self.size, self.norm, self.train, self.min_scale = size, norm, train, min_scale

    def __len__(self) -> int:
        return len(self.rows)

    def _image(self, path: Path) -> np.ndarray:
        h, w = self.size
        img = Image.open(path).convert("L")
        if not self.train:
            return np.asarray(img.resize((w, h), Image.BILINEAR), dtype=np.float32) / 255.0
        rng = np.random
        img = img.rotate(float(rng.uniform(-5, 5)), resample=Image.BILINEAR)
        if rng.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        # кадр вписывается в поле целиком, но меньше: масштаб набора крупнее нашего
        s = float(rng.uniform(self.min_scale, 1.0))
        small = img.resize((max(8, round(w * s)), max(8, round(h * s))), Image.BILINEAR)
        canvas = Image.new("L", (w, h), 0)
        canvas.paste(small, (int(rng.randint(0, w - small.size[0] + 1)), int(rng.randint(0, h - small.size[1] + 1))))
        a = np.asarray(canvas, dtype=np.float32) / 255.0
        a = a * float(rng.uniform(0.85, 1.15)) + float(rng.uniform(-0.05, 0.05))
        return np.clip(a, 0.0, 1.0)

    def __getitem__(self, i: int):  # noqa: ANN204
        r = self.rows[i]
        x = to_tensor(self._image(self.root / r["png_path"]), self.norm)
        region = r["region"]
        st = self.stats[region]
        bmd = (float(r["bmd"]) - st["bmd"][0]) / st["bmd"][1]
        neck = r.get("neck_bmd") or ""
        has_neck = region == REGION_FEMUR and neck != ""
        neck_z = (float(neck) - st["neck"][0]) / st["neck"][1] if has_neck else 0.0
        y = torch.tensor([1.0 if region == REGION_FEMUR else 0.0, bmd, neck_z], dtype=torch.float32)
        m = torch.tensor([1.0, 1.0, 1.0 if has_neck else 0.0], dtype=torch.float32)
        return x, y, m


class PretrainNet(nn.Module):
    """Бэкбон + три выхода: логит «бедро», BMD области и BMD шейки (z-оценки)."""

    def __init__(self, backbone: str, pretrained: bool = True) -> None:
        super().__init__()
        self.backbone, feat = build_backbone(backbone, pretrained)
        self.head = nn.Linear(feat, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))


def loss_fn(out: torch.Tensor, y: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    region = nn.functional.binary_cross_entropy_with_logits(out[:, 0], y[:, 0])
    reg = ((out[:, 1:] - y[:, 1:]) ** 2 * m[:, 1:]).sum() / m[:, 1:].sum().clamp(min=1.0)
    return region + reg


def fit_stats(rows: list[dict]) -> dict:
    stats = {}
    for region in (REGION_SPINE, REGION_FEMUR):
        b = np.array([float(r["bmd"]) for r in rows if r["region"] == region])
        n = np.array([float(r["neck_bmd"]) for r in rows if r["region"] == region and r.get("neck_bmd")])
        stats[region] = {
            "bmd": (float(b.mean()) if len(b) else 0.0, float(b.std()) or 1.0 if len(b) else 1.0),
            "neck": (float(n.mean()) if len(n) else 0.0, float(n.std()) or 1.0 if len(n) else 1.0),
        }
    return stats


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, rows: list[dict], device: torch.device) -> dict:
    model.eval()
    outs, ys = [], []
    for x, y, _ in loader:
        outs.append(model(x.to(device)).cpu())
        ys.append(y)
    o, y = torch.cat(outs).numpy(), torch.cat(ys).numpy()
    res = {"region_accuracy": float(((o[:, 0] > 0) == (y[:, 0] > 0.5)).mean())}
    regions = np.array([r["region"] for r in rows])
    for region, key in ((REGION_SPINE, "spine"), (REGION_FEMUR, "femur")):
        sel = regions == region
        if sel.sum() > 2:
            err = ((o[sel, 1] - y[sel, 1]) ** 2).sum()
            tot = ((y[sel, 1] - y[sel, 1].mean()) ** 2).sum()
            res[f"bmd_r2_{key}"] = float(1 - err / max(tot, 1e-9))
            res[f"bmd_corr_{key}"] = float(np.corrcoef(o[sel, 1], y[sel, 1])[0, 1])
    return res


def freeze_first_blocks(backbone: nn.Module, n: int) -> None:
    """Замораживает первые `n` блоков DenseNet (conv0 + denseblock1..n): быстрее на CPU."""
    if n <= 0:
        return
    features = getattr(backbone, "features", None)
    if features is None:
        return
    for name, module in features.named_children():
        for p in module.parameters():
            p.requires_grad = False
        if name == f"denseblock{n}":
            break


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/arak", help="папка с arak.csv и images/ (dxa.arak)")
    ap.add_argument("--out", default="runs/arak")
    ap.add_argument("--backbone", default="xrv-densenet121", help=" | ".join(XRV_BACKBONES) + " | resnet18 ...")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--height", type=int, default=384)
    ap.add_argument("--width", type=int, default=320)
    ap.add_argument("--val-fold", type=int, default=0, help="пациенты этого фолда — для контроля")
    ap.add_argument("--freeze-blocks", type=int, default=0, help="заморозить первые N блоков (для CPU)")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    data, out = Path(args.data), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = read_rows(data / "arak.csv")
    train_rows = [r for r in rows if int(r["fold"]) != args.val_fold]
    val_rows = [r for r in rows if int(r["fold"]) == args.val_fold]
    stats = fit_stats(train_rows)
    size, norm = (args.height, args.width), input_spec(args.backbone)
    device = torch.device(args.device)

    train_dl = DataLoader(
        ArakDataset(train_rows, data, stats, size, norm, train=True),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        drop_last=True,
    )
    val_dl = DataLoader(
        ArakDataset(val_rows, data, stats, size, norm, train=False),
        batch_size=args.batch_size,
        num_workers=args.workers,
    )
    model = PretrainNet(args.backbone).to(device)
    freeze_first_blocks(model.backbone, args.freeze_blocks)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(args.epochs, 1))
    logger.info("Arak: обучение %d снимков, контроль %d, %s %s", len(train_rows), len(val_rows), args.backbone, size)

    history, best = [], None
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, total, n = time.time(), 0.0, 0
        for x, y, m in train_dl:
            x, y, m = x.to(device), y.to(device), m.to(device)
            loss = loss_fn(model(x), y, m)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total, n = total + float(loss) * len(x), n + len(x)
        sched.step()
        val = evaluate(model, val_dl, val_rows, device)
        val_score = val.get("bmd_r2_spine", 0.0) + val.get("bmd_r2_femur", 0.0)
        history.append({"epoch": epoch, "train_loss": total / max(n, 1), **val, "min": (time.time() - t0) / 60})
        logger.info("эпоха %d: %s", epoch, json.dumps(history[-1], ensure_ascii=False))
        if best is None or val_score > best[0]:
            best = (val_score, epoch)
            torch.save(
                {
                    "backbone": args.backbone,
                    "state_dict": model.backbone.state_dict(),
                    "size": list(size),
                    "epoch": epoch,
                    "val": val,
                    "source": "Arak Bone Densitometry, CC BY 4.0",
                },
                out / "backbone.pt",
            )
    summary = {"config": vars(args), "best_epoch": best[1] if best else None, "history": history}
    (out / "pretrain_metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"веса бэкбона -> {out / 'backbone.pt'} (эпоха {summary['best_epoch']})")


if __name__ == "__main__":
    main()
