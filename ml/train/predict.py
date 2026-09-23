"""Инференс: папка с DICOM -> CSV/XLSX в формате выходной таблицы из ТЗ.

    python -m train.predict --input /path/studies --models runs/cnn --out result.csv

Колонки строго по ТЗ + разрешённая заказчиком `quality_prob`:
    path_to_study, study_uid, image_uid, anatomical_region, quality_class,
    violation_type, processing_status, time_of_processing, quality_prob

Ансамбль: усредняются вероятности всех найденных fold*.pt.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from dxa import inventory
from dxa.labels import REGION_FEMUR, REGION_SPINE
from train.dataset import to_tensor
from train.model import DxaQualityNet, input_spec
from train.tasks import TASK_INDEX, TASKS

logger = logging.getLogger("predict")

DEFAULT_THRESHOLDS = {t.key: 0.5 for t in TASKS}
VIOLATION_TASKS = {
    REGION_SPINE: [t for t in TASKS if t.region == REGION_SPINE and t.violation],
    REGION_FEMUR: [t for t in TASKS if t.region == REGION_FEMUR and t.violation],
}
QUALITY_KEY = {REGION_SPINE: "spine_quality", REGION_FEMUR: "femur_quality"}


def load_models(models_dir: Path, device: torch.device) -> tuple[list[DxaQualityNet], dict]:
    ckpts = sorted(models_dir.glob("fold*.pt"))
    if not ckpts:
        raise SystemExit(f"в {models_dir} нет файлов fold*.pt")
    models, cfg = [], {}
    for c in ckpts:
        blob = torch.load(c, map_location=device, weights_only=False)
        cfg = blob.get("config", {})
        m = DxaQualityNet(blob.get("backbone", "resnet18"), pretrained=False).to(device)
        m.load_state_dict(blob["model"])
        m.eval()
        models.append(m)
    logger.info("загружено моделей: %d", len(models))
    return models, cfg


def preprocess(dicom_path: Path, size: tuple[int, int], norm: str = "imagenet") -> torch.Tensor:
    """DICOM -> вход сети. Нормировка та же, что при обучении (`train.dataset.to_tensor`)."""
    import pydicom

    ds = pydicom.dcmread(dicom_path, force=True)
    arr = ds.pixel_array
    if arr.dtype != np.uint8:
        a = arr.astype(np.float32)
        lo, hi = np.percentile(a, (0.5, 99.5))
        arr = (np.clip((a - lo) / max(hi - lo, 1e-6), 0, 1) * 255).astype(np.uint8)
    img = Image.fromarray(arr).convert("L").resize((size[1], size[0]), Image.BILINEAR)
    return to_tensor(np.asarray(img, dtype=np.float32) / 255.0, norm)


def predict_folder(
    input_dir: Path,
    models: list[DxaQualityNet],
    device: torch.device,
    size: tuple[int, int],
    thresholds: dict[str, float],
    keep_duplicates: bool = False,
    norm: str = "imagenet",
) -> list[dict]:
    records = inventory.scan(input_dir)
    rows: list[dict] = []
    for rec in records:
        if rec.is_duplicate and not keep_duplicates:
            continue
        t0 = time.perf_counter()
        status, probs = "success", None
        try:
            x = preprocess(input_dir / rec.path, size, norm).unsqueeze(0).to(device)
            with torch.no_grad():
                p = np.mean([torch.sigmoid(m(x)).cpu().numpy()[0] for m in models], axis=0)
            probs = p
        except Exception:  # noqa: BLE001
            logger.exception("ошибка на %s", rec.path)
            status = "error"
        elapsed = round(time.perf_counter() - t0, 3)

        region = rec.region
        row = {
            "path_to_study": str(Path(rec.path).parent),
            "study_uid": rec.study_uid,
            "image_uid": rec.sop_uid,
            "anatomical_region": region,
            "quality_class": "",
            "violation_type": "",
            # по п. 2.5 ТЗ — ровно два значения, как в выгрузке сервиса
            "processing_status": "Success" if status == "success" else "Failure",
            "time_of_processing": elapsed,
            "quality_prob": "",
        }
        if probs is not None:
            qkey = QUALITY_KEY[region]
            qprob = float(probs[TASK_INDEX[qkey]])
            viols = [
                t.violation for t in VIOLATION_TASKS[region] if probs[TASK_INDEX[t.key]] >= thresholds.get(t.key, 0.5)
            ]
            quality = int(qprob >= thresholds.get(qkey, 0.5) or bool(viols))
            row.update(
                quality_class=quality,
                violation_type=";".join(dict.fromkeys(viols)) if quality else "",
                quality_prob=round(qprob, 4),
            )
        rows.append(row)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, help="папка с DICOM (одно или несколько исследований)")
    ap.add_argument("--models", required=True, help="папка с fold*.pt")
    ap.add_argument("--out", default="result.csv")
    ap.add_argument("--thresholds", help="json с порогами по задачам (по умолчанию 0.5)")
    ap.add_argument("--height", type=int, default=0, help="0 = взять из config обучения")
    ap.add_argument("--width", type=int, default=0)
    ap.add_argument("--xlsx", action="store_true", help="сохранить ещё и .xlsx")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    device = torch.device(args.device)
    models, cfg = load_models(Path(args.models), device)
    size = (args.height or cfg.get("height", 384), args.width or cfg.get("width", 320))
    thresholds = dict(DEFAULT_THRESHOLDS)
    if args.thresholds:
        thresholds.update(json.loads(Path(args.thresholds).read_text(encoding="utf-8")))

    norm = input_spec(cfg.get("backbone", "resnet18"))  # нормировка — как при обучении
    rows = predict_folder(Path(args.input), models, device, size, thresholds, norm=norm)
    fields = [
        "path_to_study",
        "study_uid",
        "image_uid",
        "anatomical_region",
        "quality_class",
        "violation_type",
        "processing_status",
        "time_of_processing",
        "quality_prob",
    ]
    out = Path(args.out)
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    if args.xlsx:
        from openpyxl import Workbook

        wb = Workbook()
        ws = wb.active
        ws.title = "results"
        ws.append(fields)
        for r in rows:
            ws.append([r[f] for f in fields])
        wb.save(out.with_suffix(".xlsx"))
    print(f"строк: {len(rows)} -> {out}")


if __name__ == "__main__":  # pragma: no cover
    main()
