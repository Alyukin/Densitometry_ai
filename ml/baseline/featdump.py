"""Прогон измерений по всей размеченной выгрузке -> features.csv для анализа и калибровки."""

from __future__ import annotations

import argparse
import csv
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend" / "app" / "processing"))

from dxaqc.femur import measure_femur  # noqa: E402
from dxaqc.spine import measure_spine  # noqa: E402

REGION_SPINE = "Поясничный отдел позвоночника"


def run(data_dir: Path, out: Path) -> None:
    rows = [r for r in csv.DictReader(open(data_dir / "dataset.csv", encoding="utf-8")) if r["quality_class"] != ""]
    recs = []
    for r in rows:
        a = np.asarray(Image.open(data_dir / r["png_path"]).convert("L"))
        t0 = time.perf_counter()
        if r["region"] == REGION_SPINE:
            m = measure_spine(a)
        else:
            m = measure_femur(a, side=r["side"])
        elapsed = time.perf_counter() - t0
        d = {k: v for k, v in asdict(m).items() if k != "overlay"}
        d.update(
            measure_sec=round(elapsed, 5),
            image_id=r["image_id"],
            study_dir=r["study_dir"],
            region=r["region"],
            side=r["side"],
            fold=r["fold"],
            quality_class=r["quality_class"],
            violation_type=r["violation_type"],
            violations_known=r["violations_known"],
        )
        recs.append(d)
    keys = sorted({k for d in recs for k in d})
    front = ("image_id", "study_dir", "region", "side", "fold", "quality_class", "violation_type")
    head = [k for k in front if k in keys]
    keys = head + [k for k in keys if k not in head]
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for d in recs:
            w.writerow({k: d.get(k, "") for k in keys})
    n_fail = sum(1 for d in recs if not d.get("ok"))
    print(f"измерено {len(recs)} снимков, сбоев {n_fail} -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default=str(ROOT / "ml" / "data" / "processed"))
    ap.add_argument("--out", default="/tmp/features.csv")
    args = ap.parse_args()
    run(Path(args.data), Path(args.out))
