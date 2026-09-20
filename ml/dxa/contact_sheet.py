"""Контактные листы для визуальной проверки автоопределения области и стороны.

    python -m dxa.contact_sheet --data data/processed --out qa/

Эксперту достаточно пролистать 3 листа (позвоночник, правое бедро, левое бедро) и
отметить кадры, попавшие не в свою группу: это единственное место, где в разметку
может попасть систематическая ошибка.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from PIL import Image, ImageDraw

from dxa.labels import REGION_FEMUR, REGION_SPINE

TILE = (160, 200)
# Подписи латиницей: во встроенном шрифте PIL нет кириллицы.


def sheet(rows: list[dict], root: Path, out: Path, title: str, cols: int = 10) -> None:
    if not rows:
        return
    n = len(rows)
    r = (n + cols - 1) // cols
    W, H = cols * (TILE[0] + 6) + 6, r * (TILE[1] + 26) + 30
    canvas = Image.new("RGB", (W, H), (16, 16, 20))
    d = ImageDraw.Draw(canvas)
    d.text((8, 8), f"{title} - {n} images", fill=(240, 240, 240))
    for i, row in enumerate(rows):
        img = Image.open(root / row["png_path"]).convert("L").resize(TILE)
        x = 6 + (i % cols) * (TILE[0] + 6)
        y = 30 + (i // cols) * (TILE[1] + 26)
        canvas.paste(img.convert("RGB"), (x, y))
        cls = row["quality_class"]
        color = (120, 220, 140) if cls == "0" else (240, 140, 110) if cls == "1" else (150, 150, 150)
        d.text((x, y + TILE[1] + 2), f"{row['image_id'][:18]}", fill=(200, 200, 200))
        d.text(
            (x, y + TILE[1] + 12),
            f"class={cls or '-'} conf={row['side_confidence']}",
            fill=color,
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="data/processed")
    ap.add_argument("--out", default="qa")
    args = ap.parse_args()
    data = Path(args.data)
    rows = list(csv.DictReader((data / "dataset.csv").open(encoding="utf-8")))
    groups = {
        "spine": ("SPINE (lumbar)", [r for r in rows if r["region"] == REGION_SPINE]),
        "femur_right": (
            "FEMUR detected as RIGHT",
            [r for r in rows if r["region"] == REGION_FEMUR and r["side"] == "right"],
        ),
        "femur_left": (
            "FEMUR detected as LEFT",
            [r for r in rows if r["region"] == REGION_FEMUR and r["side"] == "left"],
        ),
        "low_confidence": (
            "LOW side confidence",
            [r for r in rows if r["side"] and float(r["side_confidence"]) < 0.3],
        ),
    }
    for name, (title, rs) in groups.items():
        sheet(rs, data, Path(args.out) / f"{name}.png", title)
        print(f"{name}: {len(rs)}")


if __name__ == "__main__":  # pragma: no cover
    main()
