"""Открытый набор Arak Bone Densitometry (CC BY 4.0) -> кадры, похожие на нашу выгрузку.

Зачем: предобучить бэкбон на снимках DXA перед обучением на 246 снимках эксперта.
Меток качества укладки в наборе нет, есть минеральная плотность (BMD) и T-критерий, поэтому
он годится только для предобучения, а не для оценки качества.

Что делается и почему:

* **Берутся только файлы с 5-значным номером пациента** (10000–13642): только их можно
  связать с таблицей `TableFinal-OSTEO.xlsx`. Файлы с 10–11-значными именами пропускаются:
  разметки к ним нет, а имена почти всегда проходят контрольную сумму иранского
  национального кода, то есть это, по всей видимости, персональные данные. Исходные имена
  в результат не попадают вообще: снимок получает обезличенный id.
* **Область — по номеру снимка и размеру кадра.** Суффикс `-2` — поясничный отдел, `-1` —
  бедро; кадр 376×1104 — боковая проекция позвоночника (VFA), она пропускается. Прочие
  суффиксы (`-3`, `-4`, без суффикса) не проверены глазами и тоже пропускаются.
* **Снимается разметка сканера** — рамки ROI, ось шейки, подписи L1–L4, контур кости.
  Прямые линии находятся морфологией, подписи — как сплошные чёрные квадраты, всё это
  закрашивается `cv2.inpaint`. Короткие пунктиры частично остаются.
* **Яркость инвертируется**: у этого сканера кость тёмная на белом, у нашего GE Lunar —
  светлая на тёмном.
* **Кадр растягивается по горизонтали в 1.75 раза**: пиксель нашего сканера 1.05×0.60 мм,
  и в выгрузке анатомия шире, чем в жизни. У набора Arak пиксель квадратный.
* **Фолды — по пациентам**, чтобы снимки одного человека не попали в обе части.

    python -m dxa.arak --data-root /путь/к/Densitometry_data --out data/arak
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import re
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from dxa.labels import REGION_FEMUR, REGION_SPINE

logger = logging.getLogger(__name__)

TABLE_NAME = "TableFinal-OSTEO.xlsx"
IMAGES_DIR = "imge"  # так папка названа в наборе
LATERAL_SIZE = (376, 1104)  # ширина×высота боковой проекции (VFA)
STRETCH_X = 1.05 / 0.60  # анизотропия пикселя GE Lunar Prodigy

SUFFIX_REGION = {"1": REGION_FEMUR, "2": REGION_SPINE}
NAME_RE = re.compile(r"^(?P<id>\d+)(?:-(?P<idx>\d+))?$")

FIELDS = [
    "image_id",
    "png_path",
    "patient",
    "region",
    "fold",
    "bmd",
    "tscore",
    "neck_bmd",
    "neck_tscore",
    "age_category",
]


# --- имена файлов ---------------------------------------------------------------


def parse_name(stem: str) -> tuple[int | None, str]:
    """(номер пациента или None, суффикс). None — файл нельзя связать с таблицей."""
    m = NAME_RE.match(stem)
    if not m or len(m.group("id")) != 5:
        return None, ""
    return int(m.group("id")), m.group("idx") or ""


def looks_like_national_code(digits: str) -> bool:
    """Контрольная сумма иранского национального кода (10 цифр).

    Нужна только для отчёта: сколько имён файлов похожи на персональные данные. Сами
    имена никуда не записываются.
    """
    if len(digits) != 10 or not digits.isdigit():
        return False
    d = [int(c) for c in digits]
    r = sum(d[i] * (10 - i) for i in range(9)) % 11
    return d[9] == r if r < 2 else d[9] == 11 - r


def anon_id(stem: str) -> str:
    return "arak_" + hashlib.sha1(stem.encode("utf-8")).hexdigest()[:12]


# --- снятие разметки сканера -----------------------------------------------------


def _line_kernel(length: int, angle_deg: float):  # noqa: ANN202
    import cv2

    k = np.zeros((length, length), np.uint8)
    c = (length - 1) / 2
    t = np.deg2rad(angle_deg)
    dx, dy = np.cos(t) * c, np.sin(t) * c
    cv2.line(k, (round(c - dx), round(c - dy)), (round(c + dx), round(c + dy)), 1, 1)
    return k


_LONG = None


def overlay_mask(a: np.ndarray, contrast: int = 25, length: int = 31) -> np.ndarray:
    """Маска разметки сканера: прямые тонкие линии и чёрные квадраты подписей.

    Линии ищутся как тонкие структуры, контрастные к локальной медиане, которые
    переживают морфологическое открытие отрезком длиной `length` хотя бы под одним из
    18 углов. Текстура кости короче и не прямая, поэтому остаётся нетронутой.
    """
    import cv2

    global _LONG
    if _LONG is None or _LONG[0] != length:
        _LONG = (length, [_line_kernel(length, ang) for ang in range(0, 180, 10)])
    a = np.ascontiguousarray(a, dtype=np.uint8)
    ones3 = np.ones((3, 3), np.uint8)

    black = (a <= 8).astype(np.uint8)
    labels = cv2.morphologyEx(black, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    labels = cv2.morphologyEx(labels, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))

    med = cv2.medianBlur(a, 5)
    thin = (cv2.absdiff(a, med) > contrast).astype(np.uint8)
    thick = cv2.dilate(thin, ones3)
    lines = np.zeros_like(thin)
    for k in _LONG[1]:
        lines |= cv2.morphologyEx(thick, cv2.MORPH_OPEN, k)
    lines &= thick
    return cv2.dilate(labels | lines, ones3)


def clean(a: np.ndarray, stretch_x: float = STRETCH_X) -> np.ndarray:
    """Кадр набора -> кадр, похожий на нашу выгрузку: без разметки, кость светлая."""
    import cv2

    a = np.ascontiguousarray(a, dtype=np.uint8)
    out = 255 - cv2.inpaint(a, overlay_mask(a), 3, cv2.INPAINT_TELEA)
    if stretch_x and abs(stretch_x - 1.0) > 1e-3:
        h, w = out.shape
        out = cv2.resize(out, (round(w * stretch_x), h), interpolation=cv2.INTER_LINEAR)
    return out


# --- таблица -----------------------------------------------------------------------


def _num(v: object) -> float | None:
    try:
        return float(v)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def read_table(path: Path) -> dict[int, dict]:
    import openpyxl

    wb = openpyxl.load_workbook(path, read_only=True)
    rows = list(wb.active.iter_rows(values_only=True))
    head = [str(h) for h in rows[0]]
    out: dict[int, dict] = {}
    for r in rows[1:]:
        rec = dict(zip(head, r, strict=False))
        pid = rec.get("IDENTIFIER_1")
        if isinstance(pid, int):
            out[pid] = rec
    return out


def targets(rec: dict, region: str) -> dict:
    if region == REGION_SPINE:
        return {"bmd": _num(rec.get("SPINE_BMD")), "tscore": _num(rec.get("SPINE_TSCORE")),
                "neck_bmd": None, "neck_tscore": None}  # fmt: skip
    return {"bmd": _num(rec.get("HIP_BMD")), "tscore": _num(rec.get("HIP_TSCORE")),
            "neck_bmd": _num(rec.get("HIPNECK_BMD")), "neck_tscore": _num(rec.get("HIPNECK_TSCORE"))}  # fmt: skip


# --- сборка -------------------------------------------------------------------------


def build(arak_root: Path, out: Path, n_folds: int = 5, seed: int = 0, limit: int | None = None) -> dict:
    table = read_table(arak_root / TABLE_NAME)
    (out / "images").mkdir(parents=True, exist_ok=True)
    files = sorted((arak_root / IMAGES_DIR).glob("*.png"))
    if limit:
        files = files[:limit]

    stats: Counter = Counter()
    rows: list[dict] = []
    for f in files:
        stats["файлов"] += 1
        pid, idx = parse_name(f.stem)
        if pid is None:
            digits = f.stem.split("-")[0]
            stats["без номера пациента из таблицы"] += 1
            stats["из них похожи на национальный код"] += int(looks_like_national_code(digits))
            continue
        rec = table.get(pid)
        if rec is None:
            stats["номера нет в таблице"] += 1
            continue
        region = SUFFIX_REGION.get(idx)
        img = Image.open(f)
        if img.size == LATERAL_SIZE:
            stats["боковая проекция (VFA)"] += 1
            continue
        if region is None:
            stats[f"суффикс «{idx or '—'}» не проверен"] += 1
            continue
        t = targets(rec, region)
        if t["bmd"] is None:
            stats["нет BMD в таблице"] += 1
            continue
        a = clean(np.asarray(img.convert("L")))
        iid = anon_id(f.stem)
        png = Path("images") / f"{iid}.png"
        Image.fromarray(a).save(out / png)
        rows.append({"image_id": iid, "png_path": str(png), "patient": pid, "region": region,
                     "age_category": rec.get("AGE_CATEGORY") or "", **t})  # fmt: skip
        stats[region] += 1

    patients = sorted({r["patient"] for r in rows})
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(patients))
    fold_of = {patients[i]: int(k % n_folds) for k, i in enumerate(order)}
    for r in rows:
        r["fold"] = fold_of[r["patient"]]

    with (out / "arak.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else r[k]) for k in FIELDS})
    report = {"снимков в датасете": len(rows), "пациентов": len(patients), **dict(stats)}
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", required=True, help="папка Densitometry_data (внутри ArakDATA)")
    ap.add_argument("--out", default="data/arak")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, help="взять первые N файлов — для быстрой проверки")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    root = Path(args.data_root)
    arak = root / "ArakDATA" if (root / "ArakDATA").is_dir() else root
    report = build(arak, Path(args.out), args.folds, args.seed, args.limit)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
