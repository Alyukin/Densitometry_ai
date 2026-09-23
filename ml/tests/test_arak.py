"""Подготовка открытого набора Arak: что берётся, что выбрасывается, что не утекает.

Главное свойство — исходные имена файлов не попадают в результат: часть из них похожа на
национальные коды. Проверяется на синтетическом наборе той же структуры.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("cv2")
openpyxl = pytest.importorskip("openpyxl")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ml"))

from PIL import Image, ImageDraw  # noqa: E402

from dxa import arak  # noqa: E402
from dxa.labels import REGION_FEMUR, REGION_SPINE  # noqa: E402


def _valid_code(prefix9: str) -> str:
    """Синтетический номер с верной контрольной цифрой — не настоящий."""
    d = [int(c) for c in prefix9]
    r = sum(d[i] * (10 - i) for i in range(9)) % 11
    return prefix9 + str(r if r < 2 else 11 - r)


def test_parse_name_links_only_five_digit_patients() -> None:
    assert arak.parse_name("10002-1") == (10002, "1")
    assert arak.parse_name("13642-2") == (13642, "2")
    assert arak.parse_name("10002") == (10002, "")
    assert arak.parse_name(_valid_code("000000001")) == (None, "")
    assert arak.parse_name("abc-1") == (None, "")


def test_national_code_checksum() -> None:
    code = _valid_code("123456789")
    assert arak.looks_like_national_code(code)
    wrong = code[:-1] + str((int(code[-1]) + 1) % 10)
    assert not arak.looks_like_national_code(wrong)
    assert not arak.looks_like_national_code("12345")


def test_anon_id_is_stable_and_hides_the_name() -> None:
    name = _valid_code("000000002")
    assert arak.anon_id(name) == arak.anon_id(name)
    assert name not in arak.anon_id(name)


def _hologic_like(h: int = 300, w: int = 252, seed: int = 0) -> np.ndarray:
    """Кость тёмная на светлом фоне + рамка ROI, диагональ и чёрный квадрат подписи."""
    rng = np.random.default_rng(seed)
    a = np.full((h, w), 225, np.uint8)
    a[:, 90:150] = 90  # «кость»
    a = np.clip(a.astype(int) + rng.integers(-12, 12, a.shape), 0, 255).astype(np.uint8)
    img = Image.fromarray(a)
    d = ImageDraw.Draw(img)
    d.rectangle((30, 40, 220, 260), outline=40)
    d.line((20, 30, 230, 200), fill=250)
    d.rectangle((5, 5, 20, 20), fill=0)
    d.text((8, 8), "L1", fill=239)
    return np.asarray(img).copy()


def test_overlay_mask_finds_frame_but_not_bone() -> None:
    a = _hologic_like()
    m = arak.overlay_mask(a).astype(bool)
    assert m[40, 60:200].mean() > 0.9  # верхняя сторона рамки
    assert m[100:200, 30].mean() > 0.9  # левая сторона рамки
    assert m[8:18, 8:18].all()  # квадрат подписи
    assert m[150:250, 100:140].mean() < 0.1  # сама кость почти не тронута


def test_clean_inverts_and_stretches() -> None:
    a = _hologic_like()
    out = arak.clean(a)
    assert out.shape == (300, round(252 * arak.STRETCH_X))
    x0 = round(120 * arak.STRETCH_X)
    assert out[150, x0] > out[150, 10] + 80  # кость стала светлой, фон — тёмным


def _make_dataset(root: Path) -> Path:
    arak_dir = root / "ArakDATA"
    (arak_dir / arak.IMAGES_DIR).mkdir(parents=True)
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(["IDENTIFIER_1", "SPINE_BMD", "SPINE_TSCORE", "HIP_BMD", "HIP_TSCORE",
               "HIPNECK_BMD", "HIPNECK_TSCORE", "AGE_CATEGORY"])  # fmt: skip
    ws.append([10000, 0.95, -1.0, 0.90, -0.8, 0.70, -1.5, "60-64"])
    ws.append([10001, 0.80, -2.2, 0.85, -1.2, 0.65, -1.9, "65-69"])
    wb.save(arak_dir / arak.TABLE_NAME)
    img = Image.fromarray(_hologic_like())
    for stem in ("10000-1", "10000-2", "10001-2", "10001-3", "99999-1", _valid_code("000000003")):
        img.save(arak_dir / arak.IMAGES_DIR / f"{stem}.png")
    Image.new("L", arak.LATERAL_SIZE, 200).save(arak_dir / arak.IMAGES_DIR / "10001-1.png")
    return arak_dir


def test_build_keeps_linked_hips_and_spines_only(tmp_path: Path) -> None:
    src = _make_dataset(tmp_path / "src")
    out = tmp_path / "out"
    report = arak.build(src, out, n_folds=2)
    rows = list(csv.DictReader((out / "arak.csv").open(encoding="utf-8")))
    assert sorted(r["region"] for r in rows) == sorted([REGION_FEMUR, REGION_SPINE, REGION_SPINE])
    assert report["боковая проекция (VFA)"] == 1
    assert report["номера нет в таблице"] == 1
    assert report["без номера пациента из таблицы"] == 1
    assert report["из них похожи на национальный код"] == 1
    assert report["суффикс «3» не проверен"] == 1
    hip = next(r for r in rows if r["region"] == REGION_FEMUR)
    assert float(hip["bmd"]) == 0.90 and float(hip["neck_bmd"]) == 0.70
    # пациент целиком в одном фолде
    folds = {r["patient"]: set() for r in rows}
    for r in rows:
        folds[r["patient"]].add(r["fold"])
    assert all(len(f) == 1 for f in folds.values())


def test_build_output_never_contains_source_names(tmp_path: Path) -> None:
    src = _make_dataset(tmp_path / "src")
    out = tmp_path / "out"
    arak.build(src, out)
    blob = (out / "arak.csv").read_text(encoding="utf-8") + json.dumps(
        json.loads((out / "report.json").read_text(encoding="utf-8")), ensure_ascii=False
    )
    names = [p.stem for p in (src / arak.IMAGES_DIR).glob("*.png")]
    assert not any(n in blob for n in names)
    assert all(p.name.startswith("arak_") for p in (out / "images").iterdir())
