"""Сборка обучающего датасета: инвентарь DICOM + проверенная разметка -> dataset.csv, PNG, отчёт.

Запуск:
    python -m dxa.build_dataset \
        --data-root "/path/Densitometry_data" \
        --out data/processed

Разметка перед использованием проходит проверку (`dxa.label_qc`): исходный xlsx только
читается, в обучение идёт очищенная версия, все автоматические правки записываются в журнал.

Результат:
    data/processed/manifest.csv        — все файлы DICOM (включая дубликаты)
    data/processed/dataset.csv         — по одному снимку на строку + метки + фолды
    data/processed/images/*.png        — 8-битные PNG уникальных снимков
    data/processed/report.md           — отчёт по данным
    data/processed/labels_clean.csv    — очищенная разметка
    data/processed/label_fixes.csv     — журнал автоматических исправлений
    data/processed/manual_review.csv   — что нужно посмотреть эксперту
    data/processed/label_qc_report.md  — отчёт по проверке разметки
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import logging
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pydicom
from PIL import Image

from dxa import inventory
from dxa.label_qc import TRAINABLE, CleanLabel, QcResult, run_qc, write_outputs
from dxa.labels import REGION_FEMUR, REGION_SPINE

logger = logging.getLogger(__name__)

DATASET_FIELDS = [
    "image_id",
    "study_uid",
    "study_dir",
    "dicom_path",
    "png_path",
    "region",
    "side",
    "side_confidence",
    "rows",
    "columns",
    "quality_class",
    "violation_type",
    "violations_known",
    "label_status",
    "label_source",
    "fold",
    "warnings",
]


def _label_index(rows: list[CleanLabel]) -> dict[tuple[str, str, str], CleanLabel]:
    return {(r.study_uid, r.region, r.side): r for r in rows}


def _match_label(
    rec: inventory.ImageRecord,
    by_key: dict[tuple[str, str, str], CleanLabel],
    qc: QcResult,
) -> tuple[CleanLabel | None, str]:
    """Возвращает (метка, источник метки) для снимка.

    ВАЖНО: разметка ссылается на ИМЯ ПАПКИ исследования, а не на StudyInstanceUID из тега.
    В выгрузке это разные идентификаторы (папка 2.25.*, тег 1.2.643.*), поэтому join идёт
    по `study_dir`.

    Стороны бедра к моменту вызова уже разобраны в label_qc (правила R8/R9), поэтому здесь
    остаётся только взять метку по стороне снимка и отсеять то, что QC признал негодным.
    """
    key = rec.study_dir
    side = "" if rec.region == REGION_SPINE else rec.side
    if rec.region == REGION_FEMUR and key in qc.ambiguous_sides:
        return None, "ambiguous_side"

    lab = by_key.get((key, rec.region, side))
    if lab is None:
        return None, "unlabeled"
    if lab.label_status not in TRAINABLE:
        return None, lab.label_status  # conflict | no_image — в обучение не идёт
    if rec.region == REGION_FEMUR and key in qc.side_agnostic:
        return lab, "side_agnostic"
    if rec.region == REGION_FEMUR and key in qc.side_hints:
        return lab, "label_side"
    return lab, "direct"


def _study_stratum(labels_by_study: dict[str, list[CleanLabel]], study_uid: str) -> str:
    rows = labels_by_study.get(study_uid, [])
    sp = next((r for r in rows if r.region == REGION_SPINE), None)
    fem = [r for r in rows if r.region == REGION_FEMUR]
    s = "-" if sp is None else str(sp.quality_class)
    f = "-" if not fem else str(max(r.quality_class for r in fem))
    return f"sp{s}_fem{f}"


def _assign_folds(studies: list[str], strata: dict[str, str], n_folds: int, seed: int) -> dict[str, int]:
    """Разбиение по исследованиям (группам) со стратификацией по типу нарушений."""
    rng = np.random.default_rng(seed)
    folds: dict[str, int] = {}
    by_stratum: dict[str, list[str]] = {}
    for st in studies:
        by_stratum.setdefault(strata.get(st, "?"), []).append(st)
    for _stratum, group in sorted(by_stratum.items()):
        items = list(group)
        rng.shuffle(items)
        for i, st in enumerate(items):
            folds[st] = i % n_folds
    return folds


def export_png(dicom_path: Path, png_path: Path) -> None:
    ds = pydicom.dcmread(dicom_path, force=True)
    arr = ds.pixel_array
    if arr.dtype != np.uint8:  # на текущей выгрузке всегда uint8, но подстрахуемся
        a = arr.astype(np.float32)
        lo, hi = np.percentile(a, (0.5, 99.5))
        arr = np.clip((a - lo) / max(hi - lo, 1e-6), 0, 1) * 255
        arr = arr.astype(np.uint8)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(png_path, optimize=True)


def build(
    data_root: Path,
    out_dir: Path,
    n_folds: int = 5,
    seed: int = 42,
    write_png: bool = True,
) -> dict:
    studies_root = data_root / "НД_для_обучения" / "Исследования"
    xlsx = data_root / "НД_для_обучения" / "разметка.xlsx"
    if not studies_root.exists():
        raise SystemExit(f"нет папки с исследованиями: {studies_root}")
    if not xlsx.exists():
        raise SystemExit(f"нет файла разметки: {xlsx}")

    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("сканирую DICOM в %s", studies_root)
    records = inventory.scan(studies_root)

    logger.info("проверяю разметку %s (файл только читается)", xlsx.name)
    qc = run_qc(xlsx, records)  # правила R8/R9 могут уточнить сторону прямо в records
    write_outputs(qc, out_dir)
    logger.info(
        "разметка: годных строк %s, автоправок %s, на ручную проверку %s",
        qc.stats["годных для обучения"],
        qc.stats["автоисправлений"],
        qc.stats["на ручную проверку"],
    )
    by_key = _label_index(qc.labels)
    labels_by_study = {k: [r for r in v if r.label_status in TRAINABLE] for k, v in qc.by_study().items()}

    with (out_dir / "manifest.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(asdict(records[0])))
        w.writeheader()
        w.writerows(inventory.to_rows(records))

    uniq = [r for r in records if not r.is_duplicate]
    strata = {st: _study_stratum(labels_by_study, st) for st in {r.study_dir for r in uniq}}
    folds = _assign_folds(sorted(strata), strata, n_folds, seed)

    rows: list[dict] = []
    for rec in uniq:
        lab, source = _match_label(rec, by_key, qc)
        png_rel = f"images/{rec.study_dir[-16:]}_{rec.sop_uid[-10:]}.png"
        if write_png:
            export_png(studies_root / rec.path, out_dir / png_rel)
        rows.append(
            {
                "image_id": f"{rec.study_dir[-16:]}_{rec.sop_uid[-10:]}",
                "study_uid": rec.study_uid,
                "study_dir": rec.study_dir,
                "dicom_path": rec.path,
                "png_path": png_rel,
                "region": rec.region,
                "side": rec.side,
                "side_confidence": rec.side_confidence,
                "rows": rec.rows,
                "columns": rec.columns,
                "quality_class": "" if lab is None else lab.quality_class,
                "violation_type": "" if lab is None else lab.violation_type,
                "violations_known": "" if lab is None else lab.violations_known,
                "label_status": "" if lab is None else lab.label_status,
                "label_source": source,
                "fold": folds.get(rec.study_dir, -1) if lab is not None else -1,
                "warnings": rec.warnings,
            }
        )

    with (out_dir / "dataset.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=DATASET_FIELDS)
        w.writeheader()
        w.writerows(rows)

    stats = summarize(records, uniq, rows, qc, labels_by_study)
    (out_dir / "report.md").write_text(render_report(stats, data_root), encoding="utf-8")
    (out_dir / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("готово: %s", out_dir)
    return stats


def summarize(records, uniq, rows, qc: QcResult, labels_by_study) -> dict:  # noqa: ANN001
    per_study_files = collections.Counter(r.study_dir for r in records)
    per_study_uniq = collections.Counter(r.study_dir for r in uniq)
    comp = collections.Counter()
    for st in per_study_uniq:
        imgs = [r for r in uniq if r.study_dir == st]
        comp[
            (
                sum(i.region == REGION_SPINE for i in imgs),
                sum(i.region == REGION_FEMUR for i in imgs),
            )
        ] += 1

    labelled = [r for r in rows if r["quality_class"] != ""]
    by_region = collections.Counter((r["region"], r["quality_class"]) for r in labelled)
    viol = collections.Counter(v for r in labelled for v in filter(None, r["violation_type"].split(";")))
    sources = collections.Counter(r["label_source"] for r in rows)

    study_level = collections.Counter(
        (r.region, r.side, r.quality_class) for r in qc.labels if r.label_status in TRAINABLE
    )
    hips = [(s, [r for r in rs if r.region == REGION_FEMUR]) for s, rs in labels_by_study.items()]
    both = [(s, rs) for s, rs in hips if len(rs) == 2]
    disagree = [s for s, rs in both if rs[0].quality_class != rs[1].quality_class]

    return {
        "files_total": len(records),
        "files_unique": len(uniq),
        "studies": len(per_study_files),
        "files_per_study": dict(sorted(collections.Counter(per_study_files.values()).items())),
        "unique_per_study": dict(sorted(collections.Counter(per_study_uniq.values()).items())),
        "composition_spine_femur": {f"{k[0]}+{k[1]}": v for k, v in sorted(comp.items())},
        "images_labelled": len(labelled),
        "images_by_region_class": {f"{k[0]} / класс {k[1]}": v for k, v in sorted(by_region.items(), key=str)},
        "violations_image_level": dict(viol),
        "label_sources": dict(sources),
        "study_label_rows": {
            f"{k[0]} {k[1]} / класс {k[2]}".replace("  ", " "): v for k, v in sorted(study_level.items(), key=str)
        },
        "studies_with_both_hips": len(both),
        "studies_hips_disagree": len(disagree),
        "studies_in_labels": qc.stats["исследований"],
        "images_quality_only": sum(1 for r in rows if r["violations_known"] == 0),
        "label_qc": qc.stats,
        "label_fixes": [f"{f.rule} строка {f.excel_row} {f.region}: {f.was} -> {f.became}" for f in qc.fixes],
        "manual_review": [f"{r.priority} {r.rule} строка {r.excel_row} {r.region}: {r.question}" for r in qc.reviews],
    }


def render_report(s: dict, data_root: Path) -> str:
    def table(d: dict, k: str, v: str) -> str:
        head = f"| {k} | {v} |\n|---|---|\n"
        return head + "".join(f"| {a} | {b} |\n" for a, b in d.items())

    return f"""# Отчёт по данным

Источник: `{data_root}`

## Объём

- файлов DICOM: **{s["files_total"]}**
- уникальных снимков (после дедупликации по пикселям): **{s["files_unique"]}**
- исследований: **{s["studies"]}**, строк в разметке: **{s["studies_in_labels"]}**

Файлов на исследование: {s["files_per_study"]}

Уникальных снимков на исследование: {s["unique_per_study"]}

Состав исследования (позвоночник + бедро): {s["composition_spine_femur"]}

## Метки

Размечено снимков: **{s["images_labelled"]}** из {s["files_unique"]}

{table(s["images_by_region_class"], "область и класс (0=норма, 1=нарушение)", "снимков")}

Нарушения (на уровне снимков):

{table(s["violations_image_level"], "нарушение", "снимков")}

Источник метки:

{table(s["label_sources"], "источник", "снимков")}

- `direct` — сторона снимка определена по изображению и взята соответствующая колонка разметки
- `label_side` — сторона взята из разметки (в исследовании один снимок бедра, правило R8)
- `side_agnostic` — стороны различимы слабо, но метки обеих сторон совпадают (R9b)
- `ambiguous_side` — стороны не различить, а метки разные: снимок исключён (R9a)
- `conflict` — разметка противоречива, снимок исключён (R4/R6)
- `unlabeled` — для области нет разметки

Снимков, у которых известен только класс качества, но не вид нарушения: **{s["images_quality_only"]}**
(правило R3, головы нарушений на них маскируются).

## Проверка разметки

Исходный `разметка.xlsx` не изменялся. Подробности — в `label_qc_report.md`,
правки — в `label_fixes.csv`, спорные случаи — в `manual_review.csv`.

{table(s["label_qc"], "показатель", "значение")}

Автоматические исправления ({len(s["label_fixes"])}):

{"".join(f"- {i}" + chr(10) for i in s["label_fixes"]) or "- нет"}

На ручную проверку ({len(s["manual_review"])}):

{"".join(f"- {i}" + chr(10) for i in s["manual_review"]) or "- нет"}

## Прочее

- исследований с разметкой обоих бёдер: {s["studies_with_both_hips"]},
  из них метки левого и правого бедра различаются: **{s["studies_hips_disagree"]}**
  (в таких исследованиях сторона снимка обязана быть определена верно)
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", required=True, help="папка Densitometry_data")
    ap.add_argument("--out", default="data/processed")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-png", action="store_true", help="не выгружать PNG (только таблицы)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    stats = build(Path(args.data_root), Path(args.out), args.folds, args.seed, not args.no_png)
    print(json.dumps(stats, ensure_ascii=False, indent=2)[:2000])


if __name__ == "__main__":  # pragma: no cover
    main()
