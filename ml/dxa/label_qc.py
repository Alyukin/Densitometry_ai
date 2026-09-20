"""Проверка и нормализация экспертной разметки перед обучением.

Исходный файл `разметка.xlsx` **никогда не изменяется** — он только читается.
Результат работы модуля — отдельная очищенная разметка (`labels_clean.csv`) плюс
журнал всех автоматических исправлений и список того, что машина решить не может.

Принцип: автоматически исправляется только то, для чего есть проверяемое
основание в самих данных. Медицинское содержание разметки не додумывается:
если два экспертных поля противоречат друг другу и нельзя сказать, какое верное,
строка не «чинится», а исключается из обучения и уходит на ручную проверку.

Правила
-------
R1  Нормализация значений. В файле отметки хранятся как текст ('0'/'1'), UID —
    с возможными пробелами. Приведение к int и strip не меняет смысла.

R2  Итог пуст, компоненты заполнены -> Итог = max(компонентов).
    Основание: колонка «Итог» по устройству листа является сводкой покомпонентных
    проверок; на этой выгрузке равенство Итог = max(компоненты) выполняется
    в 246 из 249 размеченных областей (98.8%).

R3  Итог = 1, но все компоненты 0 -> quality_class = 1, вид нарушения НЕИЗВЕСТЕН.
    Основание: «Итог» — собственное заключение эксперта о пригодности области,
    оно принимается как есть. Но ни одно нарушение из закрытого списка не
    отмечено, поэтому вид нарушения не выдумывается: violations_known = 0,
    головы нарушений для такой строки маскируются, голова качества учится.
    Строка дополнительно уходит в manual_review, чтобы эксперт проставил вид.

R4  Итог = 0, но какой-то компонент = 1 -> неразрешимое противоречие.
    Два экспертных поля прямо спорят о quality_class, и предпочесть одно другому
    можно только догадкой. Строка исключается из обучения (label_status=conflict)
    и уходит в manual_review.
    Сознательно НЕ применяется «Итог = max(компоненты), значит качество 1»:
    это переписало бы явное заключение эксперта на основании догадки.

R5  Значение вне {0, 1} -> ячейка считается незаполненной + manual_review.

R6  Повтор StudyInstanceUID -> все строки этого исследования исключаются +
    manual_review (непонятно, какая строка актуальна).

R7  Сверка со снимками.
    R7a разметка области есть, снимка нет -> строка отбрасывается (no_image) +
        manual_review;
    R7b снимок есть, разметки нет -> снимок просто не используется в обучении.
        Это не ошибка и не повод для ручной проверки, только цифра в отчёте.

R8  Сторона бедра по разметке. Если в исследовании ровно один снимок бедра и
    размечена ровно одна сторона — сторона берётся из разметки, а не из эвристики
    по положению диафиза. Основание: эксперт размечал тот снимок, который есть в
    исследовании; это наблюдение, а не вывод.

R9  Неоднозначная сторона в паре. Если у двух снимков бедра разница shaft_x
    меньше SIDE_GAP_MIN:
    R9a метки левого и правого бедра различаются -> сопоставить снимки с метками
        нельзя, оба снимка исключаются (ambiguous_side) + manual_review;
    R9b метки совпадают -> перестановка сторон ни на что не влияет, снимки
        остаются в обучении, в отчёте отмечается факт.
"""

from __future__ import annotations

import csv
import logging
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

from dxa.labels import REGION_FEMUR, REGION_SPINE, RawRow, load_raw

logger = logging.getLogger(__name__)

SIDE_GAP_MIN = 0.05  # минимальная разница shaft_x, при которой сторонам в паре можно верить

# Статусы строки очищенной разметки
ST_OK = "ok"  # годится для обучения полностью
ST_QUALITY_ONLY = "quality_only"  # учим только качество, вид нарушения неизвестен
ST_CONFLICT = "conflict"  # в обучение не берём
ST_NO_IMAGE = "no_image"  # разметка без снимка

TRAINABLE = (ST_OK, ST_QUALITY_ONLY)


@dataclass
class CleanLabel:
    study_uid: str
    excel_row: int
    region: str
    side: str  # "left" | "right" | "" для позвоночника
    quality_class: int
    violations: list[str]
    violations_known: int  # 1 = виды нарушений достоверны, 0 = маскировать головы нарушений
    label_status: str
    rules: str  # коды применённых правил через ";"
    comment: str

    @property
    def violation_type(self) -> str:
        return ";".join(self.violations)

    def as_row(self) -> dict:
        d = asdict(self)
        d["violations"] = self.violation_type
        return d


@dataclass
class Fix:
    """Одно автоматическое исправление: что изменили, почему и по какому правилу."""

    rule: str
    study_uid: str
    excel_row: int
    region: str
    side: str
    was: str
    became: str
    reason: str


@dataclass
class Review:
    """Случай, который нельзя решить автоматически."""

    priority: str  # высокий | средний
    rule: str
    study_uid: str
    excel_row: int
    region: str
    side: str
    question: str  # что именно должен решить эксперт
    evidence: str  # что видно в данных
    comment: str  # комментарий эксперта из xlsx


@dataclass
class QcResult:
    labels: list[CleanLabel] = field(default_factory=list)
    fixes: list[Fix] = field(default_factory=list)
    reviews: list[Review] = field(default_factory=list)
    side_hints: dict[str, str] = field(default_factory=dict)  # study_dir -> сторона (правило R8)
    ambiguous_sides: set[str] = field(default_factory=set)  # study_dir, где стороны не различить (R9a)
    side_agnostic: set[str] = field(default_factory=set)  # study_dir, где перестановка сторон безразлична (R9b)
    stats: dict = field(default_factory=dict)

    def by_study(self) -> dict[str, list[CleanLabel]]:
        out: dict[str, list[CleanLabel]] = {}
        for row in self.labels:
            out.setdefault(row.study_uid, []).append(row)
        return out


def _side_str(side: str | None) -> str:
    return side or ""


def _region_label(region: str, side: str) -> str:
    return f"{region} ({'правое' if side == 'right' else 'левое'})" if side else region


def _check_one_region(
    raw: RawRow,
    region: str,
    side: str | None,
    fixes: list[Fix],
    reviews: list[Review],
) -> CleanLabel | None:
    """Применяет правила R1–R5 к одной области одного исследования."""
    sd = _side_str(side)
    where = _region_label(region, sd)
    rules = ["R1"]

    comps = {code: val for (reg, s, code), val in raw.components.items() if reg == region and s == side}
    total = raw.totals.get((region, side))

    # R5: мусор в ячейке
    bad = [(code, val[1]) for code, val in comps.items() if isinstance(val, tuple)]
    if isinstance(total, tuple):
        bad.append(("Итог", total[1]))
        total = None
    for code, val in bad:
        comps.pop(code, None)
        rules.append("R5")
        fixes.append(
            Fix(
                "R5",
                raw.study_uid,
                raw.excel_row,
                where,
                sd,
                f"{code} = {val!r}",
                "пусто",
                "значение вне допустимого набора {0, 1}; ячейка считается незаполненной",
            )
        )
        reviews.append(
            Review(
                "высокий",
                "R5",
                raw.study_uid,
                raw.excel_row,
                where,
                sd,
                f"какое значение должно быть в «{code}»",
                f"в ячейке {val!r}, ожидается 0 или 1",
                raw.comment,
            )
        )

    filled = {code: val for code, val in comps.items() if isinstance(val, int)}
    if total is None and not filled:
        return None  # область не размечена — как правило, снимка нет

    violations = sorted(code for code, val in filled.items() if val == 1)
    derived = 1 if violations else 0
    status = ST_OK
    violations_known = 1

    if total is None:
        # R2: Итога нет, но компоненты заполнены
        total = derived
        rules.append("R2")
        fixes.append(
            Fix(
                "R2",
                raw.study_uid,
                raw.excel_row,
                where,
                sd,
                "Итог пуст",
                f"Итог = {total}",
                "Итог по устройству листа — сводка компонентов; взят max(компонентов)",
            )
        )
    elif not filled:
        # Итог есть, компонентов нет вовсе
        if total == 1:
            status = ST_QUALITY_ONLY
            violations_known = 0
            rules.append("R3")
            fixes.append(
                Fix(
                    "R3",
                    raw.study_uid,
                    raw.excel_row,
                    where,
                    sd,
                    "Итог = 1, компоненты не заполнены",
                    "quality_class = 1, вид нарушения неизвестен",
                    "заключение эксперта взято как есть; вид нарушения не выдумывается, "
                    "головы нарушений для этой строки маскируются",
                )
            )
            reviews.append(
                Review(
                    "средний",
                    "R3",
                    raw.study_uid,
                    raw.excel_row,
                    where,
                    sd,
                    "какое именно нарушение имелось в виду",
                    "Итог = 1, ни один компонент не заполнен",
                    raw.comment,
                )
            )
    elif total != derived:
        if total == 1 and derived == 0:
            # R3: эксперт забраковал область, но не отметил ни одного нарушения
            status = ST_QUALITY_ONLY
            violations_known = 0
            rules.append("R3")
            fixes.append(
                Fix(
                    "R3",
                    raw.study_uid,
                    raw.excel_row,
                    where,
                    sd,
                    "Итог = 1, все компоненты = 0",
                    "quality_class = 1, вид нарушения неизвестен",
                    "Итог — собственное заключение эксперта о пригодности, оно сохраняется; "
                    "ни одно нарушение из закрытого списка не отмечено, поэтому вид нарушения "
                    "не назначается, а головы нарушений маскируются",
                )
            )
            reviews.append(
                Review(
                    "средний",
                    "R3",
                    raw.study_uid,
                    raw.excel_row,
                    where,
                    sd,
                    "какое нарушение соответствует Итогу = 1",
                    "Итог = 1, все покомпонентные проверки = 0",
                    raw.comment,
                )
            )
        else:
            # R4: Итог = 0, но компонент отмечен — прямое противоречие
            status = ST_CONFLICT
            rules.append("R4")
            reviews.append(
                Review(
                    "высокий",
                    "R4",
                    raw.study_uid,
                    raw.excel_row,
                    where,
                    sd,
                    "что верно: Итог = 0 или отмеченное нарушение",
                    f"Итог = 0, но отмечено: {', '.join(violations)}",
                    raw.comment,
                )
            )

    return CleanLabel(
        study_uid=raw.study_uid,
        excel_row=raw.excel_row,
        region=region,
        side=sd,
        quality_class=int(total),
        violations=violations,
        violations_known=violations_known,
        label_status=status,
        rules=";".join(dict.fromkeys(rules)),
        comment=raw.comment,
    )


def _cross_check_images(result: QcResult, records: list) -> None:  # noqa: ANN001 (ImageRecord из inventory)
    """Правила R7–R9: сверка разметки со снимками и уточнение стороны."""
    uniq = [r for r in records if not r.is_duplicate]
    by_study: dict[str, list] = {}
    for rec in uniq:
        by_study.setdefault(rec.study_dir, []).append(rec)

    labels_by_study = result.by_study()
    n_unlabeled = 0

    for study, recs in by_study.items():
        rows = labels_by_study.get(study, [])
        femurs = [r for r in recs if r.region == REGION_FEMUR]
        has = {(REGION_SPINE, "")} if any(r.region == REGION_SPINE for r in recs) else set()
        labelled_sides = {row.side for row in rows if row.region == REGION_FEMUR}

        # R8: один снимок бедра и ровно одна размеченная сторона -> сторона из разметки
        if len(femurs) == 1 and len(labelled_sides) == 1:
            side = next(iter(labelled_sides))
            result.side_hints[study] = side
            if femurs[0].side != side:
                result.fixes.append(
                    Fix(
                        "R8",
                        study,
                        rows[0].excel_row if rows else 0,
                        REGION_FEMUR,
                        side,
                        f"сторона по эвристике: {femurs[0].side or '—'}",
                        f"сторона по разметке: {side}",
                        "в исследовании один снимок бедра и размечена ровно одна сторона; "
                        "сторона берётся из разметки, а не из положения диафиза",
                    )
                )
            femurs[0].side = side
            femurs[0].side_confidence = 1.0

        # R9: пара бедёр с неразличимым положением диафиза
        elif len(femurs) == 2:
            gap = abs(femurs[0].shaft_x - femurs[1].shaft_x)
            if gap < SIDE_GAP_MIN:
                fem_rows = {row.side: row for row in rows if row.region == REGION_FEMUR}
                left, right = fem_rows.get("left"), fem_rows.get("right")
                same = (
                    left is not None
                    and right is not None
                    and (left.quality_class, left.violations) == (right.quality_class, right.violations)
                )
                if same:
                    result.side_agnostic.add(study)
                    result.fixes.append(
                        Fix(
                            "R9b",
                            study,
                            left.excel_row,
                            REGION_FEMUR,
                            "left+right",
                            f"стороны различимы слабо (разница shaft_x = {gap:.3f})",
                            "снимки оставлены в обучении",
                            "разметка левого и правого бедра совпадает, поэтому возможная "
                            "перестановка сторон не меняет ни одной метки",
                        )
                    )
                else:
                    result.ambiguous_sides.add(study)
                    result.reviews.append(
                        Review(
                            "высокий",
                            "R9a",
                            study,
                            (left or right).excel_row if (left or right) else 0,
                            REGION_FEMUR,
                            "left+right",
                            "какой снимок бедра правый, а какой левый",
                            f"разница shaft_x = {gap:.3f} (< {SIDE_GAP_MIN}), а разметка сторон различается",
                            (left or right).comment if (left or right) else "",
                        )
                    )

        for rec in femurs:
            if rec.side:
                has.add((REGION_FEMUR, rec.side))

        # R7a: разметка есть, снимка нет
        for row in rows:
            if (row.region, row.side) not in has and row.label_status in TRAINABLE:
                row.label_status = ST_NO_IMAGE
                result.reviews.append(
                    Review(
                        "высокий",
                        "R7a",
                        study,
                        row.excel_row,
                        _region_label(row.region, row.side),
                        row.side,
                        "почему размечена область, для которой нет снимка",
                        f"в папке исследования нет снимка: {_region_label(row.region, row.side)}",
                        row.comment,
                    )
                )

        # R7b: снимок есть, разметки нет вовсе (исключённые по R4/R6 сюда не считаются)
        labelled_keys = {(row.region, row.side) for row in rows}
        n_unlabeled += len(has - labelled_keys)

    # разметка на исследование, которого нет в выгрузке
    missing = set(labels_by_study) - set(by_study)
    for study in sorted(missing):
        for row in labels_by_study[study]:
            row.label_status = ST_NO_IMAGE
        result.reviews.append(
            Review(
                "высокий",
                "R7a",
                study,
                labels_by_study[study][0].excel_row,
                "исследование целиком",
                "",
                "где снимки этого исследования",
                "в разметке есть строка, а папки с таким именем нет",
                labels_by_study[study][0].comment,
            )
        )
    result.stats["снимков без разметки (R7b)"] = n_unlabeled
    result.stats["исследований без снимков (R7a)"] = len(missing)
    result.stats["исследований с неразличимыми сторонами (R9a)"] = len(result.ambiguous_sides)


def run_qc(xlsx_path: str | Path, records: list | None = None) -> QcResult:
    """Проверяет разметку и возвращает очищенную версию + журнал правок и список на ручную проверку."""
    raw_rows = load_raw(xlsx_path)
    result = QcResult()

    # R6: повторы StudyInstanceUID
    seen = Counter(r.study_uid for r in raw_rows)
    dup_uids = {uid for uid, n in seen.items() if n > 1}

    for raw in raw_rows:
        if raw.study_uid in dup_uids:
            rows = [r.excel_row for r in raw_rows if r.study_uid == raw.study_uid]
            result.reviews.append(
                Review(
                    "высокий",
                    "R6",
                    raw.study_uid,
                    raw.excel_row,
                    "исследование целиком",
                    "",
                    "какая из строк актуальна",
                    f"этот StudyInstanceUID встречается в строках {rows}",
                    raw.comment,
                )
            )
        for region, side in ((REGION_SPINE, None), (REGION_FEMUR, "right"), (REGION_FEMUR, "left")):
            row = _check_one_region(raw, region, side, result.fixes, result.reviews)
            if row is None:
                continue
            if raw.study_uid in dup_uids:
                row.label_status = ST_CONFLICT
                row.rules = ";".join(dict.fromkeys(row.rules.split(";") + ["R6"]))
            result.labels.append(row)

    if records is not None:
        _cross_check_images(result, records)

    # дедупликация списка на ручную проверку (одна и та же строка могла попасть дважды)
    uniq_reviews: dict[tuple, Review] = {}
    for rv in result.reviews:
        uniq_reviews.setdefault((rv.rule, rv.study_uid, rv.region, rv.side), rv)
    result.reviews = list(uniq_reviews.values())

    trainable = [r for r in result.labels if r.label_status in TRAINABLE]
    result.stats.update(
        {
            "строк в xlsx": len(raw_rows),
            "исследований": len({r.study_uid for r in raw_rows}),
            "строк разметки": len(result.labels),
            "годных для обучения": len(trainable),
            "из них без вида нарушения": sum(1 for r in trainable if not r.violations_known),
            "исключено (противоречие)": sum(1 for r in result.labels if r.label_status == ST_CONFLICT),
            "исключено (нет снимка)": sum(1 for r in result.labels if r.label_status == ST_NO_IMAGE),
            "автоисправлений": len(result.fixes),
            "на ручную проверку": len(result.reviews),
        }
    )
    return result


# --------------------------------------------------------------------------- вывод


def write_outputs(result: QcResult, out_dir: str | Path) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    def dump(name: str, rows: list[dict], fields: list[str]) -> None:
        with open(out / name, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)

    dump(
        "labels_clean.csv",
        [row.as_row() for row in result.labels],
        list(CleanLabel.__dataclass_fields__),
    )
    dump("label_fixes.csv", [asdict(f) for f in result.fixes], list(Fix.__dataclass_fields__))
    dump(
        "manual_review.csv",
        [asdict(r) for r in sorted(result.reviews, key=lambda r: (r.priority != "высокий", r.excel_row))],
        list(Review.__dataclass_fields__),
    )
    (out / "label_qc_report.md").write_text(render_report(result), encoding="utf-8")


def render_report(result: QcResult) -> str:
    lines = [
        "# Проверка разметки",
        "",
        "Исходный файл `разметка.xlsx` не изменялся. Очищенная разметка для обучения — "
        "`labels_clean.csv`, журнал правок — `label_fixes.csv`, спорные случаи — `manual_review.csv`.",
        "",
        "## Итоги",
        "",
        "| Показатель | Значение |",
        "|---|---:|",
    ]
    lines += [f"| {k} | {v} |" for k, v in result.stats.items()]

    by_status = Counter(r.label_status for r in result.labels)
    lines += ["", "## Статусы строк разметки", "", "| Статус | Строк |", "|---|---:|"]
    lines += [f"| {k} | {v} |" for k, v in sorted(by_status.items())]

    trainable = [r for r in result.labels if r.label_status in TRAINABLE]
    cnt = Counter((r.region, r.side, r.quality_class) for r in trainable)
    lines += [
        "",
        "## Классы после очистки (только то, что идёт в обучение)",
        "",
        "| Область | 0 | 1 |",
        "|---|---:|---:|",
    ]
    for region, side in sorted({(r, s) for r, s, _ in cnt}):
        name = _region_label(region, side)
        lines.append(f"| {name} | {cnt[(region, side, 0)]} | {cnt[(region, side, 1)]} |")

    viol = Counter(v for r in trainable for v in r.violations)
    if viol:
        lines += ["", "## Нарушения", "", "| Нарушение | Случаев |", "|---|---:|"]
        lines += [f"| {k} | {v} |" for k, v in viol.most_common()]

    by_rule = Counter(f.rule for f in result.fixes)
    lines += ["", "## Автоматические исправления", ""]
    if by_rule:
        lines += ["| Правило | Применено |", "|---|---:|"]
        lines += [f"| {k} | {v} |" for k, v in sorted(by_rule.items())]
        lines += ["", "Все правки по одной:", ""]
        for f in result.fixes:
            lines.append(f"* **{f.rule}** строка {f.excel_row}, {f.region}: `{f.was}` → `{f.became}`. {f.reason}.")
    else:
        lines.append("Не потребовались.")

    lines += ["", "## Ручная проверка", ""]
    if result.reviews:
        lines += ["| Приоритет | Правило | Строка | Область | Что решить | Что видно | Комментарий эксперта |"]
        lines += ["|---|---|---:|---|---|---|---|"]
        for r in sorted(result.reviews, key=lambda r: (r.priority != "высокий", r.excel_row)):
            lines.append(
                f"| {r.priority} | {r.rule} | {r.excel_row} | {r.region} | "
                f"{r.question} | {r.evidence} | {r.comment or '—'} |"
            )
    else:
        lines.append("Пусто.")

    lines += [
        "",
        "## Что означают статусы",
        "",
        "* `ok` — используется в обучении полностью.",
        "* `quality_only` — используется только для головы «есть нарушение / нет»; "
        "головы видов нарушений на этой строке маскируются (правило R3).",
        "* `conflict` — в обучение не идёт: экспертные поля противоречат друг другу (R4/R6).",
        "* `no_image` — разметка есть, снимка нет (R7a).",
        "",
        "Описание правил — в docstring `ml/dxa/label_qc.py`.",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":  # pragma: no cover
    import argparse

    ap = argparse.ArgumentParser(description="Проверка разметки: правила, журнал правок, список на ручную проверку")
    ap.add_argument("xlsx", help="путь к разметка.xlsx (файл только читается)")
    ap.add_argument("--images", help="папка Исследования — включает сверку со снимками (R7–R9)")
    ap.add_argument("--out", default="qc", help="куда положить результаты")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    recs = None
    if args.images:
        from dxa.inventory import scan

        recs = scan(args.images)

    res = run_qc(args.xlsx, recs)
    write_outputs(res, args.out)
    for k, v in res.stats.items():
        print(f"{k}: {v}")
    print(f"\nрезультаты записаны в {args.out}/")
