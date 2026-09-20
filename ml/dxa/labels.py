"""Разбор экспертной разметки (разметка.xlsx) в длинную таблицу на уровне «исследование × область».

Структура файла (лист «Калибровка»), проверено на выгрузке от 24.06.2026:
  A  №
  B  study                    — StudyInstanceUID
  C  Позвоночник / корректная укладка
  D  Позвоночник / правильно выравнена ось позвоночника (до 5°)
  E  Позвоночник / наличие посторонних предметов, артефактов, наложений
  F  Правое бедро / позиционирование-ротация
  G  Правое бедро / корректность области интереса
  H  Левое бедро  / позиционирование-ротация
  I  Левое бедро  / корректность области интереса
  J  Итог / Позвоночник
  K  Итог / Правое бедро
  L  Итог / Левое бедро
  M  Комментарий
  N+ сводный блок (не разметка, игнорируется)

Значение 1 = нарушение есть, 0 = нарушения нет, пусто = область не размечена
(обычно потому, что в исследовании нет соответствующего снимка).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from openpyxl import load_workbook

# Закрытые списки значений из ответов заказчика (вопросы 6 и 15).
REGION_SPINE = "Поясничный отдел позвоночника"
REGION_FEMUR = "Проксимальный отдел бедра"

VIOL_POSITION = "Некорректная укладка"
VIOL_SPINE_AXIS = "Не выравнена ось позвоночника"
VIOL_FOREIGN = "Присутствуют посторонние предметы"
VIOL_FEMUR_ROI = "Некорректная область интереса"

SPINE_VIOLATIONS = [VIOL_POSITION, VIOL_SPINE_AXIS, VIOL_FOREIGN]
FEMUR_VIOLATIONS = [VIOL_POSITION, VIOL_FEMUR_ROI]

# Колонка Excel -> (область, сторона, код нарушения)
COMPONENTS = {
    2: (REGION_SPINE, None, VIOL_POSITION),
    3: (REGION_SPINE, None, VIOL_SPINE_AXIS),
    4: (REGION_SPINE, None, VIOL_FOREIGN),
    5: (REGION_FEMUR, "right", VIOL_POSITION),
    6: (REGION_FEMUR, "right", VIOL_FEMUR_ROI),
    7: (REGION_FEMUR, "left", VIOL_POSITION),
    8: (REGION_FEMUR, "left", VIOL_FEMUR_ROI),
}
TOTALS = {
    9: (REGION_SPINE, None),
    10: (REGION_FEMUR, "right"),
    11: (REGION_FEMUR, "left"),
}
COMMENT_COL = 12
UID_COL = 1
UID_RE = re.compile(r"^[0-9.]+$")


@dataclass
class RawRow:
    """Ячейки одной строки Excel как есть, без какой-либо интерпретации.

    Нужен слою проверки разметки (label_qc), чтобы правила применялись к исходным
    значениям, а не к уже «починенным». Именно этот разбор — единственное место,
    где читается xlsx; всё остальное работает с RawRow.
    """

    excel_row: int
    study_uid: str
    comment: str
    # (область, сторона, код нарушения) -> 0 | 1 | None (пусто) | ("BAD", исходное значение)
    components: dict[tuple[str, str | None, str], object]
    totals: dict[tuple[str, str | None], object]  # (область, сторона) -> то же самое


def _cell_raw(value) -> object:  # noqa: ANN001
    """0 | 1 | None (пусто) | ("BAD", значение) — без «тихого» превращения мусора в пусто."""
    if value is None or str(value).strip() == "":
        return None
    try:
        v = int(str(value).strip())
    except (TypeError, ValueError):
        return ("BAD", value)
    return v if v in (0, 1) else ("BAD", value)


def load_raw(path: str | Path, sheet: str | None = None) -> list[RawRow]:
    """Читает xlsx построчно без интерпретации. Файл только читается, не меняется."""
    wb = load_workbook(path, data_only=True, read_only=True)
    ws = wb[sheet] if sheet else wb.worksheets[0]
    out: list[RawRow] = []
    for excel_row, values in enumerate(ws.iter_rows(values_only=True), start=1):
        if len(values) <= UID_COL:
            continue
        uid = str(values[UID_COL] or "").strip()
        if not uid or not UID_RE.match(uid):
            continue  # шапка и сводный блок справа
        out.append(
            RawRow(
                excel_row=excel_row,
                study_uid=uid,
                comment=str(values[COMMENT_COL] or "").strip() if len(values) > COMMENT_COL else "",
                components={
                    key: (_cell_raw(values[col]) if len(values) > col else None) for col, key in COMPONENTS.items()
                },
                totals={key: (_cell_raw(values[col]) if len(values) > col else None) for col, key in TOTALS.items()},
            )
        )
    wb.close()
    return out


if __name__ == "__main__":  # pragma: no cover
    import argparse
    import collections

    ap = argparse.ArgumentParser(description="Сырой разбор разметки (без правил; правила — в dxa.label_qc)")
    ap.add_argument("xlsx")
    args = ap.parse_args()
    raw = load_raw(args.xlsx)
    print(f"строк-исследований: {len(raw)}, уникальных UID: {len({r.study_uid for r in raw})}")
    filled = collections.Counter()
    for r in raw:
        for key, val in list(r.components.items()) + [((*k, "Итог"), v) for k, v in r.totals.items()]:
            if isinstance(val, int):
                filled[(key[0], key[1] or "", key[2], val)] += 1
    for k in sorted(filled, key=str):
        print("  ", k, filled[k])
