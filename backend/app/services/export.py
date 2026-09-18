"""CSV / XLSX export in the format required by the technical specification."""

from __future__ import annotations

import csv
import io
import json
from collections.abc import Iterable
from datetime import UTC, datetime

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

from app.models import ImageResult

# Порядок колонок строго по ТЗ
EXPORT_COLUMNS = [
    "path_to_study",
    "study_uid",
    "image_uid",
    "anatomical_region",
    "quality_class",
    "violation_type",
    "processing_status",
    "time_of_processing",
]

CONTENT_TYPES = {
    "csv": "text/csv; charset=utf-8",
    "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
}


def result_to_row(r: ImageResult) -> dict:
    return {
        "path_to_study": r.path_to_study,
        "study_uid": r.study_uid or "",
        "image_uid": r.image_uid or "",
        "anatomical_region": r.anatomical_region or "",
        "quality_class": r.quality_class or "",
        "violation_type": r.violation_type or "",
        "processing_status": r.processing_status,
        "time_of_processing": round(r.time_of_processing or 0.0, 3),
    }


def to_csv(results: Iterable[ImageResult]) -> bytes:
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=EXPORT_COLUMNS, lineterminator="\n")
    writer.writeheader()
    for r in results:
        writer.writerow(result_to_row(r))
    return buf.getvalue().encode("utf-8")


def _autosize(ws) -> None:  # noqa: ANN001
    for col_cells in ws.columns:
        width = max((len(str(c.value)) if c.value is not None else 0) for c in col_cells)
        ws.column_dimensions[get_column_letter(col_cells[0].column)].width = min(max(width + 2, 10), 80)


def to_xlsx(results: Iterable[ImageResult], is_mock: bool = False) -> bytes:
    results = list(results)
    wb = Workbook()
    ws = wb.active
    ws.title = "results"
    ws.append(EXPORT_COLUMNS)
    header_fill = PatternFill("solid", fgColor="1F4E79")
    for cell in ws[1]:
        cell.font = Font(bold=True, color="FFFFFF")
        cell.fill = header_fill
        cell.alignment = Alignment(horizontal="center")
    for r in results:
        row = result_to_row(r)
        ws.append([row[c] for c in EXPORT_COLUMNS])
    ws.freeze_panes = "A2"
    ws.auto_filter.ref = ws.dimensions
    _autosize(ws)

    # Explainability sheet: per-check verdicts
    ws2 = wb.create_sheet("checks")
    ws2.append(["image_uid", "check", "title", "passed", "value", "confidence", "error"])
    for c in ws2[1]:
        c.font = Font(bold=True)
    for r in results:
        checks = (r.details or {}).get("checks") or []
        if not checks:
            ws2.append([r.image_uid, "", "", "", "", r.confidence, r.error_message])
        for ch in checks:
            value = ch.get("value")
            ws2.append(
                [
                    r.image_uid,
                    ch.get("code"),
                    ch.get("title"),
                    "yes" if ch.get("passed") else "no",
                    json.dumps(value, ensure_ascii=False) if isinstance(value, dict) else value,
                    r.confidence,
                    r.error_message,
                ]
            )
    _autosize(ws2)

    meta = wb.create_sheet("meta")
    meta.append(["generated_at", datetime.now(UTC).isoformat(timespec="seconds")])
    meta.append(["rows", len(results)])
    meta.append(["mock", "yes — тестовые результаты, не для клинического использования" if is_mock else "no"])
    _autosize(meta)

    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()
