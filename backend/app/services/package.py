"""ZIP-пакет результата: DICOM SR, вторичная серия с разметкой и таблица по ТЗ.

Одна кнопка вместо трёх: то, что нужно положить обратно в PACS (SR и серия с
разметкой), и то, что нужно человеку (CSV/XLSX), собирается в один архив.
"""

from __future__ import annotations

import io
import logging
import zipfile
from pathlib import Path

from pydicom.uid import generate_uid

from app.models import Study
from app.services import export
from app.services.dicom_sr import build_secondary_capture, build_sr, to_bytes
from app.services.overlay import render_overlay_png

logger = logging.getLogger(__name__)

README = """Densitometry AI — результат контроля качества DXA

sr.dcm                  DICOM Structured Report: область, класс качества, тип нарушения,
                        измеренные величины с критериями и координаты найденных структур.
overlay/*.dcm           Вторичная серия (Secondary Capture): снимок с наложенной разметкой.
overlay/*.png           То же изображение в PNG — открыть без DICOM-вьюера.
results.csv             Таблица в формате ТЗ: одна строка на изображение.
results.xlsx            То же плюс лист со всеми проверками.

Результат автоматического контроля качества, не медицинское заключение.
"""


def build_zip(study: Study, data_dir: Path) -> bytes:
    results = list(study.results)
    by_image = {i.id: i for i in study.images}
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("README.txt", README)
        z.writestr("results.csv", export.to_csv(results))
        z.writestr("results.xlsx", export.to_xlsx(results, study.is_mock))
        z.writestr("sr.dcm", to_bytes(build_sr(study, results, data_dir)))

        series_uid = generate_uid()
        n = 0
        for r in results:
            overlay = (r.details or {}).get("overlay")
            image = by_image.get(r.image_id or "")
            if not overlay or image is None:
                continue
            source = data_dir / image.stored_path
            if not source.exists():
                continue
            try:
                png = render_overlay_png(source, overlay)
            except Exception:  # noqa: BLE001 — одна неудачная картинка не должна ронять весь пакет
                logger.warning("Не удалось построить разметку для %s", image.original_filename, exc_info=True)
                continue
            n += 1
            # имена файлов в исследовании повторяются (IM0001.dcm в каждой серии),
            # поэтому в архиве к ним добавляется порядковый номер
            stem = f"{n:02d}_{Path(image.original_filename).stem or 'image'}"
            label = f"{r.anatomical_region or 'DXA'}: {r.violation_type or 'нарушений не найдено'}"
            z.writestr(f"overlay/{stem}.png", png)
            z.writestr(f"overlay/{stem}.dcm", to_bytes(build_secondary_capture(png, source, label, n, series_uid)))
    return buf.getvalue()
