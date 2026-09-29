"""Отрисовка найденных структур поверх снимка — чтобы решение можно было проверить глазами.

Рисуются только измеренные величины: ось позвоночника и границы столба, контур
бедренной кости и ориентиры (шейка, вертелы, верх диафиза, малый вертел), рамка
отсканированного поля и найденные посторонние объекты.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from app.services.dicom import render_preview_png

logger = logging.getLogger(__name__)

# Меняется, когда меняется отрисовка: PNG кэшируются на диске, и без версии в имени файла
# старая картинка показывалась бы до повторной обработки.
OVERLAY_VERSION = 2

# В шрифте Pillow по умолчанию нет кириллицы — подписи выходили квадратиками. В образе
# Docker ставится fonts-dejavu-core (backend/Dockerfile), в Linux DejaVu обычно уже есть.
FONT_PATHS = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",  # Debian, Ubuntu
    "/usr/share/fonts/dejavu/DejaVuSans.ttf",  # Fedora, Alpine
    "DejaVuSans.ttf",  # в путях поиска шрифтов системы
)

COLOR_AXIS = (255, 80, 80)
COLOR_CONTOUR = (80, 200, 255)
COLOR_LANDMARK = (120, 255, 140)
COLOR_LT = (255, 90, 255)
COLOR_FOREIGN = (255, 210, 60)
COLOR_FIELD = (90, 110, 140)


def _scale(pt: list[float] | tuple[float, float], k: float) -> tuple[float, float]:
    return pt[0] * k, pt[1] * k


@lru_cache(maxsize=8)
def label_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Шрифт с кириллицей для подписей; без него — шрифт по умолчанию и предупреждение."""
    for path in FONT_PATHS:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    logger.warning("Нет шрифта DejaVu: подписи на разметке будут без русских букв")
    return ImageFont.load_default()


def _label(d: ImageDraw.ImageDraw, xy: tuple[float, float], text: str, color: tuple, scale: int) -> None:
    """Подпись слева направо от точки, по центру по высоте, с тёмной обводкой — читается на кости."""
    font = label_font(max(11, round(4.5 * scale)))
    try:
        d.text(xy, text, fill=color, font=font, anchor="lm", stroke_width=max(1, scale // 2), stroke_fill=(0, 0, 0))
    except (ValueError, TypeError):  # растровый шрифт по умолчанию не умеет anchor и обводку
        d.text((xy[0], xy[1] - 6), text, fill=color, font=font)


def render_overlay_png(dicom_path: Path, overlay: dict, scale: int = 3) -> bytes:
    """Собирает PNG: превью снимка + геометрия из `details.overlay`."""
    import io

    base = Image.open(io.BytesIO(render_preview_png(dicom_path))).convert("RGB")
    shape = overlay.get("shape")
    k = 1.0
    if shape and shape[1]:
        k = base.width / float(shape[1])
    img = base.resize((base.width * scale, base.height * scale), Image.LANCZOS)
    k *= scale
    d = ImageDraw.Draw(img)

    fb = overlay.get("field_bbox")
    if fb:
        y0, y1, x0, x1 = fb
        d.rectangle([x0 * k, y0 * k, x1 * k, y1 * k], outline=COLOR_FIELD, width=1)

    for name in ("axis", "shaft_axis"):
        seg = overlay.get(name)
        if seg and len(seg) == 2:
            d.line([_scale(seg[0], k), _scale(seg[1], k)], fill=COLOR_AXIS, width=max(2, scale))

    for name in ("column_left", "column_right"):
        pts = overlay.get(name) or []
        for x, y in pts:
            d.ellipse([x * k - 1.5, y * k - 1.5, x * k + 1.5, y * k + 1.5], fill=COLOR_CONTOUR)

    for left, right, y in overlay.get("femur_contour") or []:
        for x in (left, right):
            d.ellipse([x * k - 1.5, y * k - 1.5, x * k + 1.5, y * k + 1.5], fill=COLOR_CONTOUR)

    for name, label in (("neck", "шейка"), ("trochanter", "вертелы"), ("shaft_top", "верх диафиза")):
        pt = overlay.get(name)
        if pt:
            x, y = _scale(pt, k)
            r = 4 * scale / 3
            d.ellipse([x - r, y - r, x + r, y + r], outline=COLOR_LANDMARK, width=2)
            _label(d, (x + r + 3, y), label, COLOR_LANDMARK, scale)

    lt = overlay.get("lesser_trochanter")
    if lt:
        x, y = _scale(lt, k)
        r = 5 * scale / 3
        d.ellipse([x - r, y - r, x + r, y + r], outline=COLOR_LT, width=2)
        _label(d, (x + r + 3, y), "малый вертел", COLOR_LT, scale)

    for box in overlay.get("foreign") or []:
        y0, y1, x0, x1 = box
        d.rectangle([x0 * k - 2, y0 * k - 2, x1 * k + 2, y1 * k + 2], outline=COLOR_FOREIGN, width=2)

    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()
