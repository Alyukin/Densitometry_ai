"""Отрисовка найденных структур поверх снимка — чтобы решение можно было проверить глазами.

Рисуются только измеренные величины: ось позвоночника и границы столба, контур
бедренной кости и ориентиры (шейка, вертелы, верх диафиза, малый вертел), рамка
отсканированного поля и найденные посторонние объекты.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

from app.services.dicom import render_preview_png

COLOR_AXIS = (255, 80, 80)
COLOR_CONTOUR = (80, 200, 255)
COLOR_LANDMARK = (120, 255, 140)
COLOR_LT = (255, 90, 255)
COLOR_FOREIGN = (255, 210, 60)
COLOR_FIELD = (90, 110, 140)


def _scale(pt: list[float] | tuple[float, float], k: float) -> tuple[float, float]:
    return pt[0] * k, pt[1] * k


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
            d.text((x + r + 2, y - 6), label, fill=COLOR_LANDMARK)

    lt = overlay.get("lesser_trochanter")
    if lt:
        x, y = _scale(lt, k)
        r = 5 * scale / 3
        d.ellipse([x - r, y - r, x + r, y + r], outline=COLOR_LT, width=2)
        d.text((x + r + 2, y - 6), "малый вертел", fill=COLOR_LT)

    for box in overlay.get("foreign") or []:
        y0, y1, x0, x1 = box
        d.rectangle([x0 * k - 2, y0 * k - 2, x1 * k + 2, y1 * k + 2], outline=COLOR_FOREIGN, width=2)

    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()
