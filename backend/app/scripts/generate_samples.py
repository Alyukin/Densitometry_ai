"""Generate synthetic DXA-like DICOM files for testing the pipeline (no real patient data).

Usage:
    python -m app.scripts.generate_samples --out ../samples
"""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

import numpy as np
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

SECONDARY_CAPTURE = "1.2.840.10008.5.1.4.1.1.7"

# Геометрия фантомов вынесена в константы: по ним же строится «истина» для метрик
# локализации (ml/baseline/localization.py), чтобы картинка и эталон не разъехались.
SPINE_N_VERTEBRAE = 5
SPINE_Y0 = 45  # центр L1 по вертикали
SPINE_DY = 45  # шаг между телами позвонков
SPINE_HALF_W = 36  # половина ширины тела позвонка
SPINE_HALF_H = 17  # половина высоты тела позвонка

HIP_SHAFT_X0 = 96.0  # ось диафиза у нижнего края кадра
HIP_SHAFT_HALF_W = 21  # половина ширины диафиза
HIP_SHAFT_TOP_Y = 150  # верх диафиза
HIP_TROCH_X, HIP_TROCH_Y = 84, 120  # центр большого вертела
# Малый вертел: смещение от оси диафиза и высота. Бугорок должен выступать за
# медиальный край шейки — иначе на проекции его попросту не видно, и фантом
# перестаёт быть похожим на снимок, по которому оценивают ротацию.
HIP_LT_DX, HIP_LT_Y = 33, 147


def _uid(*parts: str) -> str:
    return generate_uid(entropy_srcs=["densitometry-demo", *parts])


def _soft_tissue(rng: np.random.Generator, h: int, w: int, margin_x: int, margin_y: int) -> np.ndarray:
    """Мягкие ткани на нулевом фоне — как в выгрузке сканера: вне тела строго 0."""
    img = np.zeros((h, w), dtype=np.float32)
    yy, xx = np.mgrid[0:h, 0:w]
    body = (xx > margin_x) & (xx < w - margin_x) & (yy > margin_y) & (yy < h - margin_y)
    img[body] = rng.normal(78, 6, int(body.sum()))
    return img


def _finish(img: np.ndarray) -> np.ndarray:
    """8 бит, потолок 252 и мягкие границы — как у GE Lunar Prodigy в этой выгрузке.

    Размытие важно: у настоящего денситометра нет резких ступенек яркости, а
    детектор посторонних предметов ищет как раз тонкие резкие структуры.
    """
    inside = img > 0
    blurred = _blur(img, 2.0)
    return np.clip(np.where(inside, blurred, 0.0), 0, 252).astype(np.uint8)


def _blur(img: np.ndarray, sigma: float) -> np.ndarray:
    """Гауссово сглаживание через разделимую свёртку (без scipy)."""
    r = max(1, int(sigma * 3))
    x = np.arange(-r, r + 1, dtype=np.float32)
    k = np.exp(-(x**2) / (2 * sigma**2))
    k /= k.sum()
    out = np.apply_along_axis(lambda m: np.convolve(m, k, mode="same"), 0, img.astype(np.float32))
    return np.apply_along_axis(lambda m: np.convolve(m, k, mode="same"), 1, out)


def _spine_image(
    rng: np.random.Generator,
    h: int = 290,
    w: int = 300,
    tilt: float = 0.0,
    iliac: bool = True,
    foreign: bool = False,
) -> np.ndarray:
    """Поясничный отдел: столб позвонков, при iliac=True — крылья подвздошных костей."""
    img = _soft_tissue(rng, h, w, margin_x=18, margin_y=6)
    yy, xx = np.mgrid[0:h, 0:w]
    cx = w / 2
    for i in range(SPINE_N_VERTEBRAE):  # L1..L5
        cy = SPINE_Y0 + i * SPINE_DY
        x0 = cx + tilt * (cy - h / 2)
        body = (np.abs(xx - x0) < SPINE_HALF_W) & (np.abs(yy - cy) < SPINE_HALF_H)
        img[body] += 110
        disc = (np.abs(xx - x0) < 34) & (np.abs(yy - (cy + 22)) < 4)
        img[disc] -= 35
        process = (np.abs(xx - x0) < 7) & (np.abs(yy - cy) < 14)
        img[process] += 55
    if iliac:
        for side in (-1, 1):
            wing = ((xx - (cx + side * 105)) ** 2 / 55**2 + (yy - (h - 22)) ** 2 / 38**2) < 1
            img[wing] += 120
    out = _finish(img)
    if foreign:
        # Тонкая яркая дуга поверх мягких тканей — застёжка, цепочка, край одежды.
        # Добавляется ПОСЛЕ сглаживания: у металла край действительно резкий,
        # в отличие от анатомии.
        arc = (np.abs((yy - 34) - 0.018 * (xx - cx) ** 2) < 1.1) & (out > 0)
        out = np.where(arc, 245, out).astype(np.uint8)
    return out


def spine_truth(h: int = 290, w: int = 300, tilt: float = 0.0) -> dict:
    """Истинная геометрия фантома позвоночника — эталон для метрик локализации.

    Считается по тем же константам, что и сама картинка, поэтому эталон не может
    разойтись с изображением.

    `column` — маска позвоночного столба (то, что трекает `measure_spine`);
    `axis_x(y)` — истинная координата оси столба на высоте y.
    """
    cx = w / 2.0
    y_top = SPINE_Y0 - SPINE_HALF_H
    y_bot = SPINE_Y0 + (SPINE_N_VERTEBRAE - 1) * SPINE_DY + SPINE_HALF_H
    yy, xx = np.mgrid[0:h, 0:w]
    axis = cx + tilt * (yy - h / 2.0)
    column = (yy >= y_top) & (yy <= y_bot) & (np.abs(xx - axis) < SPINE_HALF_W)
    return {
        "column": column,
        "axis_x": lambda y: cx + tilt * (float(y) - h / 2.0),
        "y_top": float(y_top),
        "y_bot": float(y_bot),
        "shape": (h, w),
    }


def hip_truth(
    h: int = 290,
    w: int = 280,
    shaft_tilt: float = 0.10,
    lesser_troch: float = 0.45,
    crop_top: int = 0,
) -> dict:
    """Истинная геометрия фантома бедра — эталон для метрик локализации.

    `femur` — внешняя огибающая бедренной кости построчно (именно её отслеживает
    `measure_femur`: костномозговой канал темнее порога и делит диафиз надвое, поэтому
    сравнивать надо огибающую, а не «всю кость по порогу»). Седалищная кость в эталон
    не входит — она не часть бедра.
    """
    yy, xx = np.mgrid[0:h, 0:w]
    shaft_cx = HIP_SHAFT_X0 + shaft_tilt * (yy - h)
    bone = (np.abs(xx - shaft_cx) < HIP_SHAFT_HALF_W) & (yy > HIP_SHAFT_TOP_Y)
    bone |= (np.abs((yy - 118) + 0.55 * (xx - 150)) < 17) & (xx > 95) & (xx < 185)
    bone |= ((xx - 196) ** 2 / 34**2 + (yy - 78) ** 2 / 34**2) < 1
    bone |= ((xx - HIP_TROCH_X) ** 2 / 26**2 + (yy - HIP_TROCH_Y) ** 2 / 30**2) < 1
    lt_tip = None
    if lesser_troch > 0:
        rx = 13.0 * lesser_troch
        bone |= ((xx - (HIP_SHAFT_X0 + HIP_LT_DX)) ** 2 / rx**2 + (yy - HIP_LT_Y) ** 2 / 13**2) < 1
        lt_tip = (HIP_SHAFT_X0 + HIP_LT_DX + rx, float(HIP_LT_Y))

    # построчная огибающая: между самой левой и самой правой точкой кости в строке
    envelope = np.zeros_like(bone)
    for y in range(h):
        xs = np.flatnonzero(bone[y])
        if len(xs):
            envelope[y, xs[0] : xs[-1] + 1] = True

    if crop_top:
        envelope = envelope[crop_top:]
        lt_tip = None if lt_tip is None else (lt_tip[0], lt_tip[1] - crop_top)

    return {
        "femur": envelope,
        "shaft_x": lambda y: HIP_SHAFT_X0 + shaft_tilt * (float(y) + crop_top - h),
        "lesser_trochanter": lt_tip,
        "shape": envelope.shape,
    }


def _hip_image(
    rng: np.random.Generator,
    h: int = 290,
    w: int = 280,
    shaft_tilt: float = 0.10,
    lesser_troch: float = 0.45,
    crop_top: int = 0,
) -> np.ndarray:
    """Правое бедро: диафиз слева, головка и шейка вверх-вправо (как у ППОБ в эталоне)."""
    img = _soft_tissue(rng, h, w, margin_x=10, margin_y=4)
    yy, xx = np.mgrid[0:h, 0:w]
    shaft_x0 = HIP_SHAFT_X0
    shaft_cx = shaft_x0 + shaft_tilt * (yy - h)
    shaft = (np.abs(xx - shaft_cx) < HIP_SHAFT_HALF_W) & (yy > HIP_SHAFT_TOP_Y)
    cortex = shaft & (np.abs(xx - shaft_cx) > 13)  # канал темнее кортикального слоя
    img[shaft] += 60
    img[cortex] += 75
    neck = (np.abs((yy - 118) + 0.55 * (xx - 150)) < 17) & (xx > 95) & (xx < 185)
    img[neck] += 95
    head = ((xx - 196) ** 2 / 34**2 + (yy - 78) ** 2 / 34**2) < 1
    img[head] += 105
    great = ((xx - HIP_TROCH_X) ** 2 / 26**2 + (yy - HIP_TROCH_Y) ** 2 / 30**2) < 1
    img[great] += 90
    if lesser_troch > 0:  # бугорок на медиальной стороне сразу выше диафиза
        lt = ((xx - (shaft_x0 + HIP_LT_DX)) ** 2 / (13 * lesser_troch) ** 2 + (yy - HIP_LT_Y) ** 2 / 13**2) < 1
        img[lt] += 95
    ischium = ((xx - 250) ** 2 / 30**2 + (yy - 150) ** 2 / 48**2) < 1
    img[ischium] += 80
    out = _finish(img)
    if crop_top:
        out = out[crop_top:]
    return out


def _write(
    path: Path,
    *,
    study_key: str,
    image_key: str,
    body_part: str,
    pixels: np.ndarray | None,
    laterality: str = "",
) -> None:
    meta = FileMetaDataset()
    sop_uid = _uid(study_key, image_key)
    meta.MediaStorageSOPClassUID = SECONDARY_CAPTURE
    meta.MediaStorageSOPInstanceUID = sop_uid
    meta.TransferSyntaxUID = ExplicitVRLittleEndian

    ds = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0" * 128)
    ds.SOPClassUID = SECONDARY_CAPTURE
    ds.SOPInstanceUID = sop_uid
    ds.StudyInstanceUID = _uid(study_key)
    ds.SeriesInstanceUID = _uid(study_key, image_key, "series")
    ds.Modality = "BMD"
    ds.Manufacturer = "SYNTHETIC"
    ds.PatientName = "ANONYMOUS^DEMO"
    ds.PatientID = "DEMO"
    ds.StudyDate = "20260101"
    ds.BodyPartExamined = body_part
    if laterality:
        ds.ImageLaterality = laterality
    ds.StudyDescription = "Synthetic DXA demo"

    if pixels is not None:
        ds.Rows, ds.Columns = pixels.shape
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.BitsAllocated = 8
        ds.BitsStored = 8
        ds.HighBit = 7
        ds.PixelRepresentation = 0
        ds.PixelData = pixels.tobytes()

    path.parent.mkdir(parents=True, exist_ok=True)
    ds.save_as(path, enforce_file_format=True)


def generate(out: Path) -> list[Path]:
    rng = np.random.default_rng(7)
    out.mkdir(parents=True, exist_ok=True)
    files: list[Path] = []

    def add(rel: str, **kw) -> None:  # noqa: ANN003
        p = out / rel
        _write(p, **kw)
        files.append(p)

    add("study_spine_01/IM0001.dcm", study_key="spine01", image_key="1", body_part="LSPINE", pixels=_spine_image(rng))
    add(
        "study_spine_02_tilted/IM0001.dcm",
        study_key="spine02",
        image_key="1",
        body_part="LSPINE",
        pixels=_spine_image(rng, tilt=0.12, iliac=False),
    )
    add(
        "study_spine_03_foreign/IM0001.dcm",
        study_key="spine03",
        image_key="1",
        body_part="LSPINE",
        pixels=_spine_image(rng, foreign=True),
    )
    add(
        "study_hip_01/IM0001.dcm",
        study_key="hip01",
        image_key="R",
        body_part="HIP",
        laterality="R",
        pixels=_hip_image(rng),
    )
    add(
        "study_hip_01/IM0002.dcm",
        study_key="hip01",
        image_key="L",
        body_part="HIP",
        laterality="L",
        pixels=np.fliplr(_hip_image(rng)).copy(),
    )
    add(
        "study_hip_02_rotated/IM0001.dcm",
        study_key="hip02",
        image_key="R",
        body_part="HIP",
        laterality="R",
        pixels=_hip_image(rng, shaft_tilt=0.35, lesser_troch=1.6),
    )
    add(
        "study_hip_03_tight_roi/IM0001.dcm",
        study_key="hip03",
        image_key="R",
        body_part="HIP",
        laterality="R",
        pixels=_hip_image(rng, crop_top=70),
    )
    add(
        "study_combined/IM0001.dcm", study_key="comb01", image_key="spine", body_part="LSPINE", pixels=_spine_image(rng)
    )
    add(
        "study_combined/IM0002.dcm",
        study_key="comb01",
        image_key="hipR",
        body_part="HIP",
        laterality="R",
        pixels=_hip_image(rng),
    )

    add(
        "study_combined/IM0003.dcm",
        study_key="comb01",
        image_key="hipL",
        body_part="HIP",
        laterality="L",
        pixels=np.fliplr(_hip_image(rng)).copy(),
    )

    # Пограничные случаи — для проверки обработки ошибок, в демо-архив не кладутся
    add("edge_cases/no_pixel_data.dcm", study_key="edge01", image_key="nopix", body_part="LSPINE", pixels=None)
    (out / "edge_cases").mkdir(parents=True, exist_ok=True)
    (out / "edge_cases" / "not_a_dicom.dcm").write_bytes(b"not a dicom file at all\n" * 8)
    add(
        "edge_cases/tiny.dcm",
        study_key="edge02",
        image_key="tiny",
        body_part="HIP",
        pixels=np.zeros((24, 24), dtype=np.uint8),
    )

    archive = out / "demo_studies.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in files:
            if "edge_cases" not in f.parts:
                zf.write(f, f.relative_to(out).as_posix())
    return [*files, archive]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("samples"))
    args = parser.parse_args()
    for f in generate(args.out):
        print(f)


if __name__ == "__main__":
    main()
