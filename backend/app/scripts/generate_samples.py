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


def _uid(*parts: str) -> str:
    return generate_uid(entropy_srcs=["densitometry-demo", *parts])


def _spine_image(rng: np.random.Generator, h: int = 384, w: int = 256, tilt: float = 0.0) -> np.ndarray:
    img = rng.normal(600, 40, (h, w))
    yy, xx = np.mgrid[0:h, 0:w]
    cx = w / 2
    for i in range(6):  # Th12 + L1..L5
        cy = 40 + i * 58
        x0 = cx + tilt * (cy - h / 2)
        mask = (np.abs(xx - x0) < 34) & (np.abs(yy - cy) < 22)
        img[mask] += 1800
    # iliac crests
    for side in (-1, 1):
        mask = ((xx - (cx + side * 95)) ** 2 / 45**2 + (yy - 360) ** 2 / 60**2) < 1
        img[mask] += 1200
    return np.clip(img, 0, 4095).astype(np.uint16)


def _hip_image(rng: np.random.Generator, h: int = 384, w: int = 320) -> np.ndarray:
    img = rng.normal(600, 40, (h, w))
    yy, xx = np.mgrid[0:h, 0:w]
    shaft = (np.abs(xx - 200 - (yy - 380) * 0.1) < 30) & (yy > 170)
    neck = (np.abs((yy - 150) - (xx - 150) * 0.6) < 22) & (xx > 110) & (xx < 220)
    head = ((xx - 105) ** 2 + (yy - 125) ** 2) < 42**2
    trochanter = ((xx - 235) ** 2 / 30**2 + (yy - 150) ** 2 / 38**2) < 1
    ischium = ((xx - 60) ** 2 / 28**2 + (yy - 300) ** 2 / 45**2) < 1
    for m, v in ((shaft, 2000), (neck, 1600), (head, 1900), (trochanter, 1500), (ischium, 1100)):
        img[m] += v
    return np.clip(img, 0, 4095).astype(np.uint16)


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
        ds.BitsAllocated = 16
        ds.BitsStored = 12
        ds.HighBit = 11
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
        pixels=_spine_image(rng, tilt=0.15),
    )
    add(
        "study_hip_01/IM0001.dcm",
        study_key="hip01",
        image_key="L",
        body_part="HIP",
        laterality="L",
        pixels=_hip_image(rng),
    )
    add(
        "study_hip_01/IM0002.dcm",
        study_key="hip01",
        image_key="R",
        body_part="HIP",
        laterality="R",
        pixels=np.fliplr(_hip_image(rng)).copy(),
    )
    add(
        "study_combined/IM0001.dcm", study_key="comb01", image_key="spine", body_part="LSPINE", pixels=_spine_image(rng)
    )
    add(
        "study_combined/IM0002.dcm",
        study_key="comb01",
        image_key="hipL",
        body_part="HIP",
        laterality="L",
        pixels=_hip_image(rng),
    )
    add(
        "study_combined/IM0003.dcm",
        study_key="comb01",
        image_key="hipR",
        body_part="HIP",
        laterality="R",
        pixels=np.fliplr(_hip_image(rng)).copy(),
    )
    # Edge cases
    add("edge_cases/no_pixel_data.dcm", study_key="nopix", image_key="1", body_part="LSPINE", pixels=None)
    bad = out / "edge_cases/not_a_dicom.dcm"
    bad.write_text("this is not a DICOM file\n")
    files.append(bad)

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
