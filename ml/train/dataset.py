"""Torch-датасет поверх dataset.csv, собранного `dxa.build_dataset`."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from dxa.labels import REGION_FEMUR
from train.tasks import targets_for


def to_tensor(a: np.ndarray, norm: str = "imagenet") -> torch.Tensor:
    """Кадр со значениями [0, 1] (H×W) -> вход сети в той нормировке, в которой учился бэкбон.

    Одна функция и для обучения, и для инференса (`train.predict`): если нормировка
    разойдётся, модель будет получать в бою не то, на чём училась, и ошибка будет тихой.
    """
    x = torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32))[None]
    if norm == "xrv":
        return (2.0 * x - 1.0) * 1024.0  # как xrv.utils.normalize: [-1024, 1024], один канал
    if norm == "imagenet":
        return ((x - 0.449) / 0.226).repeat(3, 1, 1)  # ImageNet-бэкбон ждёт 3 канала
    raise ValueError(f"неизвестная нормировка: {norm}")


@dataclass
class Sample:
    image_id: str
    study_dir: str
    png: Path
    region: str
    side: str
    quality_class: int
    violations: list[str]
    violations_known: bool  # False => головы видов нарушений маскируются (правило R3)
    fold: int


def read_dataset(csv_path: str | Path, root: str | Path | None = None) -> list[Sample]:
    """Читает dataset.csv. Всё, что проверка разметки признала негодным, туда попадает
    с пустым quality_class и fold = -1, поэтому достаточно одного фильтра."""
    csv_path = Path(csv_path)
    root = Path(root) if root else csv_path.parent
    out: list[Sample] = []
    with csv_path.open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["quality_class"] == "" or int(r["fold"]) < 0:
                continue
            out.append(
                Sample(
                    image_id=r["image_id"],
                    study_dir=r["study_dir"],
                    png=root / r["png_path"],
                    region=r["region"],
                    side=r["side"],
                    quality_class=int(r["quality_class"]),
                    violations=[v for v in r["violation_type"].split(";") if v],
                    violations_known=r.get("violations_known", "1") != "0",
                    fold=int(r["fold"]),
                )
            )
    return out


class DxaDataset(Dataset):
    """Аугментации намеренно консервативные.

    Повороты меняют смысл метки «не выравнена ось позвоночника» (порог 5°), поэтому для
    позвоночника вращение отключено, а для бедра ограничено малым углом. Горизонтальное
    отражение снимка бедра превращает левое бедро в правое — анатомически это корректная
    аугментация, но по умолчанию она выключена (флаг `hflip_femur`).
    """

    def __init__(
        self,
        samples: list[Sample],
        size: tuple[int, int] = (384, 320),
        train: bool = False,
        hflip_femur: bool = False,
        max_shift: float = 0.06,
        max_scale: float = 0.08,
        femur_max_rot_deg: float = 3.0,
        brightness: float = 0.15,
        norm: str = "imagenet",
    ) -> None:
        self.samples = samples
        self.norm = norm
        self.size = size
        self.train = train
        self.hflip_femur = hflip_femur
        self.max_shift = max_shift
        self.max_scale = max_scale
        self.femur_max_rot = femur_max_rot_deg
        self.brightness = brightness

    def __len__(self) -> int:
        return len(self.samples)

    def _load(self, s: Sample) -> np.ndarray:
        img = Image.open(s.png).convert("L")
        if self.train:
            rng = np.random
            if s.region == REGION_FEMUR and self.femur_max_rot > 0:
                img = img.rotate(
                    float(rng.uniform(-self.femur_max_rot, self.femur_max_rot)),
                    resample=Image.BILINEAR,
                )
            if self.hflip_femur and s.region == REGION_FEMUR and rng.random() < 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            w, h = img.size
            sc = 1.0 + float(rng.uniform(-self.max_scale, self.max_scale))
            dx = int(w * float(rng.uniform(-self.max_shift, self.max_shift)))
            dy = int(h * float(rng.uniform(-self.max_shift, self.max_shift)))
            img = img.transform(
                (w, h),
                Image.AFFINE,
                (1 / sc, 0, dx, 0, 1 / sc, dy),
                resample=Image.BILINEAR,
                fillcolor=0,
            )
        img = img.resize((self.size[1], self.size[0]), Image.BILINEAR)
        a = np.asarray(img, dtype=np.float32) / 255.0
        if self.train and self.brightness:
            a = np.clip(
                a * (1.0 + float(np.random.uniform(-self.brightness, self.brightness))),
                0,
                1,
            )
        return a

    def __getitem__(self, i: int):  # noqa: ANN204
        s = self.samples[i]
        x = to_tensor(self._load(s), self.norm)
        t, m = targets_for(s.region, s.quality_class, s.violations, s.violations_known)
        return x, torch.tensor(t), torch.tensor(m), i
