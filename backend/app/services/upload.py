"""Upload handling: stream files to disk, unpack ZIP archives, validate DICOM, group by StudyInstanceUID."""

from __future__ import annotations

import logging
import posixpath
import shutil
import uuid
import zipfile
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path

from fastapi import UploadFile
from fastapi.concurrency import run_in_threadpool
from sqlalchemy.orm import Session

from app.core.config import Settings
from app.models import Study, StudyImage
from app.models.study import new_id
from app.services.dicom import DicomMeta, DicomValidationError, read_metadata

logger = logging.getLogger(__name__)

CHUNK = 1024 * 1024
ZIP_MAGIC = b"PK\x03\x04"
SKIP_NAMES = {".ds_store", "thumbs.db", "desktop.ini"}


class UploadLimitError(Exception):
    pass


@dataclass
class _Candidate:
    original_name: str  # relative path as provided by the client / inside archive
    path: Path
    size: int


@dataclass
class _Valid:
    cand: _Candidate
    meta: DicomMeta


@dataclass
class UploadOutcome:
    studies: list[Study] = field(default_factory=list)
    rejected: list[dict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _clean_name(name: str | None) -> str:
    name = (name or "file").replace("\\", "/").lstrip("/")
    parts = [p for p in name.split("/") if p not in ("", ".", "..")]
    return "/".join(parts) or "file"


def _is_zip(path: Path, name: str) -> bool:
    if name.lower().endswith(".zip"):
        return True
    with path.open("rb") as fh:
        return fh.read(4) == ZIP_MAGIC


def _should_skip(name: str) -> bool:
    base = posixpath.basename(name).lower()
    return (
        base in SKIP_NAMES
        or base.startswith("._")
        or "__macosx/" in name.lower()
        or base.endswith((".txt", ".json", ".xml", ".csv", ".xlsx", ".pdf", ".jpg", ".png"))
    )


class UploadService:
    def __init__(self, db: Session, settings: Settings) -> None:
        self.db = db
        self.settings = settings
        self._total_bytes = 0
        self._created_dirs: list[Path] = []

    async def _save_stream(self, upload: UploadFile, dest: Path) -> int:
        size = 0
        with dest.open("wb") as out:
            while chunk := await upload.read(CHUNK):
                size += len(chunk)
                self._total_bytes += len(chunk)
                if self._total_bytes > self.settings.max_upload_size_bytes:
                    raise UploadLimitError(f"Превышен лимит размера загрузки ({self.settings.max_upload_size_mb} МБ)")
                out.write(chunk)
        return size

    def _extract_zip(self, zpath: Path, zname: str, workdir: Path, out: UploadOutcome) -> list[_Candidate]:
        result: list[_Candidate] = []
        try:
            with zipfile.ZipFile(zpath) as zf:
                for info in zf.infolist():
                    if info.is_dir():
                        continue
                    inner = _clean_name(info.filename)
                    display = f"{zname}/{inner}"
                    if _should_skip(inner):
                        continue
                    self._total_bytes += info.file_size
                    if self._total_bytes > self.settings.max_upload_size_bytes:
                        raise UploadLimitError(
                            f"Превышен лимит размера после распаковки ({self.settings.max_upload_size_mb} МБ)"
                        )
                    dest = workdir / uuid.uuid4().hex
                    with zf.open(info) as src, dest.open("wb") as dst:
                        shutil.copyfileobj(src, dst, CHUNK)
                    result.append(_Candidate(display, dest, info.file_size))
        except zipfile.BadZipFile:
            out.rejected.append({"filename": zname, "reason": "Повреждённый ZIP-архив"})
        return result

    async def handle(self, files: list[UploadFile]) -> UploadOutcome:
        out = UploadOutcome()
        if len(files) > self.settings.max_files_per_upload:
            raise UploadLimitError(f"Слишком много файлов за раз (максимум {self.settings.max_files_per_upload})")

        workdir = self.settings.uploads_dir / "_incoming" / uuid.uuid4().hex
        workdir.mkdir(parents=True, exist_ok=True)
        try:
            candidates: list[_Candidate] = []
            zips: list[tuple[Path, str]] = []
            for up in files:
                name = _clean_name(up.filename)
                tmp = workdir / uuid.uuid4().hex
                size = await self._save_stream(up, tmp)
                if size == 0:
                    out.rejected.append({"filename": name, "reason": "Пустой файл"})
                elif _is_zip(tmp, name):
                    zips.append((tmp, name))
                elif _should_skip(name):
                    out.rejected.append({"filename": name, "reason": "Не DICOM (пропущен по типу файла)"})
                else:
                    candidates.append(_Candidate(name, tmp, size))

            # CPU/disk-bound part runs in a worker thread to keep the event loop responsive
            await run_in_threadpool(self._finalize, candidates, zips, workdir, out)
            return out
        except Exception:
            self.db.rollback()
            for d in self._created_dirs:
                shutil.rmtree(d, ignore_errors=True)
            raise
        finally:
            shutil.rmtree(workdir, ignore_errors=True)

    def _finalize(
        self, candidates: list[_Candidate], zips: list[tuple[Path, str]], workdir: Path, out: UploadOutcome
    ) -> None:
        for zpath, zname in zips:
            candidates.extend(self._extract_zip(zpath, zname, workdir, out))
            zpath.unlink(missing_ok=True)

        valid: list[_Valid] = []
        for c in candidates:
            try:
                valid.append(_Valid(c, read_metadata(c.path)))
            except DicomValidationError as exc:
                out.rejected.append({"filename": c.original_name, "reason": str(exc)})

        groups: OrderedDict[str, list[_Valid]] = OrderedDict()
        for v in valid:
            key = v.meta.study_instance_uid or f"__single__{v.cand.original_name}"
            groups.setdefault(key, []).append(v)

        for items in groups.values():
            study = self._create_study(items, out)
            if study is not None:
                out.studies.append(study)

        self.db.commit()
        for s in out.studies:
            self.db.refresh(s)

    def _create_study(self, items: list[_Valid], out: UploadOutcome) -> Study | None:
        seen: set[str] = set()
        unique: list[_Valid] = []
        for v in items:
            uid = v.meta.sop_instance_uid
            if uid and uid in seen:
                out.rejected.append({"filename": v.cand.original_name, "reason": "Дубликат SOPInstanceUID"})
                continue
            if uid:
                seen.add(uid)
            unique.append(v)
        if not unique:
            return None

        first = unique[0]
        study_id = new_id()
        rel_dir = Path("uploads") / study_id
        abs_dir = self.settings.data_dir / rel_dir

        abs_dir.mkdir(parents=True, exist_ok=True)
        self._created_dirs.append(abs_dir)

        dirs = {posixpath.dirname(v.cand.original_name) for v in unique}
        common_dir = next(iter(dirs)) if len(dirs) == 1 else ""
        source_path = common_dir or first.cand.original_name
        first_base = posixpath.basename(first.cand.original_name)
        if common_dir and not common_dir.lower().endswith(".zip"):
            name = posixpath.basename(common_dir)
        elif len(unique) == 1:
            name = first_base
        else:
            name = f"{first_base} (+{len(unique) - 1})"

        warnings: list[str] = []
        if len(unique) > self.settings.max_images_per_study:
            warnings.append(
                f"В исследовании {len(unique)} изображений — больше ожидаемых "
                f"{self.settings.max_images_per_study}. Все будут обработаны."
            )
        if not first.meta.study_instance_uid:
            warnings.append("Нет StudyInstanceUID — файл загружен как отдельное исследование.")

        study = Study(
            id=study_id,
            name=name[:255],
            source_path=source_path,
            storage_dir=str(rel_dir),
            study_instance_uid=first.meta.study_instance_uid,
            modality=first.meta.modality,
            manufacturer=first.meta.manufacturer,
            warnings=warnings,
        )
        self.db.add(study)

        for v in unique:
            image_id = new_id()
            rel_path = rel_dir / f"{image_id}.dcm"
            shutil.move(str(v.cand.path), self.settings.data_dir / rel_path)
            m = v.meta
            study.images.append(
                StudyImage(
                    id=image_id,
                    original_filename=v.cand.original_name[:1024],
                    stored_path=str(rel_path),
                    size_bytes=v.cand.size,
                    sop_instance_uid=m.sop_instance_uid,
                    series_instance_uid=m.series_instance_uid,
                    modality=m.modality,
                    body_part_examined=m.body_part_examined,
                    rows=m.rows,
                    columns=m.columns,
                    has_pixel_data=m.has_pixel_data,
                )
            )
        out.warnings.extend(f"{name}: {w}" for w in warnings)
        return study
