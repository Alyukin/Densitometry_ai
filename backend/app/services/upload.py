"""Upload handling: stream files to disk, unpack ZIP archives, validate DICOM, group by StudyInstanceUID.

Файл, который не открывается или не разбирается как DICOM, не отбрасывается: по
определению заказчика это строка с processing_status = Failure. Такой файл
сохраняется как изображение с причиной (`invalid_reason`) и получает свою строку в
выгрузке. Пропускаются без строки только служебные файлы (DICOMDIR, .DS_Store,
__MACOSX) и документы рядом с данными (таблицы, PDF, текст) — это не снимки.
"""

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
from app.services.dicom import DicomMeta, DicomValidationError, NotAnImageFile, read_metadata

logger = logging.getLogger(__name__)

CHUNK = 1024 * 1024
ZIP_MAGIC = b"PK\x03\x04"
SKIP_NAMES = {".ds_store", "thumbs.db", "desktop.ini"}
# Документы, которые лежат рядом со снимками (разметка, отчёты), — не входные данные
DOCUMENT_SUFFIXES = (".txt", ".json", ".xml", ".csv", ".xlsx", ".xls", ".pdf", ".doc", ".docx", ".md", ".html", ".log")


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
class _Invalid:
    cand: _Candidate
    reason: str


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


def _is_junk(name: str) -> bool:
    """Служебные файлы ОС и архиваторов — пропускаются молча."""
    base = posixpath.basename(name).lower()
    return base in SKIP_NAMES or base.startswith("._") or "__macosx/" in name.lower()


def _is_document(name: str) -> bool:
    return posixpath.basename(name).lower().endswith(DOCUMENT_SUFFIXES)


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
                    if _is_junk(inner):
                        continue
                    if _is_document(inner):
                        out.rejected.append({"filename": display, "reason": "Документ, а не снимок: пропущен"})
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
                if _is_junk(name):
                    continue
                if size > 0 and _is_zip(tmp, name):
                    zips.append((tmp, name))
                elif _is_document(name):
                    out.rejected.append({"filename": name, "reason": "Документ, а не снимок: пропущен"})
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
        invalid: list[_Invalid] = []
        for c in candidates:
            if c.size == 0:
                invalid.append(_Invalid(c, "Пустой файл"))
                continue
            try:
                valid.append(_Valid(c, read_metadata(c.path)))
            except NotAnImageFile as exc:
                out.rejected.append({"filename": c.original_name, "reason": str(exc)})
            except DicomValidationError as exc:
                invalid.append(_Invalid(c, str(exc)))

        groups: OrderedDict[str, list[_Valid]] = OrderedDict()
        for v in valid:
            key = v.meta.study_instance_uid or f"__single__{v.cand.original_name}"
            groups.setdefault(key, []).append(v)

        # Нераспознанный файл кладётся к исследованию из той же папки, если оно там
        # одно; иначе становится отдельным исследованием. Строка Failure в выгрузке
        # будет в обоих случаях.
        by_dir: dict[str, set[str]] = {}
        for key, items in groups.items():
            for v in items:
                by_dir.setdefault(posixpath.dirname(v.cand.original_name), set()).add(key)
        attached: dict[str, list[_Invalid]] = {}
        standalone: list[_Invalid] = []
        for bad in invalid:
            keys = by_dir.get(posixpath.dirname(bad.cand.original_name), set())
            if posixpath.dirname(bad.cand.original_name) and len(keys) == 1:
                attached.setdefault(next(iter(keys)), []).append(bad)
            else:
                standalone.append(bad)
            out.warnings.append(
                f"{bad.cand.original_name}: {bad.reason} — в результатах будет строка со статусом Failure"
            )

        for key, items in groups.items():
            study = self._create_study(items, out, attached.get(key, []))
            if study is not None:
                out.studies.append(study)
        for bad in standalone:
            out.studies.append(self._create_invalid_study(bad))

        self.db.commit()
        for s in out.studies:
            self.db.refresh(s)

    def _new_study_dir(self) -> tuple[str, Path, Path]:
        study_id = new_id()
        rel_dir = Path("uploads") / study_id
        abs_dir = self.settings.data_dir / rel_dir
        abs_dir.mkdir(parents=True, exist_ok=True)
        self._created_dirs.append(abs_dir)
        return study_id, rel_dir, abs_dir

    def _invalid_image(self, bad: _Invalid, rel_dir: Path) -> StudyImage:
        image_id = new_id()
        rel_path = rel_dir / f"{image_id}.bin"
        shutil.move(str(bad.cand.path), self.settings.data_dir / rel_path)
        return StudyImage(
            id=image_id,
            original_filename=bad.cand.original_name[:1024],
            stored_path=str(rel_path),
            size_bytes=bad.cand.size,
            has_pixel_data=False,
            invalid_reason=bad.reason[:255],
        )

    def _create_invalid_study(self, bad: _Invalid) -> Study:
        """Отдельное «исследование» из одного нераспознанного файла — ради строки Failure."""
        study_id, rel_dir, _ = self._new_study_dir()
        name = posixpath.basename(bad.cand.original_name) or bad.cand.original_name
        study = Study(
            id=study_id,
            name=name[:255],
            source_path=posixpath.dirname(bad.cand.original_name) or bad.cand.original_name,
            storage_dir=str(rel_dir),
            warnings=[f"{bad.reason}: в результатах будет строка с processing_status = Failure"],
        )
        self.db.add(study)
        study.images.append(self._invalid_image(bad, rel_dir))
        return study

    def _create_study(
        self, items: list[_Valid], out: UploadOutcome, invalid: list[_Invalid] | None = None
    ) -> Study | None:
        invalid = invalid or []
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
            out.studies.extend(self._create_invalid_study(bad) for bad in invalid)
            return None

        first = unique[0]
        study_id, rel_dir, _ = self._new_study_dir()

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
        for bad in invalid:
            study.images.append(self._invalid_image(bad, rel_dir))
        out.warnings.extend(f"{name}: {w}" for w in warnings)
        return study
