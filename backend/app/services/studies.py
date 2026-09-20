"""Study-level business logic shared by API routes."""

from __future__ import annotations

import shutil

from fastapi import HTTPException, status
from sqlalchemy import func, select
from sqlalchemy.orm import Session, selectinload

from app.core.config import Settings
from app.models import ACTIVE_STATUSES, ImageResult, Study, StudyImage, StudyStatus
from app.models.study import utcnow
from app.schemas.study import (
    ImageOut,
    QualitySummary,
    ResultRow,
    StudyDetail,
    StudyResultOut,
    StudyStatusOut,
    StudySummary,
)
from app.services.task_runner import TaskRunner

# «1» — значение quality_class по ТЗ; остальные оставлены для mock-процессора
QUALITY_BAD_VALUES = {"1", "unacceptable", "bad", "poor"}


def get_study_or_404(db: Session, study_id: str) -> Study:
    study = db.get(Study, study_id, options=[selectinload(Study.images), selectinload(Study.results)])
    if study is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Исследование {study_id} не найдено")
    return study


def build_summary(results: list[ImageResult]) -> QualitySummary | None:
    if not results:
        return None
    s = QualitySummary(total=len(results))
    regions: list[str] = []
    for r in results:
        if r.processing_status == "success":
            s.success += 1
            if (r.quality_class or "").lower() in QUALITY_BAD_VALUES:
                s.unacceptable += 1
            else:
                s.acceptable += 1
            if r.anatomical_region and r.anatomical_region not in regions:
                regions.append(r.anatomical_region)
        else:
            s.errors += 1
    s.regions = regions
    if s.success:
        s.overall_quality = "unacceptable" if s.unacceptable else "acceptable"
    return s


def to_summary(study: Study) -> StudySummary:
    return StudySummary.model_validate(
        {
            **{k: getattr(study, k) for k in StudySummary.model_fields if hasattr(study, k)},
            "image_count": len(study.images),
            "summary": build_summary(study.results),
        }
    )


def preview_url(study_id: str, image: StudyImage) -> str | None:
    if not image.has_pixel_data:
        return None
    return f"/api/v1/studies/{study_id}/images/{image.id}/preview"


def to_detail(study: Study) -> StudyDetail:
    base = to_summary(study).model_dump()
    images = [
        ImageOut.model_validate(i).model_copy(update={"preview_url": preview_url(study.id, i)}) for i in study.images
    ]
    return StudyDetail.model_validate(
        {
            **base,
            "warnings": study.warnings or [],
            "processor_name": study.processor_name,
            "processor_version": study.processor_version,
            "queued_at": study.queued_at,
            "started_at": study.started_at,
            "created_at": study.created_at,
            "updated_at": study.updated_at,
            "finished_at": study.finished_at,
            "images": images,
        }
    )


def to_status(study: Study) -> StudyStatusOut:
    return StudyStatusOut(
        study_id=study.id,
        status=study.status,
        progress=study.progress,
        processed_images=len(study.results),
        total_images=len(study.images),
        error_message=study.error_message,
        queued_at=study.queued_at,
        started_at=study.started_at,
        finished_at=study.finished_at,
        processing_time_sec=study.processing_time_sec,
    )


def to_result(study: Study) -> StudyResultOut:
    names = {i.id: i.original_filename for i in study.images}
    rows = [
        ResultRow.model_validate(r).model_copy(update={"original_filename": names.get(r.image_id or "")})
        for r in study.results
    ]
    return StudyResultOut(
        study_id=study.id,
        status=study.status,
        processor=study.processor_name,
        processor_version=study.processor_version,
        is_mock=study.is_mock,
        processing_time_sec=study.processing_time_sec,
        finished_at=study.finished_at,
        summary=build_summary(study.results) or QualitySummary(),
        rows=rows,
    )


def ensure_results(study: Study) -> None:
    if study.status in ACTIVE_STATUSES:
        raise HTTPException(status.HTTP_409_CONFLICT, "Обработка ещё не завершена")
    if not study.results:
        raise HTTPException(status.HTTP_409_CONFLICT, "Результатов нет: исследование ещё не обрабатывалось")


def enqueue(db: Session, study: Study, runner: TaskRunner) -> None:
    if study.status in ACTIVE_STATUSES:
        raise HTTPException(status.HTTP_409_CONFLICT, f"Исследование уже в обработке ({study.status.value})")
    study.status = StudyStatus.queued
    study.queued_at = utcnow()
    study.started_at = None
    study.finished_at = None
    study.progress = 0
    study.error_message = None
    db.commit()
    runner.submit(study.id)


def list_studies(db: Session, status_filter: StudyStatus | None, limit: int, offset: int) -> tuple[list[Study], int]:
    q = select(Study)
    cq = select(func.count(Study.id))
    if status_filter:
        q = q.where(Study.status == status_filter)
        cq = cq.where(Study.status == status_filter)
    q = (
        q.options(selectinload(Study.images), selectinload(Study.results))
        .order_by(Study.created_at.desc())
        .limit(limit)
        .offset(offset)
    )
    return list(db.scalars(q).all()), int(db.scalar(cq) or 0)


def delete_study(db: Session, study: Study, settings: Settings) -> None:
    if study.status == StudyStatus.processing:
        raise HTTPException(status.HTTP_409_CONFLICT, "Нельзя удалить исследование во время обработки")
    storage = settings.data_dir / study.storage_dir
    previews = settings.data_dir / "previews" / study.id
    db.delete(study)
    db.commit()
    shutil.rmtree(storage, ignore_errors=True)
    shutil.rmtree(previews, ignore_errors=True)
