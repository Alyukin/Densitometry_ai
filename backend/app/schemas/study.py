"""Pydantic API schemas."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, PlainSerializer

from app.models import StudyStatus


def _utc(dt: datetime | None) -> str | None:
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.isoformat()


UtcDatetime = Annotated[datetime, PlainSerializer(_utc, return_type=str, when_used="json")]


class ORMModel(BaseModel):
    model_config = ConfigDict(from_attributes=True)


# --- Health ---
class HealthOut(BaseModel):
    status: Literal["ok", "degraded"]
    version: str
    environment: str
    processor: str
    processor_version: str | None = None
    is_mock: bool | None = None
    database: Literal["ok", "error"]
    workers: int
    tasks_in_progress: int


# --- Images ---
class ImageOut(ORMModel):
    id: str
    original_filename: str
    size_bytes: int
    sop_instance_uid: str | None
    series_instance_uid: str | None
    modality: str | None
    body_part_examined: str | None
    rows: int | None
    columns: int | None
    has_pixel_data: bool
    preview_url: str | None = None


# --- Studies ---
class QualitySummary(BaseModel):
    total: int = 0
    success: int = 0
    errors: int = 0
    acceptable: int = 0
    unacceptable: int = 0
    non_standard: int = Field(default=0, description="Обработано, но это не снимок позвоночника или бедра")
    regions: list[str] = Field(default_factory=list)
    overall_quality: str | None = Field(
        default=None, description="unacceptable, если хотя бы одно изображение с нарушением"
    )


class StudySummary(ORMModel):
    id: str
    name: str
    source_path: str
    study_instance_uid: str | None
    modality: str | None
    manufacturer: str | None
    status: StudyStatus
    progress: int
    image_count: int
    is_mock: bool
    processing_time_sec: float | None
    error_message: str | None
    created_at: UtcDatetime
    updated_at: UtcDatetime
    finished_at: UtcDatetime | None
    summary: QualitySummary | None = None


class StudyDetail(StudySummary):
    warnings: list[str]
    processor_name: str | None
    processor_version: str | None
    queued_at: UtcDatetime | None
    started_at: UtcDatetime | None
    images: list[ImageOut]


class StudyList(BaseModel):
    items: list[StudySummary]
    total: int
    limit: int
    offset: int


class RejectedFile(BaseModel):
    filename: str
    reason: str


class UploadResponse(BaseModel):
    studies: list[StudySummary]
    rejected: list[RejectedFile]
    warnings: list[str]


class StudyStatusOut(BaseModel):
    study_id: str
    status: StudyStatus
    progress: int = Field(ge=0, le=100)
    processed_images: int
    total_images: int
    error_message: str | None
    queued_at: UtcDatetime | None
    started_at: UtcDatetime | None
    finished_at: UtcDatetime | None
    processing_time_sec: float | None


class ProcessResponse(BaseModel):
    study_id: str
    status: StudyStatus
    message: str


# --- Results ---
class ResultRow(ORMModel):
    """Строка результата. Первые 8 полей совпадают с колонками CSV/XLSX из ТЗ."""

    path_to_study: str
    study_uid: str | None
    image_uid: str | None
    anatomical_region: str | None
    quality_class: str | None
    violation_type: str | None
    processing_status: str
    time_of_processing: float
    # extra
    image_id: str | None
    original_filename: str | None = None
    confidence: float | None
    error_message: str | None
    details: dict[str, Any]
    # проверка специалистом: автоматический вердикт выше остаётся нетронутым
    review_status: str = ""
    reviewed_quality_class: str | None = None
    reviewed_violation_type: str | None = None
    reviewed_by: str | None = None
    review_comment: str | None = None
    reviewed_at: UtcDatetime | None = None


# --- Проверка специалистом ---
class ReviewRequest(BaseModel):
    """Подтверждение или исправление вердикта (ТЗ: коррекция с подтверждением специалиста)."""

    action: Literal["confirm", "correct", "reset"] = Field(
        description="confirm — согласиться с сервисом, correct — заменить вердикт, reset — снять проверку"
    )
    violation_type: list[str] = Field(
        default_factory=list,
        description="Только для correct: нарушения из закрытого списка своей области. Пустой список — «годно».",
    )
    reviewed_by: str | None = Field(default=None, max_length=128, description="Кто проверил")
    comment: str | None = Field(default=None, max_length=2000, description="Комментарий специалиста")

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {"action": "confirm", "reviewed_by": "Иванов И.И."},
                {
                    "action": "correct",
                    "violation_type": ["Не выравнена ось позвоночника"],
                    "reviewed_by": "Иванов И.И.",
                    "comment": "сколиоз, ось не выровнена",
                },
            ]
        }
    )


class ReviewOut(BaseModel):
    study_id: str
    row: ResultRow
    allowed_violations: list[str] = Field(description="Закрытый список нарушений для области этого изображения")
    agreement: dict[str, Any] = Field(description="Насколько сервис сходится с врачом по этому исследованию")


class StudyResultOut(BaseModel):
    study_id: str
    status: StudyStatus
    processor: str | None
    processor_version: str | None
    is_mock: bool
    processing_time_sec: float | None
    finished_at: UtcDatetime | None
    summary: QualitySummary
    rows: list[ResultRow]


# --- Batch ---
class BatchProcessRequest(BaseModel):
    study_ids: list[str] = Field(default_factory=list, description="ID исследований для обработки")
    all_pending: bool = Field(default=False, description="Добавить все исследования со статусом uploaded/failed")
    force: bool = Field(default=False, description="Переобработать уже завершённые исследования")

    model_config = ConfigDict(
        json_schema_extra={"examples": [{"study_ids": ["<id1>", "<id2>"]}, {"all_pending": True}]}
    )


class BatchSkipped(BaseModel):
    study_id: str
    reason: str


class BatchProcessResponse(BaseModel):
    accepted: list[str]
    skipped: list[BatchSkipped]
