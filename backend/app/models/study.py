"""ORM models: Study -> StudyImage -> ImageResult."""

import enum
import uuid
from datetime import UTC, datetime

from sqlalchemy import JSON, Boolean, DateTime, Enum, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.session import Base


def utcnow() -> datetime:
    return datetime.now(UTC)


def new_id() -> str:
    return uuid.uuid4().hex


class StudyStatus(enum.StrEnum):
    uploaded = "uploaded"
    queued = "queued"
    processing = "processing"
    completed = "completed"
    failed = "failed"


ACTIVE_STATUSES = (StudyStatus.queued, StudyStatus.processing)

# Состояния проверки специалистом (ТЗ: «автоматическая коррекция разметки с
# подтверждением специалиста»).
REVIEW_NONE = ""  # врач ещё не смотрел
REVIEW_CONFIRMED = "confirmed"  # врач согласен с автоматическим вердиктом
REVIEW_CORRECTED = "corrected"  # врач исправил вердикт
REVIEW_STATUSES = (REVIEW_NONE, REVIEW_CONFIRMED, REVIEW_CORRECTED)


class Study(Base):
    __tablename__ = "studies"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    name: Mapped[str] = mapped_column(String(255))
    source_path: Mapped[str] = mapped_column(String(1024), default="")
    storage_dir: Mapped[str] = mapped_column(String(1024))  # relative to DATA_DIR

    # Non-personal DICOM metadata only
    study_instance_uid: Mapped[str | None] = mapped_column(String(128), index=True)
    modality: Mapped[str | None] = mapped_column(String(32))
    manufacturer: Mapped[str | None] = mapped_column(String(128))

    status: Mapped[StudyStatus] = mapped_column(Enum(StudyStatus), default=StudyStatus.uploaded, index=True)
    progress: Mapped[int] = mapped_column(Integer, default=0)
    error_message: Mapped[str | None] = mapped_column(Text)
    warnings: Mapped[list[str]] = mapped_column(JSON, default=list)

    processor_name: Mapped[str | None] = mapped_column(String(64))
    processor_version: Mapped[str | None] = mapped_column(String(64))
    is_mock: Mapped[bool] = mapped_column(Boolean, default=False)
    processing_time_sec: Mapped[float | None] = mapped_column(Float)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)
    queued_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    images: Mapped[list["StudyImage"]] = relationship(
        back_populates="study", cascade="all, delete-orphan", order_by="StudyImage.original_filename"
    )
    results: Mapped[list["ImageResult"]] = relationship(
        back_populates="study", cascade="all, delete-orphan", order_by="ImageResult.id"
    )


class StudyImage(Base):
    __tablename__ = "study_images"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    study_id: Mapped[str] = mapped_column(ForeignKey("studies.id", ondelete="CASCADE"), index=True)
    original_filename: Mapped[str] = mapped_column(String(1024))
    stored_path: Mapped[str] = mapped_column(String(1024))  # relative to DATA_DIR
    size_bytes: Mapped[int] = mapped_column(Integer)

    sop_instance_uid: Mapped[str | None] = mapped_column(String(128))
    series_instance_uid: Mapped[str | None] = mapped_column(String(128))
    modality: Mapped[str | None] = mapped_column(String(32))
    body_part_examined: Mapped[str | None] = mapped_column(String(64))
    rows: Mapped[int | None] = mapped_column(Integer)
    columns: Mapped[int | None] = mapped_column(Integer)
    has_pixel_data: Mapped[bool] = mapped_column(Boolean, default=False)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    study: Mapped[Study] = relationship(back_populates="images")


class ImageResult(Base):
    """One row per image / anatomical region — mirrors the required CSV/XLSX output."""

    __tablename__ = "image_results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    study_id: Mapped[str] = mapped_column(ForeignKey("studies.id", ondelete="CASCADE"), index=True)
    image_id: Mapped[str | None] = mapped_column(ForeignKey("study_images.id", ondelete="CASCADE"))

    path_to_study: Mapped[str] = mapped_column(String(1024))
    study_uid: Mapped[str | None] = mapped_column(String(128))
    image_uid: Mapped[str | None] = mapped_column(String(128))
    anatomical_region: Mapped[str | None] = mapped_column(String(64))
    quality_class: Mapped[str | None] = mapped_column(String(64))
    violation_type: Mapped[str | None] = mapped_column(String(512))
    processing_status: Mapped[str] = mapped_column(String(32))
    time_of_processing: Mapped[float] = mapped_column(Float)

    # Extra fields for explainability (not part of the mandatory export)
    confidence: Mapped[float | None] = mapped_column(Float)
    details: Mapped[dict] = mapped_column(JSON, default=dict)
    error_message: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    # --- проверка специалистом --------------------------------------------
    # Автоматический вердикт выше НИКОГДА не переписывается: врач заполняет
    # отдельные поля. Так в любой момент видно, что предложил сервис и что решил
    # человек, и по этим же полям считается, как часто сервис ошибается.
    review_status: Mapped[str] = mapped_column(String(16), default=REVIEW_NONE)
    reviewed_quality_class: Mapped[str | None] = mapped_column(String(8))
    reviewed_violation_type: Mapped[str | None] = mapped_column(String(512))
    reviewed_by: Mapped[str | None] = mapped_column(String(128))
    review_comment: Mapped[str | None] = mapped_column(Text)
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    study: Mapped[Study] = relationship(back_populates="results")

    @property
    def final_quality_class(self) -> str | None:
        """Что считать итогом: решение врача, если он его вынес, иначе автоматическое."""
        if self.review_status == REVIEW_CORRECTED:
            return self.reviewed_quality_class
        return self.quality_class

    @property
    def final_violation_type(self) -> str | None:
        if self.review_status == REVIEW_CORRECTED:
            return self.reviewed_violation_type or ""
        return self.violation_type
