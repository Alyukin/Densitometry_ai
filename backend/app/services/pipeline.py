"""Study processing pipeline — independent from the concrete processor (mock / AI).

Перед процессором каждый файл проходит приём (`app.processing.intake`):

* не разбирается как DICOM — строка с `Failure` и причиной, процессор не вызывается;
* DICOM открылся, но это не снимок позвоночника или бедра — `Success` без области и
  класса, с причиной в `details["non_standard"]`;
* остальное уходит в процессор.

Повторная обработка пересоздаёт строки результата, но решение специалиста переносится
на новые строки того же снимка (`app.services.review.restore`). Если перенести нельзя —
снимок теперь не обработан или область другая, — решение врача записывается в
предупреждения исследования, чтобы оно не пропало молча.
"""

from __future__ import annotations

import logging
import time

from sqlalchemy import delete, select

from app.core.config import get_settings
from app.db.session import session_scope
from app.models import ImageResult, Study, StudyImage, StudyStatus
from app.models.study import utcnow
from app.processing import intake
from app.processing.base import ImageInput, ProcessingError
from app.processing.registry import get_processor
from app.services import review

logger = logging.getLogger(__name__)

STATUS_SUCCESS = "success"
STATUS_ERROR = "error"
STATUS_TIMEOUT = "timeout"


def mark_failed(study_id: str, message: str) -> None:
    with session_scope() as db:
        study = db.get(Study, study_id)
        if study is None:
            return
        study.status = StudyStatus.failed
        study.error_message = message
        study.finished_at = utcnow()


def process_study(study_id: str) -> None:
    settings = get_settings()
    try:
        processor = get_processor()
    except Exception as exc:  # noqa: BLE001
        logger.exception("Processor init failed")
        mark_failed(study_id, f"Не удалось инициализировать модель: {exc}")
        return

    # 1. Claim the study
    with session_scope() as db:
        study = db.get(Study, study_id)
        if study is None or study.status != StudyStatus.queued:
            logger.info("Study %s is not queued (skipping)", study_id)
            return
        study.status = StudyStatus.processing
        study.started_at = utcnow()
        study.finished_at = None
        study.progress = 0
        study.error_message = None
        study.processor_name = processor.name
        study.processor_version = processor.version
        study.is_mock = processor.is_mock
        saved_reviews = {
            r.image_id: saved
            for r in db.scalars(select(ImageResult).where(ImageResult.study_id == study_id))
            if (saved := review.snapshot(r)) is not None
        }
        db.execute(delete(ImageResult).where(ImageResult.study_id == study_id))
        images = db.scalars(
            select(StudyImage).where(StudyImage.study_id == study_id).order_by(StudyImage.original_filename)
        ).all()
        path_to_study = study.source_path or study.storage_dir
        study_uid = study.study_instance_uid
        invalid = {i.id: i.invalid_reason for i in images if i.invalid_reason}
        inputs = [
            ImageInput(
                image_id=i.id,
                path=settings.data_dir / i.stored_path,
                original_filename=i.original_filename,
                study_uid=study_uid,
                image_uid=i.sop_instance_uid,
                series_uid=i.series_instance_uid,
                modality=i.modality,
                body_part_examined=i.body_part_examined,
                rows=i.rows,
                columns=i.columns,
                has_pixel_data=i.has_pixel_data,
            )
            for i in images
        ]

    if not inputs:
        mark_failed(study_id, "В исследовании нет изображений")
        return

    logger.info("Processing study %s (%d images) with %s", study_id, len(inputs), processor.name)
    started = time.perf_counter()
    ok_count = 0
    lost_reviews: list[str] = []

    # 2. Process images one by one, persisting progress
    for idx, inp in enumerate(inputs, start=1):
        if time.perf_counter() - started > settings.processing_timeout_sec:
            row = ImageResult(
                study_id=study_id,
                image_id=inp.image_id,
                path_to_study=path_to_study,
                study_uid=study_uid,
                image_uid=inp.image_uid,
                processing_status=STATUS_TIMEOUT,
                time_of_processing=0.0,
                error_message="Превышен лимит времени обработки исследования",
            )
            pred = None
        else:
            t0 = time.perf_counter()
            pred, err = None, None
            try:
                if inp.image_id in invalid:
                    raise ProcessingError(invalid[inp.image_id])
                reason = intake.inspect(inp.path)
                pred = intake.non_standard(reason) if reason else processor.predict(inp)
            except ProcessingError as exc:
                err = str(exc)
            except Exception as exc:  # noqa: BLE001
                logger.exception("Unexpected error on image %s", inp.image_id)
                err = f"Внутренняя ошибка обработки: {type(exc).__name__}"
            elapsed = round(time.perf_counter() - t0, 3)

            row = ImageResult(
                study_id=study_id,
                image_id=inp.image_id,
                path_to_study=path_to_study,
                study_uid=study_uid,
                image_uid=inp.image_uid,
                processing_status=STATUS_SUCCESS if pred else STATUS_ERROR,
                time_of_processing=elapsed,
                error_message=err,
            )
            if pred:
                ok_count += 1
                row.anatomical_region = pred.anatomical_region or None
                row.quality_class = pred.quality_class or None
                row.violation_type = ";".join(pred.violation_types)
                row.confidence = pred.confidence
                row.details = pred.details

        saved = saved_reviews.get(inp.image_id)
        if saved is not None and (lost := review.restore(saved, row)):
            lost_reviews.append(
                f"{inp.original_filename}: проверка специалиста не перенесена после повторной обработки "
                f"({lost}). Решение было: {saved.describe()}"
            )

        with session_scope() as db:
            study = db.get(Study, study_id)
            if study is None:
                logger.info("Study %s was deleted during processing", study_id)
                return
            db.add(row)
            study.progress = int(idx / len(inputs) * 100)

    # 3. Finalize
    total = round(time.perf_counter() - started, 3)
    with session_scope() as db:
        study = db.get(Study, study_id)
        if study is None:
            return
        study.processing_time_sec = total
        study.finished_at = utcnow()
        if lost_reviews:
            study.warnings = list(dict.fromkeys([*(study.warnings or []), *lost_reviews]))
        study.progress = 100
        if ok_count:
            study.status = StudyStatus.completed
            failed = len(inputs) - ok_count
            study.error_message = f"Не обработано изображений: {failed}" if failed else None
        else:
            study.status = StudyStatus.failed
            study.error_message = "Ни одно изображение не удалось обработать"
    logger.info("Study %s done in %.2fs (%d/%d ok)", study_id, total, ok_count, len(inputs))
