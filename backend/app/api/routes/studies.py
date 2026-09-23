from datetime import UTC, datetime
from typing import Annotated, Literal

from fastapi import APIRouter, File, HTTPException, Query, Response, UploadFile, status
from fastapi.responses import FileResponse

from app.api.deps import DbDep, RunnerDep, SettingsDep
from app.models import StudyStatus
from app.schemas.study import (
    ProcessResponse,
    ResultRow,
    ReviewOut,
    ReviewRequest,
    StudyDetail,
    StudyList,
    StudyResultOut,
    StudyStatusOut,
    UploadResponse,
)
from app.services import export, package, review
from app.services import studies as svc
from app.services.dicom import render_preview_png
from app.services.dicom_sr import build_sr, to_bytes
from app.services.overlay import render_overlay_png
from app.services.upload import UploadLimitError, UploadService

router = APIRouter(prefix="/api/v1/studies", tags=["studies"])

ExportFormat = Literal["csv", "xlsx"]
ExportSource = Literal["auto", "reviewed"]


@router.post(
    "/upload",
    response_model=UploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Загрузка DICOM-файлов",
    description=(
        "Принимает один или несколько DICOM-файлов (`.dcm` или без расширения) и/или ZIP-архивы. "
        "Файлы группируются в исследования по `StudyInstanceUID`. Файл, который не разбирается как DICOM, "
        "тоже принимается: в результатах у него будет строка с `processing_status = Failure`. В `rejected` "
        "попадают только служебные файлы (DICOMDIR) и документы (таблицы, PDF, текст)."
    ),
)
async def upload_studies(
    db: DbDep,
    settings: SettingsDep,
    files: Annotated[list[UploadFile], File(description="DICOM-файлы или ZIP-архивы")],
) -> UploadResponse:
    if not files:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Файлы не переданы")
    try:
        outcome = await UploadService(db, settings).handle(files)
    except UploadLimitError as exc:
        raise HTTPException(status.HTTP_413_CONTENT_TOO_LARGE, str(exc)) from exc
    if not outcome.studies and outcome.rejected:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            {"message": "Ни один файл не распознан как DICOM", "rejected": outcome.rejected},
        )
    return UploadResponse(
        studies=[svc.to_summary(s) for s in outcome.studies],
        rejected=outcome.rejected,
        warnings=outcome.warnings,
    )


@router.get("", response_model=StudyList, summary="Список исследований")
def list_studies(
    db: DbDep,
    status_filter: Annotated[StudyStatus | None, Query(alias="status")] = None,
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> StudyList:
    items, total = svc.list_studies(db, status_filter, limit, offset)
    return StudyList(items=[svc.to_summary(s) for s in items], total=total, limit=limit, offset=offset)


@router.get("/{study_id}", response_model=StudyDetail, summary="Информация об исследовании")
def get_study(study_id: str, db: DbDep) -> StudyDetail:
    return svc.to_detail(svc.get_study_or_404(db, study_id))


@router.delete("/{study_id}", status_code=status.HTTP_204_NO_CONTENT, summary="Удалить исследование")
def delete_study(study_id: str, db: DbDep, settings: SettingsDep) -> Response:
    svc.delete_study(db, svc.get_study_or_404(db, study_id), settings)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post(
    "/{study_id}/process",
    response_model=ProcessResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Запуск обработки",
    description="Ставит исследование в очередь. Повторный запуск перезаписывает предыдущий результат.",
)
def process_study(study_id: str, db: DbDep, runner: RunnerDep) -> ProcessResponse:
    study = svc.get_study_or_404(db, study_id)
    svc.enqueue(db, study, runner)
    return ProcessResponse(study_id=study.id, status=study.status, message="Исследование поставлено в очередь")


@router.get("/{study_id}/status", response_model=StudyStatusOut, summary="Статус обработки")
def get_status(study_id: str, db: DbDep) -> StudyStatusOut:
    return svc.to_status(svc.get_study_or_404(db, study_id))


@router.get(
    "/{study_id}/result",
    response_model=StudyResultOut,
    summary="Результат обработки",
    responses={409: {"description": "Обработка не завершена или не запускалась"}},
)
def get_result(study_id: str, db: DbDep) -> StudyResultOut:
    study = svc.get_study_or_404(db, study_id)
    svc.ensure_results(study)
    return svc.to_result(study)


@router.get(
    "/{study_id}/download",
    summary="Скачать результат (CSV/XLSX)",
    response_class=Response,
    responses={
        200: {
            "content": {"text/csv": {}, export.CONTENT_TYPES["xlsx"]: {}},
            "description": "Файл результата",
        },
        409: {"description": "Результатов нет"},
    },
)
def download_result(
    study_id: str,
    db: DbDep,
    format: Annotated[ExportFormat, Query(description="Формат файла")] = "csv",  # noqa: A002
    source: Annotated[
        ExportSource,
        Query(
            description=(
                "auto — вердикт сервиса (по умолчанию; именно он оценивается по ТЗ), "
                "reviewed — решение специалиста там, где он его вынес"
            )
        ),
    ] = "auto",
) -> Response:
    study = svc.get_study_or_404(db, study_id)
    svc.ensure_results(study)
    content = (
        export.to_csv(study.results, source)
        if format == "csv"
        else export.to_xlsx(study.results, study.is_mock, source)
    )
    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    filename = f"densitometry_{study.id[:8]}_{stamp}.{format}"
    return Response(
        content=content,
        media_type=export.CONTENT_TYPES[format],
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.get(
    "/{study_id}/sr",
    summary="Заключение в виде DICOM Structured Report",
    description=(
        "Comprehensive SR со всем, что видно в интерфейсе: область, класс качества, тип нарушения, "
        "измеренные величины с критериями и координаты найденных структур. Идентификаторы пациента "
        "переносятся из исходного файла, чтобы отчёт встал в PACS к тому же исследованию."
    ),
    response_class=Response,
    responses={200: {"content": {"application/dicom": {}}}, 409: {"description": "Результатов нет"}},
)
def download_sr(study_id: str, db: DbDep, settings: SettingsDep) -> Response:
    study = svc.get_study_or_404(db, study_id)
    svc.ensure_results(study)
    content = to_bytes(build_sr(study, list(study.results), settings.data_dir))
    return Response(
        content=content,
        media_type="application/dicom",
        headers={"Content-Disposition": f'attachment; filename="densitometry_{study.id[:8]}_sr.dcm"'},
    )


@router.get(
    "/{study_id}/package",
    summary="ZIP-пакет: DICOM SR, вторичная серия с разметкой и таблица",
    response_class=Response,
    responses={200: {"content": {"application/zip": {}}}, 409: {"description": "Результатов нет"}},
)
def download_package(study_id: str, db: DbDep, settings: SettingsDep) -> Response:
    study = svc.get_study_or_404(db, study_id)
    svc.ensure_results(study)
    content = package.build_zip(study, settings.data_dir)
    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    return Response(
        content=content,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="densitometry_{study.id[:8]}_{stamp}.zip"'},
    )


@router.post(
    "/{study_id}/images/{image_id}/review",
    response_model=ReviewOut,
    summary="Подтвердить или исправить вердикт (проверка специалистом)",
    description=(
        "Автоматический вердикт не переписывается: решение врача пишется в отдельные поля, "
        "поэтому всегда видно, что предложил сервис и что решил человек. Исправление "
        "проверяется по тому же закрытому списку нарушений, что и автоматический вердикт."
    ),
    responses={404: {"description": "Изображение или результат не найдены"}},
)
def review_image(study_id: str, image_id: str, body: ReviewRequest, db: DbDep) -> ReviewOut:
    study = svc.get_study_or_404(db, study_id)
    result = next((r for r in study.results if r.image_id == image_id), None)
    if result is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Результат для этого изображения не найден")
    review.apply_review(result, body.action, body.violation_type, body.reviewed_by, body.comment)
    db.add(result)
    db.commit()
    db.refresh(result)
    names = {i.id: i.original_filename for i in study.images}
    row = ResultRow.model_validate(result).model_copy(update={"original_filename": names.get(result.image_id or "")})
    return ReviewOut(
        study_id=study.id,
        row=row,
        allowed_violations=list(review.allowed_for(result.anatomical_region)),
        agreement=review.agreement(list(study.results)),
    )


@router.get(
    "/{study_id}/images/{image_id}/preview",
    summary="PNG-превью изображения",
    response_class=FileResponse,
    responses={200: {"content": {"image/png": {}}}},
)
def image_preview(study_id: str, image_id: str, db: DbDep, settings: SettingsDep) -> FileResponse:
    study = svc.get_study_or_404(db, study_id)
    image = next((i for i in study.images if i.id == image_id), None)
    if image is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Изображение не найдено")
    cache = settings.data_dir / "previews" / study.id / f"{image.id}.png"
    if not cache.exists():
        try:
            png = render_preview_png(settings.data_dir / image.stored_path)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, f"Не удалось построить превью: {exc}") from exc
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_bytes(png)
    return FileResponse(cache, media_type="image/png", headers={"Cache-Control": "max-age=86400"})


@router.get(
    "/{study_id}/images/{image_id}/overlay",
    summary="PNG-превью с найденными структурами (ось, контур, ориентиры, посторонние объекты)",
    response_class=FileResponse,
    responses={200: {"content": {"image/png": {}}}},
)
def image_overlay(study_id: str, image_id: str, db: DbDep, settings: SettingsDep) -> FileResponse:
    study = svc.get_study_or_404(db, study_id)
    image = next((i for i in study.images if i.id == image_id), None)
    if image is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Изображение не найдено")
    result = next((r for r in study.results if r.image_id == image_id), None)
    overlay = (result.details or {}).get("overlay") if result else None
    if not overlay:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            "Разметка недоступна: исследование ещё не обработано или процессор её не возвращает",
        )
    cache = settings.data_dir / "previews" / study.id / f"{image.id}_overlay.png"
    if not cache.exists():
        try:
            png = render_overlay_png(settings.data_dir / image.stored_path, overlay)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(
                status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, f"Не удалось построить разметку: {exc}"
            ) from exc
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_bytes(png)
    return FileResponse(cache, media_type="image/png", headers={"Cache-Control": "max-age=86400"})
