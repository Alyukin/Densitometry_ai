from datetime import UTC, datetime
from typing import Annotated, Literal

from fastapi import APIRouter, File, HTTPException, Query, Response, UploadFile, status
from fastapi.responses import FileResponse

from app.api.deps import DbDep, RunnerDep, SettingsDep
from app.models import StudyStatus
from app.schemas.study import (
    ProcessResponse,
    StudyDetail,
    StudyList,
    StudyResultOut,
    StudyStatusOut,
    UploadResponse,
)
from app.services import export
from app.services import studies as svc
from app.services.dicom import render_preview_png
from app.services.upload import UploadLimitError, UploadService

router = APIRouter(prefix="/api/v1/studies", tags=["studies"])

ExportFormat = Literal["csv", "xlsx"]


@router.post(
    "/upload",
    response_model=UploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Загрузка DICOM-файлов",
    description=(
        "Принимает один или несколько DICOM-файлов (`.dcm` или без расширения) и/или ZIP-архивы. "
        "Файлы группируются в исследования по `StudyInstanceUID`. Невалидные файлы возвращаются в `rejected`."
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
) -> Response:
    study = svc.get_study_or_404(db, study_id)
    svc.ensure_results(study)
    content = export.to_csv(study.results) if format == "csv" else export.to_xlsx(study.results, study.is_mock)
    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    filename = f"densitometry_{study.id[:8]}_{stamp}.{format}"
    return Response(
        content=content,
        media_type=export.CONTENT_TYPES[format],
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
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
