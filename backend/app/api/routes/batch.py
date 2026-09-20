from datetime import UTC, datetime
from typing import Annotated, Literal

from fastapi import APIRouter, HTTPException, Query, Response, status
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from app.api.deps import DbDep, RunnerDep
from app.models import ACTIVE_STATUSES, Study, StudyStatus
from app.schemas.study import BatchProcessRequest, BatchProcessResponse, BatchSkipped
from app.services import export
from app.services import studies as svc

router = APIRouter(prefix="/api/v1/batch", tags=["batch"])


@router.post(
    "/process",
    response_model=BatchProcessResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Пакетная обработка",
    description=(
        "Ставит в очередь несколько исследований. Можно передать список `study_ids` "
        "и/или `all_pending=true` (все со статусом uploaded/failed). "
        "Завершённые исследования пропускаются, если не указан `force=true`."
    ),
)
def batch_process(payload: BatchProcessRequest, db: DbDep, runner: RunnerDep) -> BatchProcessResponse:
    ids: list[str] = list(dict.fromkeys(payload.study_ids))
    if payload.all_pending:
        pending = db.scalars(
            select(Study.id)
            .where(Study.status.in_([StudyStatus.uploaded, StudyStatus.failed]))
            .order_by(Study.created_at)
        ).all()
        ids.extend(i for i in pending if i not in ids)
    if not ids:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Не указаны исследования для обработки")

    accepted: list[str] = []
    skipped: list[BatchSkipped] = []
    for sid in ids:
        study = db.get(Study, sid)
        if study is None:
            skipped.append(BatchSkipped(study_id=sid, reason="not_found"))
        elif study.status in ACTIVE_STATUSES:
            skipped.append(BatchSkipped(study_id=sid, reason=f"already_{study.status.value}"))
        elif study.status == StudyStatus.completed and not payload.force:
            skipped.append(BatchSkipped(study_id=sid, reason="already_completed"))
        else:
            svc.enqueue(db, study, runner)
            accepted.append(sid)
    return BatchProcessResponse(accepted=accepted, skipped=skipped)


@router.get(
    "/download",
    summary="Скачать сводный результат (CSV/XLSX)",
    description="Объединяет результаты нескольких исследований в один файл. Без `study_ids` — все обработанные.",
    response_class=Response,
    responses={200: {"content": {"text/csv": {}, export.CONTENT_TYPES["xlsx"]: {}}}},
)
def batch_download(
    db: DbDep,
    study_ids: Annotated[list[str] | None, Query()] = None,
    format: Annotated[Literal["csv", "xlsx"], Query()] = "csv",  # noqa: A002
    source: Annotated[
        Literal["auto", "reviewed"],
        Query(description="auto — вердикт сервиса (по умолчанию), reviewed — решение специалиста"),
    ] = "auto",
) -> Response:
    q = select(Study).options(selectinload(Study.results)).order_by(Study.created_at)
    if study_ids:
        q = q.where(Study.id.in_(study_ids))
    studies = [s for s in db.scalars(q).all() if s.results and s.status not in ACTIVE_STATUSES]
    if not studies:
        raise HTTPException(status.HTTP_409_CONFLICT, "Нет обработанных исследований для выгрузки")
    results = [r for s in studies for r in s.results]
    is_mock = any(s.is_mock for s in studies)
    content = export.to_csv(results, source) if format == "csv" else export.to_xlsx(results, is_mock, source)
    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    return Response(
        content=content,
        media_type=export.CONTENT_TYPES[format],
        headers={"Content-Disposition": f'attachment; filename="densitometry_batch_{stamp}.{format}"'},
    )
