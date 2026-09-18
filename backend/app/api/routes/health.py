import logging

from fastapi import APIRouter
from sqlalchemy import text

from app.api.deps import DbDep, RunnerDep, SettingsDep
from app.processing.registry import get_processor
from app.schemas.study import HealthOut

router = APIRouter(tags=["health"])
logger = logging.getLogger(__name__)


@router.get("/health", response_model=HealthOut, summary="Проверка состояния сервиса")
def health(db: DbDep, runner: RunnerDep, settings: SettingsDep) -> HealthOut:
    db_ok = True
    try:
        db.execute(text("SELECT 1"))
    except Exception:  # noqa: BLE001
        logger.exception("DB health check failed")
        db_ok = False

    proc_name, proc_version, is_mock = settings.processor_backend, None, None
    proc_ok = True
    try:
        proc = get_processor()
        proc_name, proc_version, is_mock = proc.name, proc.version, proc.is_mock
    except Exception:  # noqa: BLE001
        logger.exception("Processor health check failed")
        proc_ok = False

    return HealthOut(
        status="ok" if db_ok and proc_ok else "degraded",
        version=settings.app_version,
        environment=settings.environment,
        processor=proc_name,
        processor_version=proc_version,
        is_mock=is_mock,
        database="ok" if db_ok else "error",
        workers=runner.workers,
        tasks_in_progress=runner.inflight,
    )
