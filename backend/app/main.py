"""FastAPI application entrypoint."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.docs import get_redoc_html, get_swagger_ui_html
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy import select

from app.api.routes import batch, health, studies
from app.core.config import get_settings
from app.core.logging import setup_logging
from app.db.session import init_db, session_scope
from app.models import ACTIVE_STATUSES, Study, StudyStatus
from app.processing.registry import get_processor
from app.services.task_runner import TaskRunner

logger = logging.getLogger("app")

# Swagger UI / ReDoc assets are vendored into the Docker image (see backend/Dockerfile),
# so the API docs work without internet access. Locally they fall back to the CDN.
DOCS_ASSETS_DIR = Path(__file__).resolve().parent / "static" / "docs"
DOCS_ASSETS_URL = "/docs-static"

TAGS = [
    {"name": "health", "description": "Состояние сервиса"},
    {"name": "studies", "description": "Загрузка, просмотр и обработка DICOM-исследований"},
    {"name": "batch", "description": "Пакетная обработка и сводная выгрузка"},
]


def _recover_interrupted(runner: TaskRunner) -> None:
    """Re-queue studies that were queued/processing when the service stopped."""
    with session_scope() as db:
        stuck = db.scalars(select(Study).where(Study.status.in_(ACTIVE_STATUSES))).all()
        ids = []
        for s in stuck:
            s.status = StudyStatus.queued
            s.progress = 0
            ids.append(s.id)
    for sid in ids:
        runner.submit(sid)
    if ids:
        logger.warning("Re-queued %d interrupted studies", len(ids))


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    setup_logging(settings.log_level)
    settings.ensure_dirs()
    init_db()
    try:
        get_processor()  # load model once at startup
    except Exception:  # noqa: BLE001
        logger.exception("Processor failed to load; /health will report degraded")
    runner = TaskRunner(settings.worker_concurrency)
    app.state.runner = runner
    _recover_interrupted(runner)
    logger.info("%s %s started (processor=%s)", settings.app_name, settings.app_version, settings.processor_backend)
    yield
    runner.shutdown(wait=False)


def _setup_docs(app: FastAPI) -> None:
    local = (DOCS_ASSETS_DIR / "swagger-ui-bundle.js").exists()
    if local:
        app.mount(DOCS_ASSETS_URL, StaticFiles(directory=DOCS_ASSETS_DIR), name="docs-static")
    cdn_swagger = "https://cdn.jsdelivr.net/npm/swagger-ui-dist@5"
    swagger_js = f"{DOCS_ASSETS_URL}/swagger-ui-bundle.js" if local else f"{cdn_swagger}/swagger-ui-bundle.js"
    swagger_css = f"{DOCS_ASSETS_URL}/swagger-ui.css" if local else f"{cdn_swagger}/swagger-ui.css"
    redoc_js = (
        f"{DOCS_ASSETS_URL}/redoc.standalone.js"
        if (DOCS_ASSETS_DIR / "redoc.standalone.js").exists()
        else "https://cdn.jsdelivr.net/npm/redoc@2/bundles/redoc.standalone.js"
    )

    @app.get("/docs", include_in_schema=False)
    def swagger_ui() -> HTMLResponse:
        return get_swagger_ui_html(
            openapi_url=app.openapi_url or "/openapi.json",
            title=f"{app.title} — Swagger UI",
            swagger_js_url=swagger_js,
            swagger_css_url=swagger_css,
            swagger_favicon_url=f"{DOCS_ASSETS_URL}/favicon-32x32.png"
            if local
            else "https://fastapi.tiangolo.com/img/favicon.png",
        )

    @app.get("/redoc", include_in_schema=False)
    def redoc() -> HTMLResponse:
        return get_redoc_html(
            openapi_url=app.openapi_url or "/openapi.json",
            title=f"{app.title} — ReDoc",
            redoc_js_url=redoc_js,
            with_google_fonts=False,
        )


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title=f"{settings.app_name} API",
        version=settings.app_version,
        description=(
            "API сервиса контроля качества DXA-исследований (денситометрия).\n\n"
            "**Этап 1:** обработка выполняется mock-процессором — результаты тестовые."
        ),
        openapi_tags=TAGS,
        lifespan=lifespan,
        docs_url=None,
        redoc_url=None,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["Content-Disposition"],
    )

    @app.exception_handler(Exception)
    async def unhandled(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled error on %s %s", request.method, request.url.path)
        return JSONResponse(status_code=500, content={"detail": "Внутренняя ошибка сервера"})

    _setup_docs(app)
    app.include_router(health.router)
    app.include_router(studies.router)
    app.include_router(batch.router)
    return app


app = create_app()
