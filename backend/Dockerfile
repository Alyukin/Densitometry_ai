# syntax=docker/dockerfile:1
ARG PYTHON_IMAGE=python:3.12-slim-bookworm
ARG NODE_IMAGE=node:22-alpine

# ---------- Swagger UI / ReDoc assets (offline API docs) ----------
FROM ${NODE_IMAGE} AS docs-assets
WORKDIR /tmp/docs
RUN npm init -y >/dev/null \
    && npm install --no-audit --no-fund --omit=dev --ignore-scripts swagger-ui-dist@5.33.0 redoc@2.5.4 \
    && mkdir -p /out \
    && cp node_modules/swagger-ui-dist/swagger-ui-bundle.js \
          node_modules/swagger-ui-dist/swagger-ui.css \
          node_modules/swagger-ui-dist/favicon-32x32.png \
          node_modules/redoc/bundles/redoc.standalone.js /out/

# ---------- Второе мнение: замороженный DenseNet121 (TorchXRayVision) -> ONNX ----------
# torch нужен только здесь, в образ сервиса попадает один файл ONNX (~27 МБ), а выполняет
# его onnxruntime. Веса публичные (Apache-2.0) и скачиваются с GitHub при сборке.
FROM ${PYTHON_IMAGE} AS second-opinion
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cpu \
    && pip install torchxrayvision==1.5.4 onnx==1.23.0
COPY app/scripts/export_second_opinion.py /tmp/export_second_opinion.py
RUN python /tmp/export_second_opinion.py --out /out/xrv_densenet121.onnx

# ---------- Base: runtime dependencies ----------
FROM ${PYTHON_IMAGE} AS base
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /app
# шрифт с кириллицей для подписей на разметке (app/services/overlay.py)
RUN apt-get update \
    && apt-get install -y --no-install-recommends fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install -r requirements.txt

# ---------- Test image: `make test-docker` ----------
FROM base AS test
COPY requirements-dev.txt pyproject.toml ./
RUN pip install -r requirements-dev.txt
COPY app ./app
COPY tests ./tests
CMD ["pytest"]

# ---------- Runtime image ----------
FROM base AS runtime
RUN groupadd --system --gid 10001 app \
    && useradd --system --uid 10001 --gid app --home-dir /app --no-create-home app \
    && mkdir -p /data \
    && chown app:app /data
COPY app ./app
COPY --from=docs-assets /out ./app/static/docs
COPY --from=second-opinion /out/xrv_densenet121.onnx ./app/processing/second_opinion_model/
# Код принадлежит root и доступен всем только на чтение — независимо от прав файлов на хосте
# (иначе при umask 077 пользователь app не сможет импортировать модули).
RUN chmod -R a+rX,go-w /app
USER app
ENV DATA_DIR=/data
VOLUME ["/data"]
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --start-period=20s --retries=5 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)"]
# Один процесс uvicorn: очередь обработки живёт внутри процесса API (см. OVERVIEW.md, «Архитектура»).
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--proxy-headers", "--forwarded-allow-ips", "*"]
