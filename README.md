# Densitometry AI — контроль качества DXA-исследований

Прототип сервиса (этап 1): **веб-интерфейс + REST API + mock-обработка**.
AI-модель и медицинская логика на этом этапе не реализованы: обработка выполняется
детерминированной заглушкой, которая возвращает тестовые результаты в итоговом формате.
Всё остальное (загрузка DICOM, очередь, статусы, результаты, выгрузка CSV/XLSX, пакетная
обработка) работает полностью.

## Быстрый старт

Требуется Docker с плагином Docker Compose v2 (`docker compose version`).

```bash
docker compose up --build        # или: make up
```

| Что | Адрес |
|---|---|
| Веб-интерфейс | http://localhost:8080 |
| Swagger UI | http://localhost:8080/docs (или напрямую http://localhost:8000/docs) |
| ReDoc | http://localhost:8080/redoc |
| OpenAPI JSON | http://localhost:8080/openapi.json |
| Health | http://localhost:8080/health |

`.env` необязателен: без него используются значения по умолчанию. Для настройки
скопируйте `cp .env.example .env` (`make up` делает это сам).

Тестовые данные: в `samples/` лежат синтетические DICOM (без персональных данных) —
`samples/demo_studies.zip` можно сразу перетащить в веб-интерфейс.
Проверка запущенного сервиса из консоли (нужны `curl` и `python3`): `make smoke`.

## Возможности интерфейса

- загрузка одного или нескольких DICOM-файлов, целой папки (drag & drop) или ZIP-архива;
  файлы автоматически группируются в исследования по `StudyInstanceUID`;
- отчёт о загрузке: отклонённые файлы (не DICOM, DICOMDIR, дубликаты) и предупреждения;
- список исследований с фильтрами по статусу, сводкой и счётчиками;
- запуск обработки одного исследования, выбранных или всех новых (batch);
- статус и прогресс в реальном времени (polling);
- карточка исследования: превью изображений, метаданные, результат по каждому изображению
  (область, класс качества, нарушения, пошаговые проверки, уверенность, время);
- скачивание CSV/XLSX по исследованию, по выбранным или сводно по всем;
- явная пометка **MOCK** у тестовых результатов.

## API

Базовый префикс — `/api/v1`. Полное описание схем — в Swagger.

| Метод | Путь | Назначение |
|---|---|---|
| GET | `/health` | Состояние сервиса (БД, процессор, очередь) |
| POST | `/api/v1/studies/upload` | Загрузка DICOM / ZIP (`multipart/form-data`, поле `files`, можно несколько) |
| GET | `/api/v1/studies` | Список исследований (`?status=&limit=&offset=`) |
| GET | `/api/v1/studies/{study_id}` | Информация об исследовании и его изображениях |
| DELETE | `/api/v1/studies/{study_id}` | Удалить исследование и файлы |
| POST | `/api/v1/studies/{study_id}/process` | Поставить в очередь (202). Повторный запуск перезаписывает результат |
| GET | `/api/v1/studies/{study_id}/status` | Статус и прогресс |
| GET | `/api/v1/studies/{study_id}/result` | Результат в JSON (409, если ещё нет) |
| GET | `/api/v1/studies/{study_id}/download?format=csv\|xlsx` | Файл результата |
| GET | `/api/v1/studies/{study_id}/images/{image_id}/preview` | PNG-превью изображения |
| POST | `/api/v1/batch/process` | Пакетная обработка: `{"study_ids": [...]}` и/или `{"all_pending": true}`, `force` |
| GET | `/api/v1/batch/download?format=csv\|xlsx&study_ids=...` | Сводный файл по нескольким исследованиям |

Статусы исследования: `uploaded → queued → processing → completed | failed`.

Пример из консоли:

```bash
# загрузка
curl -F "files=@samples/demo_studies.zip" http://localhost:8080/api/v1/studies/upload
# обработать всё загруженное
curl -H 'Content-Type: application/json' -d '{"all_pending": true}' \
     http://localhost:8080/api/v1/batch/process
# статус и результат
curl http://localhost:8080/api/v1/studies/<study_id>/status
curl -o result.xlsx "http://localhost:8080/api/v1/studies/<study_id>/download?format=xlsx"
```

### Формат результата

Одна строка на изображение (анатомическую область), колонки строго по ТЗ:

```
path_to_study,study_uid,image_uid,anatomical_region,quality_class,violation_type,processing_status,time_of_processing
```

- `path_to_study` — относительный путь исследования в загрузке (папка / путь внутри ZIP);
- `violation_type` — коды нарушений через `;`, пусто при отсутствии нарушений;
- `processing_status` — `success` | `error` | `timeout`;
- `time_of_processing` — время обработки изображения, секунды.

В XLSX дополнительно есть листы `checks` (результат каждой проверки — для объяснимости)
и `meta` (время выгрузки, признак mock).

> **Коды классов в mock — заглушки.** `lumbar_spine / proximal_femur`, `acceptable / unacceptable`
> и коды нарушений взяты из текста ТЗ. Реальные классы нужно определить по экспертной
> разметке датасета и синхронизировать в `backend/app/processing/` и
> `frontend/src/utils/format.ts`.

## Архитектура

```
┌──────────────┐   /api, /docs, /health   ┌───────────────────────────────────────────┐
│  frontend    │ ───────────────────────▶ │ backend (FastAPI, 1 процесс uvicorn)       │
│  nginx +     │                          │  routes → services → TaskRunner (threads) │
│  React SPA   │                          │                       └→ pipeline         │
└──────────────┘                          │                            └→ Processor   │
     :8080                                │  SQLite + файлы  ← volume /data           │
                                          └───────────────────────────────────────────┘
                                                             :8000
```

- **Хранение:** SQLite (`/data/densitometry.db`) и файлы в `/data/uploads/<study_id>/`
  в Docker-volume `densitometry-data`. Из DICOM сохраняются только технические
  метаданные (UID, модальность, производитель, BodyPartExamined, размер) — без ПДн.
- **Очередь:** пул потоков внутри процесса API (`WORKER_CONCURRENCY`). Поэтому backend
  запускается одним процессом uvicorn. После перезапуска незавершённые задачи
  автоматически ставятся в очередь заново.
- **Обработка:** `pipeline.process_study` не зависит от конкретной модели — он вызывает
  процессор для каждого изображения, пишет результат и прогресс, соблюдает лимит времени
  (`PROCESSING_TIMEOUT_SEC`), изолирует ошибки отдельных изображений.

### Как подключить AI-модель

1. Реализовать класс в `backend/app/processing/`:

   ```python
   class DxaQualityModel(BaseProcessor):
       name, version = "dxa_qc", "1.0.0"

       def load(self):                       # один раз при старте: веса, прогрев GPU
           ...

       def predict(self, image: ImageInput) -> ImagePrediction:
           # image.path — путь к DICOM; image.body_part_examined и др. метаданные
           return ImagePrediction(
               anatomical_region=..., quality_class=..., violation_types=[...],
               confidence=..., details={"checks": [...], "keypoints": ..., "mask_path": ...},
           )
   ```

   Ожидаемые ошибки отдельного изображения — `raise ProcessingError("...")`.
2. Зарегистрировать в `backend/app/processing/registry.py` и выставить
   `PROCESSOR_BACKEND=dxa_qc` в `.env`.
3. Добавить зависимости модели в `backend/requirements.txt` и, при необходимости,
   GPU-ресурсы в `docker-compose.yml` (заготовка закомментирована).

API, UI, экспорт и тесты API менять не нужно. `details.checks` уже отображаются в
интерфейсе и выгружаются в XLSX. Если потребуется вынести инференс в отдельный
GPU-воркер, заменяется только `services/task_runner.py` (например, на Celery/RQ) —
контракт `submit(study_id)` сохраняется.

## Структура проекта

```
.
├── docker-compose.yml        # запуск одной командой
├── .env.example              # все настройки с комментариями
├── Makefile                  # make up / test / dev-backend / ...
├── samples/                  # синтетические DICOM + demo_studies.zip
├── scripts/smoke_test.sh     # e2e-проверка запущенного сервиса
├── backend/
│   ├── Dockerfile            # runtime + test stage, офлайн-ассеты Swagger/ReDoc
│   ├── requirements*.txt     # зафиксированные версии
│   ├── app/
│   │   ├── main.py           # FastAPI, lifespan, docs
│   │   ├── core/config.py    # настройки из env
│   │   ├── db/session.py     # SQLAlchemy, SQLite
│   │   ├── models/           # Study, StudyImage, ImageResult
│   │   ├── schemas/          # Pydantic-схемы API
│   │   ├── api/routes/       # health, studies, batch
│   │   ├── services/         # upload, dicom, pipeline, task_runner, export, studies
│   │   ├── processing/       # BaseProcessor, MockProcessor, registry  ← сюда модель
│   │   └── scripts/          # генератор синтетических DICOM
│   └── tests/                # pytest: API end-to-end + mock-процессор
└── frontend/
    ├── Dockerfile            # сборка Vite → nginx
    ├── nginx.conf.template   # SPA + proxy на backend
    └── src/
        ├── api/              # типизированный клиент API
        ├── components/       # UploadPanel, StudiesTable, StudyDrawer, ...
        ├── hooks/ utils/
        └── styles.css
```

## Разработка без Docker

Нужны Python 3.12 и Node.js 22.

```bash
make install        # venv + npm ci
make dev-backend    # http://localhost:8000 (autoreload)
make dev-frontend   # http://localhost:5173 (proxy /api → :8000)
make test           # pytest
make lint           # ruff + tsc
make samples        # пересоздать samples/
```

Тесты в контейнере: `make test-docker`.

## Настройки

Все переменные описаны в `.env.example`. Основные:

| Переменная | По умолчанию | Описание |
|---|---|---|
| `FRONTEND_PORT` / `BACKEND_PORT` | 8080 / 8000 | Порты на хосте |
| `PROCESSOR_BACKEND` | `mock` | Какой процессор использовать |
| `WORKER_CONCURRENCY` | 2 | Параллельно обрабатываемых исследований |
| `PROCESSING_TIMEOUT_SEC` | 180 | Лимит на исследование (ТЗ: ≤ 3 мин) |
| `MOCK_DELAY_PER_IMAGE_SEC` | 1.5 | Имитация времени инференса |
| `MAX_UPLOAD_SIZE_MB` | 1024 | Лимит размера одного запроса (backend и nginx) |
| `MAX_IMAGES_PER_STUDY` | 3 | Порог предупреждения (по ТЗ) |

## Ограничения этапа 1

- результаты — тестовые, **не являются медицинским заключением**;
- нет аутентификации — сервис рассчитан на работу в закрытом контуре;
- SQLite и очередь в процессе рассчитаны на один экземпляр backend;
- превью строится для несжатых и JPEG/JPEG 2000 DICOM, overlay/маски появятся вместе с моделью.
