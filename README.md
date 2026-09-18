# Densitometry AI — контроль качества DXA-исследований

Этап 1: сайт + API + заглушка вместо AI. Загрузка DICOM, очередь, статусы, результаты и
выгрузка CSV/XLSX работают по-настоящему; сами результаты пока тестовые.

## Запуск

Нужен Docker с плагином Compose v2 (`docker compose version`).

```bash
docker compose up --build
```

- Сайт: http://localhost:8080
- Документация API (Swagger): http://localhost:8080/docs

Остановить — `docker compose down`, удалить данные — `docker compose down -v`.

Настройки менять необязательно. Если нужно — `cp .env.example .env`, там всё с
комментариями (порты, число параллельных обработок, лимиты).

Тестовые файлы лежат в `samples/`: перетащите `samples/demo_studies.zip` на сайт.

## Что умеет сайт

Загрузить файлы, папку или ZIP → нажать «Обработать» → увидеть статус и результат →
скачать CSV или XLSX. Файлы сами группируются в исследования по `StudyInstanceUID`,
не-DICOM отклоняются с указанием причины. Есть пакетная обработка выбранных или всех
новых исследований и сводная выгрузка по всем.

## API

Полное описание — в Swagger. Кратко:

| Метод | Путь | Что делает |
|---|---|---|
| GET | `/health` | Сервис жив, какой процессор используется |
| POST | `/api/v1/studies/upload` | Загрузить DICOM или ZIP (поле `files`) |
| GET | `/api/v1/studies` | Список исследований |
| GET | `/api/v1/studies/{id}` | Одно исследование и его снимки |
| DELETE | `/api/v1/studies/{id}` | Удалить исследование |
| POST | `/api/v1/studies/{id}/process` | Запустить обработку |
| GET | `/api/v1/studies/{id}/status` | Статус и прогресс |
| GET | `/api/v1/studies/{id}/result` | Результат в JSON |
| GET | `/api/v1/studies/{id}/download?format=csv\|xlsx` | Скачать результат |
| GET | `/api/v1/studies/{id}/images/{image_id}/preview` | PNG-превью снимка |
| POST | `/api/v1/batch/process` | Пакетная обработка |
| GET | `/api/v1/batch/download?format=csv\|xlsx` | Сводный файл |

Статусы: `uploaded → queued → processing → completed | failed`.

Пример:

```bash
curl -F "files=@samples/demo_studies.zip" http://localhost:8080/api/v1/studies/upload
curl -H 'Content-Type: application/json' -d '{"all_pending": true}' \
     http://localhost:8080/api/v1/batch/process
curl -o result.xlsx "http://localhost:8080/api/v1/batch/download?format=xlsx"
```

Проверить всё сразу: `make smoke` (или `bash scripts/smoke_test.sh`).

## Результат

Одна строка на снимок, колонки по ТЗ:

```
path_to_study,study_uid,image_uid,anatomical_region,quality_class,violation_type,processing_status,time_of_processing
```

Нарушения перечисляются через `;`, время обработки — в секундах. В XLSX есть ещё лист
`checks` с результатом каждой проверки.

> Названия классов и нарушений сейчас — заглушки по тексту ТЗ. Реальные нужно взять из
> разметки датасета и поправить в `backend/app/processing/` и `frontend/src/utils/format.ts`.

## Как устроено

- `frontend/` — React + Vite, раздаётся через nginx, он же проксирует запросы к API.
- `backend/` — FastAPI. Файлы и база SQLite лежат в Docker-томе `densitometry-data`.
  Из DICOM сохраняются только технические метаданные, без персональных данных.
- Обработка идёт в фоновых потоках внутри процесса API, поэтому backend запускается
  одним процессом. Если сервис перезапустить, незавершённые задачи встанут в очередь заново.

## Как подключить модель

1. В `backend/app/processing/` написать класс на основе `BaseProcessor` с методами
   `load()` (загрузка весов) и `predict(image)` (обработка одного снимка).
2. Зарегистрировать его в `registry.py` и указать `PROCESSOR_BACKEND=<имя>` в `.env`.
3. Добавить зависимости в `backend/requirements.txt`, при необходимости раскомментировать
   GPU-секцию в `docker-compose.yml`.

Сайт, API и выгрузку менять не нужно. Подробности и пример кода — в комментариях
`backend/app/processing/base.py`.

## Разработка без Docker

Нужны Python 3.12 и Node.js 22.

```bash
make install        # зависимости
make dev-backend    # API на :8000
make dev-frontend   # сайт на :5173
make test           # тесты
```

Тесты в контейнере — `make test-docker`. Все команды есть в `make help`.

## Ограничения

- Результаты тестовые, это не медицинское заключение.
- Входа по паролю нет — сервис рассчитан на закрытый контур.
- SQLite и очередь в процессе рассчитаны на один экземпляр backend.
