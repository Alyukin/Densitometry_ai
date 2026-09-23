# Densitometry AI — контроль качества DXA-исследований

Сервис принимает DICOM денситометрии и говорит, годно ли исследование, какие нарушения
найдены и почему — с измеренной величиной и критерием из ТЗ рядом с каждым выводом.
Решает baseline на явных правилах; нейросеть подключается отдельным процессором.

## Запуск

Нужен Docker с плагином Compose v2 (`docker compose version`).
```bash
docker compose up --build
```

Сайт — http://localhost:8080, документация API (Swagger) — http://localhost:8080/docs.

Остановить — `docker compose down`, удалить данные — `docker compose down -v`. Настройки
менять необязательно; если нужно — `cp .env.example .env`, там всё с комментариями.
Тестовые файлы лежат в `samples/`: перетащите `samples/demo_studies.zip` на сайт.

## Что умеет сайт

Загрузить файлы, папку или ZIP → «Обработать» → статус и результат → скачать CSV, XLSX,
DICOM SR или ZIP-пакет. Файлы группируются по `StudyInstanceUID`, не-DICOM отклоняются с
причиной; есть пакетная обработка и сводная выгрузка по всем.

У каждого снимка — разметка поверх изображения (ось, контур бедра, ориентиры, посторонние
предметы) и все проверки с числами и порогами. Врач подтверждает или исправляет вердикт;
автоматический сохраняется отдельно, поэтому видно и предложение сервиса, и решение врача.

## API

Полное описание — в Swagger, кратко:

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
| GET | `/api/v1/studies/{id}/download?format=csv\|xlsx&source=auto\|reviewed` | Скачать результат |
| GET | `/api/v1/studies/{id}/sr` | Заключение в виде DICOM SR |
| GET | `/api/v1/studies/{id}/package` | ZIP: SR + серия с разметкой + таблицы |
| GET | `/api/v1/studies/{id}/images/{image_id}/preview` | PNG-превью снимка |
| GET | `/api/v1/studies/{id}/images/{image_id}/overlay` | PNG с найденными структурами |
| POST | `/api/v1/studies/{id}/images/{image_id}/review` | Подтвердить или исправить вердикт |
| POST | `/api/v1/batch/process` | Пакетная обработка |
| GET | `/api/v1/batch/download?format=csv\|xlsx` | Сводный файл |

Статусы: `uploaded → queued → processing → completed | failed`. Пример:

```bash
curl -F "files=@samples/demo_studies.zip" http://localhost:8080/api/v1/studies/upload
curl -H 'Content-Type: application/json' -d '{"all_pending": true}' \
     http://localhost:8080/api/v1/batch/process
curl -o result.xlsx "http://localhost:8080/api/v1/batch/download?format=xlsx"
```

Проверить всё сразу: `make smoke` (или `bash scripts/smoke_test.sh`).

## Результат

Одна строка на снимок, колонки по ТЗ плюс разрешённая заказчиком `quality_prob`:
```
path_to_study,study_uid,image_uid,anatomical_region,quality_class,violation_type,processing_status,time_of_processing,quality_prob
```

`quality_class` — целое 0 или 1, `processing_status` — `Success` или `Failure` (только
когда оценивать нечего: не DICOM, нет пикселей, пустой кадр), время в секундах, `anatomical_region` и `violation_type` — из закрытых списков заказчика (несколько
нарушений через `;`). В XLSX есть листы `checks` (проверки с порогами) и `review`.

## Как устроено

- `frontend/` — React + Vite, раздаётся через nginx, он же проксирует запросы к API.
- `backend/` — FastAPI, файлы и база SQLite в Docker-томе `densitometry-data`; из DICOM
  сохраняются только технические метаданные, без персональных данных.
- Обработка идёт в фоновых потоках процесса API: backend запускается одним процессом, а
  после перезапуска незавершённые задачи встают в очередь заново.
- `ml/` — датасет, калибровка порогов и метрики: [ml/README.md](ml/README.md).

## Как подключить модель

1. В `backend/app/processing/` — класс на основе `BaseProcessor` с `load()` и `predict()`.
2. Зарегистрировать в `registry.py` и указать `PROCESSOR_BACKEND=<имя>` в `.env`.
3. Зависимости в `backend/requirements.txt`, при необходимости — GPU-секция в compose.

Сайт, API и выгрузку менять не нужно: процессоры взаимозаменяемы, метрики считаются одним
кодом на одних фолдах. Есть `rulebased` (по умолчанию) и `mock`; подробности — в
комментариях `backend/app/processing/base.py`.

## Разработка без Docker

Нужны Python 3.12 и Node.js 22.
```bash
make install        # зависимости
make dev-backend    # API на :8000
make dev-frontend   # сайт на :5173
make test           # тесты backend и метрик
make localization   # метрики локализации по ТЗ
```

Тесты в контейнере — `make test-docker`. Все команды есть в `make help`.

## Ограничения

- Результаты тестовые, это не медицинское заключение.
- Входа по паролю нет — сервис рассчитан на закрытый контур; SQLite и очередь внутри
  процесса рассчитаны на один экземпляр backend.

Что проверяется и что делать при сбоях — [status.md](status.md); качество и метрики —
[ml/baseline/REPORT.md](ml/baseline/REPORT.md); принятые решения — [questions.md](questions.md);
дальнейшие этапы — [roadmap.md](roadmap.md).
