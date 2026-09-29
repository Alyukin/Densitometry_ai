# Densitometry AI — контроль качества DXA-исследований

Сервис принимает DICOM денситометрии и по каждому снимку поясничного отдела
позвоночника или проксимального отдела бедра говорит, годен ли он, какие нарушения
укладки найдены и почему — с измеренной величиной и критерием из ТЗ рядом с каждым
выводом. Вердикт выносят правила ТЗ; для бедра рядом показывается справочная оценка
ИИ-модели.

Здесь — как запустить и пользоваться. Всё остальное — цель, данные, архитектура,
метрики, что пробовали, решения и риски — в [OVERVIEW.md](OVERVIEW.md).

## Запуск

Нужен Docker с плагином Compose v2 (`docker compose version`).

```bash
docker compose up --build
```

Сайт — http://localhost:8080, документация API (Swagger) — http://localhost:8080/docs.
Остановить — `docker compose down`, удалить все данные — `docker compose down -v`.
Тестовые файлы лежат в `samples/`: перетащите `samples/demo_studies.zip` на сайт.

Видеокарта не нужна: правила и ИИ-модель второго мнения работают на CPU.

## Что умеет сайт

Загрузить файлы, папку или ZIP → «Обработать» → статус и результат → скачать CSV, XLSX,
DICOM SR или ZIP-пакет. Есть пакетная обработка и сводная выгрузка по всем исследованиям.
Удалить всё сразу: галочка в шапке таблицы → корзина рядом с выгрузкой (без подтверждения).

- **Загрузка.** Файлы группируются по `StudyInstanceUID`. Большая папка уходит частями по
  200 файлов, исследование собирается обратно. Снимок, который уже есть в сервисе,
  отклоняется; новые снимки уже загруженного исследования добавляются к нему. Документы
  рядом со снимками (таблицы, PDF, текст, в том числе .xlsx) пропускаются с причиной.
  Файл, который не разбирается как DICOM, получает строку `Failure`.
- **Результат.** У каждого снимка — разметка поверх изображения (ось, контур бедра,
  ориентиры, посторонние предметы) и все проверки ТЗ с числами и порогами; справочные
  помечены. Для бедра — три строки «Оценка ИИ-модели» с её точностью; в вердикт и
  выгрузку они не входят.
- **Проверка врачом.** «Подтвердить» или «Исправить» по тому же закрытому списку
  нарушений. Автоматический вердикт не переписывается: решение врача хранится отдельно и
  переживает повторную обработку. Выгрузка по умолчанию — вердикт сервиса, решение врача —
  `?source=reviewed`; в XLSX обе версии рядом на листе `review`.

## Результат

Одна строка на снимок, колонки по п. 2.5 ТЗ плюс разрешённая заказчиком `quality_prob`:

```
path_to_study,study_uid,image_uid,anatomical_region,quality_class,violation_type,processing_status,time_of_processing,quality_prob
```

- `path_to_study` — путь от загруженной папки (или ZIP) до общей папки снимков исследования;
  `study_uid` и `image_uid` — StudyInstanceUID и SOPInstanceUID из тегов DICOM.
- `quality_class` — целое 0 или 1; `violation_type` — из закрытых списков заказчика,
  несколько через `;`:
  - позвоночник: «Некорректная укладка», «Не выравнена ось позвоночника», «Присутствуют
    посторонние предметы»;
  - бедро: «Некорректная укладка», «Некорректная область интереса».
- `processing_status` — `Success` или `Failure` (файл не открывается или не разбирается
  как DICOM; строка есть и для него).
- DICOM, который не является снимком позвоночника или бедра, — `Success` с пустыми
  областью, классом и `quality_prob`, причина видна на сайте и в SR.
- В XLSX ещё листы `checks` (все проверки с порогами и пометкой, решает ли проверка) и
  `review`.

## API

Полное описание — в Swagger, кратко:

| Метод | Путь | Что делает |
|---|---|---|
| GET | `/health` | Сервис жив, какой процессор используется |
| POST | `/api/v1/studies/upload` | Загрузить DICOM или ZIP (поле `files`, до 500 файлов за запрос) |
| GET | `/api/v1/studies` | Список исследований |
| GET | `/api/v1/studies/{id}` | Одно исследование и его снимки |
| DELETE | `/api/v1/studies/{id}` | Удалить исследование вместе с файлами |
| POST | `/api/v1/studies/{id}/process` | Запустить обработку (повторная сохраняет решение врача) |
| GET | `/api/v1/studies/{id}/status` | Статус и прогресс |
| GET | `/api/v1/studies/{id}/result` | Результат в JSON, со всеми проверками |
| GET | `/api/v1/studies/{id}/download?format=csv\|xlsx&source=auto\|reviewed` | Скачать результат |
| GET | `/api/v1/studies/{id}/sr` | Заключение в виде DICOM SR |
| GET | `/api/v1/studies/{id}/package` | ZIP: SR + серия с разметкой + таблицы |
| GET | `/api/v1/studies/{id}/images/{image_id}/preview` | PNG-превью снимка |
| GET | `/api/v1/studies/{id}/images/{image_id}/overlay` | PNG с найденными структурами |
| POST | `/api/v1/studies/{id}/images/{image_id}/review` | Подтвердить, исправить или снять проверку врача |
| POST | `/api/v1/batch/process` | Пакетная обработка |
| GET | `/api/v1/batch/download?format=csv\|xlsx` | Сводный файл |

Статусы: `uploaded → queued → processing → completed | failed`. Пример:

```bash
curl -F "files=@samples/demo_studies.zip" http://localhost:8080/api/v1/studies/upload
curl -H 'Content-Type: application/json' -d '{"all_pending": true}' \
     http://localhost:8080/api/v1/batch/process
curl -o result.xlsx "http://localhost:8080/api/v1/batch/download?format=xlsx"
curl -H 'Content-Type: application/json' \
     -d '{"action":"correct","violation_type":["Не выравнена ось позвоночника"],"reviewed_by":"Иванов И.И."}' \
     "http://localhost:8080/api/v1/studies/<id>/images/<image_id>/review"
```

Проверить запущенный сервис целиком: `make smoke`.

## Настройки

Менять необязательно. Если нужно — `cp .env.example .env`, там всё с комментариями.
Главное:

| Переменная | По умолчанию | Что |
|---|---|---|
| `FRONTEND_PORT` | 8080 | Порт сайта |
| `BACKEND_PORT`, `BACKEND_BIND` | 8000, 127.0.0.1 | API напрямую — только с этой же машины |
| `SECOND_OPINION` | true | Справочная оценка ИИ-модели для бедра |
| `WORKER_CONCURRENCY` | 2 | Сколько исследований обрабатывается параллельно |
| `PROCESSING_TIMEOUT_SEC` | 180 | Лимит на исследование (ТЗ: 3 минуты) |
| `MAX_UPLOAD_SIZE_MB`, `MAX_FILES_PER_UPLOAD` | 1024, 500 | Лимиты одного запроса загрузки |
| `THRESHOLDS_PATH` | пусто | Свой файл порогов правил вместо встроенного |
| `PROCESSOR_BACKEND` | rulebased | `mock` — тестовые результаты без обработки |

## Пакет для сдачи

```bash
make package
```

В `dist/densitometry-ai-<версия>/`: образы Docker, `docker-compose.yml` без сборки из
исходников, `thresholds.json`, `.env.example`, документы, `INSTALL.txt` и контрольные
суммы. На целевой машине нужен только Docker:

```bash
docker load -i images.tar.gz
```
```bash
docker compose up -d
```

## Если сайт открывается, а запросы к API отдают 502

В журнале frontend при этом «backend could not be resolved» — в этой установке Docker
контейнеры не находят друг друга по имени. Проверить:

```bash
docker compose exec frontend sh -c 'cat /etc/resolv.conf; nslookup backend; wget -qO- http://backend:8000/health'
```

Если `nslookup backend` не отвечает, прописать в `.env` одно из:

```
BACKEND_URL=http://host.docker.internal:8000   # через хост; вместе с BACKEND_BIND=0.0.0.0
BACKEND_URL=http://<IP backend>:8000           # docker compose exec backend hostname -i
```

и выполнить `docker compose up -d --force-recreate frontend backend`.

## Разработка без Docker

Нужны Python 3.12 и Node.js 22.

```bash
make install        # зависимости
make dev-backend    # API на :8000
make dev-frontend   # сайт на :5173
make test           # тесты backend и ml
make test-docker    # тесты backend в контейнере
make lint           # линтер и проверка типов
make localization   # метрики локализации по ТЗ
```

Все команды — `make help`. Обучение и оценка моделей — раздел «Как воспроизвести» в
[OVERVIEW.md](OVERVIEW.md).

## Ограничения

- Результат — вспомогательный контроль качества, а не медицинское заключение.
- Входа по паролю нет, сервис рассчитан на закрытый контур. Снаружи открыт только сайт
  (порт 8080); всё, что он отдаёт, включая DICOM SR с данными пациента, доступно любому в
  этой сети. В поле «кто проверил» пишется то, что ввели: подпись врача не удостоверяется.
- В базу из DICOM попадают только технические теги, но загруженные файлы с данными
  пациента лежат в томе, пока исследование не удалено.
- SQLite и очередь внутри процесса рассчитаны на один экземпляр backend.
- Правила настроены на снимки GE Lunar Prodigy Advance (8 бит, известный размер
  пикселя); на другом аппарате их нужно перекалибровать.
