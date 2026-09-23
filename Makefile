.DEFAULT_GOAL := help
SHELL := /bin/bash

COMPOSE ?= docker compose
PYTHON  ?= python3.12
VENV    := .venv
PY      := $(VENV)/bin/python

help: ## Показать список команд
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

.env:
	cp .env.example .env

# ---------- Docker ----------
up: .env ## Собрать и запустить сервис (http://localhost:8080)
	$(COMPOSE) up -d --build
	@echo ""
	@echo "  Web UI:   http://localhost:$${FRONTEND_PORT:-8080}"
	@echo "  Swagger:  http://localhost:$${FRONTEND_PORT:-8080}/docs"

down: ## Остановить сервис
	$(COMPOSE) down

logs: ## Логи контейнеров
	$(COMPOSE) logs -f --tail=200

ps: ## Статус контейнеров
	$(COMPOSE) ps

clean: ## Остановить и удалить данные (volume)
	$(COMPOSE) down -v

test-docker: ## Тесты backend внутри Docker
	docker build --target test -t densitometry-backend-test ./backend
	docker run --rm densitometry-backend-test

# ---------- Локальная разработка без Docker ----------
$(VENV):
	$(PYTHON) -m venv $(VENV)
	$(PY) -m pip install -r backend/requirements-dev.txt

install: $(VENV) ## Установить зависимости (Python venv + npm)
	cd frontend && npm ci

dev-backend: $(VENV) ## Backend с автоперезагрузкой на :8000
	cd backend && ../$(PY) -m uvicorn app.main:app --reload --port 8000

dev-frontend: ## Frontend (Vite) на :5173 с proxy на backend
	cd frontend && npm run dev

test: $(VENV) ## Тесты backend и метрик ТЗ
	cd backend && ../$(PY) -m pytest
	cd ml && ../$(PY) -m pytest

lint: $(VENV) ## Линтер backend + ml + typecheck frontend
	cd backend && ../$(VENV)/bin/ruff check app tests && ../$(VENV)/bin/ruff format --check app tests
	cd ml && ../$(VENV)/bin/ruff check . && ../$(VENV)/bin/ruff format --check .
	cd frontend && npm run typecheck

localization: $(VENV) ## Метрики локализации по ТЗ: Dice / IoU / keypoint distance
	$(PY) ml/baseline/localization.py

samples: $(VENV) ## Сгенерировать синтетические DICOM в ./samples
	cd backend && ../$(PY) -m app.scripts.generate_samples --out ../samples

smoke: ## Проверить запущенный сервис: загрузка → обработка → CSV
	bash ./scripts/smoke_test.sh

.PHONY: help up down logs ps clean test-docker install dev-backend dev-frontend test lint localization samples smoke
