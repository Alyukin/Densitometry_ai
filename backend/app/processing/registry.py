"""Registry of available processors. Add the real model here when it is ready."""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable

from app.core.config import Settings, get_settings
from app.processing.base import BaseProcessor
from app.processing.mock import MockProcessor
from app.processing.rulebased import RuleBasedProcessor

logger = logging.getLogger(__name__)

ProcessorFactory = Callable[[Settings], BaseProcessor]

_REGISTRY: dict[str, ProcessorFactory] = {
    # Рабочий baseline: измерения по снимку + явные правила из ТЗ, без обучения.
    "rulebased": lambda s: RuleBasedProcessor(thresholds_path=s.thresholds_path or None),
    # Тестовые результаты без какой-либо обработки изображения.
    "mock": lambda s: MockProcessor(delay_per_image_sec=s.mock_delay_per_image_sec, seed=s.mock_seed),
    # Обучаемая модель подключается сюда же и сравнивается с baseline на тех же метриках:
    # "cnn": lambda s: DxaQualityModel(weights=s.model_weights_path, device=s.device),
}

_instance: BaseProcessor | None = None
_lock = threading.Lock()


def register_processor(name: str, factory: ProcessorFactory) -> None:
    _REGISTRY[name] = factory


def available_processors() -> list[str]:
    return sorted(_REGISTRY)


def get_processor() -> BaseProcessor:
    """Return a lazily created, loaded singleton processor."""
    global _instance
    if _instance is None:
        with _lock:
            if _instance is None:
                settings = get_settings()
                name = settings.processor_backend
                if name not in _REGISTRY:
                    raise RuntimeError(f"Unknown PROCESSOR_BACKEND={name!r}. Available: {available_processors()}")
                proc = _REGISTRY[name](settings)
                logger.info("Loading processor %s (%s)", proc.name, proc.version)
                proc.load()
                _instance = proc
    return _instance


def reset_processor() -> None:
    global _instance
    _instance = None
