"""In-process background task runner (thread pool).

Интерфейс `submit(study_id)` намеренно минимален: при переходе на отдельный воркер
(Celery/RQ/Arq + GPU-контейнер) достаточно заменить реализацию этого класса.
"""

from __future__ import annotations

import logging
import threading
from concurrent.futures import ThreadPoolExecutor

from app.services.pipeline import mark_failed, process_study

logger = logging.getLogger(__name__)


class TaskRunner:
    def __init__(self, workers: int) -> None:
        self.workers = workers
        self._executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="study-worker")
        self._inflight: set[str] = set()
        self._lock = threading.Lock()

    def submit(self, study_id: str) -> bool:
        with self._lock:
            if study_id in self._inflight:
                return False
            self._inflight.add(study_id)
        self._executor.submit(self._run, study_id)
        return True

    def _run(self, study_id: str) -> None:
        try:
            process_study(study_id)
        except Exception:  # noqa: BLE001
            logger.exception("Unhandled error while processing %s", study_id)
            mark_failed(study_id, "Внутренняя ошибка сервиса обработки")
        finally:
            with self._lock:
                self._inflight.discard(study_id)

    @property
    def inflight(self) -> int:
        with self._lock:
            return len(self._inflight)

    def shutdown(self, wait: bool = False) -> None:
        self._executor.shutdown(wait=wait, cancel_futures=True)
