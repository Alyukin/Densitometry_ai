"""Очередь обработки: повторный запуск в момент завершения предыдущего не теряется."""

from __future__ import annotations

import threading

import pytest

from app.services import task_runner as tr


def test_submit_during_a_run_is_not_lost(monkeypatch: pytest.MonkeyPatch) -> None:
    started, release = threading.Event(), threading.Event()
    runs: list[str] = []
    done = threading.Semaphore(0)

    def fake_process(study_id: str) -> None:
        runs.append(study_id)
        if len(runs) == 1:
            started.set()
            release.wait(5)
        done.release()

    monkeypatch.setattr(tr, "process_study", fake_process)
    runner = tr.TaskRunner(1)
    try:
        assert runner.submit("s1") is True
        assert started.wait(5)
        # первая обработка ещё идёт: второй запуск откладывается, а не пропадает
        assert runner.submit("s1") is False
        release.set()
        assert done.acquire(timeout=5) and done.acquire(timeout=5)
        assert runs == ["s1", "s1"]
    finally:
        runner.shutdown(wait=True)
