import time
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.core.config import get_settings
from app.db.session import reset_engine
from app.processing.registry import reset_processor
from app.scripts.generate_samples import generate


@pytest.fixture(scope="session")
def samples(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("samples")
    generate(out)
    return out


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("DATABASE_URL", "")
    monkeypatch.setenv("MOCK_DELAY_PER_IMAGE_SEC", "0")
    monkeypatch.setenv("ENVIRONMENT", "test")
    get_settings.cache_clear()
    reset_engine()
    reset_processor()

    from app.main import create_app

    with TestClient(create_app()) as c:
        yield c

    reset_engine()
    reset_processor()
    get_settings.cache_clear()


def upload(client: TestClient, *paths: tuple[Path, str]):
    files = [("files", (name, p.read_bytes(), "application/dicom")) for p, name in paths]
    return client.post("/api/v1/studies/upload", files=files)


def wait_done(client: TestClient, study_id: str, timeout: float = 10.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        st = client.get(f"/api/v1/studies/{study_id}/status").json()
        if st["status"] in ("completed", "failed"):
            return st
        time.sleep(0.05)
    raise AssertionError(f"Study {study_id} did not finish: {st}")
