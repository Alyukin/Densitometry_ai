"""Технические требования ТЗ: воспроизводимость и независимость от персональных данных.

Оба свойства легко потерять незаметно — от случайного `random` без зерна до признака,
подсмотренного в метаданных пациента. Поэтому они закреплены тестами, а не обещанием.
"""

from __future__ import annotations

from pathlib import Path

import pydicom
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text

from app.processing.base import ImageInput
from app.processing.rulebased import RuleBasedProcessor
from tests.conftest import upload, wait_done

# Теги, от которых решение не должно зависеть ни при каких условиях.
PERSONAL_TAGS = {
    "PatientName": "ПОДМЕНА^ТЕСТ^",
    "PatientID": "ZZZ-999999",
    "PatientBirthDate": "19000101",
    "PatientSex": "O",
    "PatientAge": "099Y",
    "AccessionNumber": "ACC-000",
    "InstitutionName": "Другая клиника",
    "ReferringPhysicianName": "Другой^Врач",
    "StudyDate": "19700101",
}


def _verdict(processor: RuleBasedProcessor, path: Path) -> dict:
    pred = processor.predict(
        ImageInput(
            image_id="x",
            path=path,
            original_filename=path.name,
            study_uid=None,
            image_uid=None,
        )
    )
    return {
        "region": pred.anatomical_region,
        "quality_class": pred.quality_class,
        "violations": list(pred.violation_types),
        "confidence": pred.confidence,
        "measurements": pred.details["measurements"],
        "checks": [(c["rule_id"], c["value"], c["fired"]) for c in pred.details["checks"]],
    }


@pytest.fixture(scope="module")
def processor() -> RuleBasedProcessor:
    p = RuleBasedProcessor()
    p.load()
    return p


def _all_samples(samples: Path) -> list[Path]:
    return sorted(p for p in samples.rglob("*.dcm") if "edge_cases" not in p.parts)


def test_same_file_twice_gives_bit_identical_verdict(processor: RuleBasedProcessor, samples: Path) -> None:
    for path in _all_samples(samples):
        first = _verdict(processor, path)
        second = _verdict(processor, path)
        assert first == second, f"результат не воспроизводится на {path.name}"


def test_fresh_processor_gives_the_same_verdict(samples: Path) -> None:
    """Состояние между вызовами не накапливается: новый процессор решает так же."""
    path = _all_samples(samples)[0]
    a = _verdict(RuleBasedProcessor(), path)
    b = _verdict(RuleBasedProcessor(), path)
    assert a == b


def test_verdict_does_not_depend_on_patient_data(processor: RuleBasedProcessor, samples: Path, tmp_path: Path) -> None:
    """Решение считается по пикселям; подмена персональных тегов ничего не меняет."""
    for path in _all_samples(samples):
        original = _verdict(processor, path)

        ds = pydicom.dcmread(path)
        ds.SpecificCharacterSet = "ISO_IR 192"  # чтобы кириллица в ФИО записалась как есть
        for tag, value in PERSONAL_TAGS.items():
            setattr(ds, tag, value)
        altered = tmp_path / f"altered_{path.parent.name}_{path.name}"
        ds.save_as(altered, enforce_file_format=True)

        assert _verdict(processor, altered) == original, f"вердикт изменился от метаданных: {path.name}"


def test_verdict_does_not_depend_on_file_name(processor: RuleBasedProcessor, samples: Path, tmp_path: Path) -> None:
    """Имена файлов в закрытом тесте будут другими (ответ заказчика на вопрос 15)."""
    path = _all_samples(samples)[0]
    renamed = tmp_path / "CR000000_совсем_другое_имя.dcm"
    renamed.write_bytes(path.read_bytes())
    assert _verdict(processor, renamed) == _verdict(processor, path)


def test_reprocessing_a_study_gives_the_same_rows(rb_client: TestClient, samples: Path) -> None:
    files = sorted((samples / "study_combined").glob("*.dcm"))
    sid = upload(rb_client, *[(f, f.name) for f in files]).json()["studies"][0]["id"]

    def run() -> list[tuple]:
        rb_client.post(f"/api/v1/studies/{sid}/process")
        assert wait_done(rb_client, sid)["status"] == "completed"
        rows = rb_client.get(f"/api/v1/studies/{sid}/result").json()["rows"]
        # время обработки естественным образом плавает, всё остальное — нет
        return [
            (r["image_uid"], r["anatomical_region"], r["quality_class"], r["violation_type"], r["confidence"])
            for r in rows
        ]

    assert run() == run()


# --- обновление схемы на уже существующей базе --------------------------------


def test_missing_columns_are_added_without_losing_data(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Обновление сервиса не должно стирать уже загруженные исследования."""
    from app.core.config import get_settings
    from app.db.session import get_engine, init_db, reset_engine

    db_path = tmp_path / "old.db"
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")
    get_settings.cache_clear()
    reset_engine()

    # база «старой версии»: та же таблица, но без колонок проверки специалистом
    old = create_engine(f"sqlite:///{db_path}")
    with old.begin() as conn:
        conn.execute(
            text(
                "CREATE TABLE image_results ("
                "id INTEGER PRIMARY KEY, study_id VARCHAR(32), image_id VARCHAR(32), "
                "path_to_study VARCHAR(1024), processing_status VARCHAR(32), time_of_processing FLOAT)"
            )
        )
        conn.execute(
            text(
                "INSERT INTO image_results (study_id, path_to_study, processing_status, time_of_processing) "
                "VALUES ('s1', '/data/old', 'success', 0.5)"
            )
        )
    old.dispose()

    init_db()

    with get_engine().connect() as conn:
        columns = {row[1] for row in conn.execute(text("PRAGMA table_info(image_results)"))}
        assert {"review_status", "reviewed_quality_class", "reviewed_by", "reviewed_at"} <= columns
        kept = conn.execute(text("SELECT path_to_study, processing_status FROM image_results")).fetchall()
    assert kept == [("/data/old", "success")]

    reset_engine()
    get_settings.cache_clear()
