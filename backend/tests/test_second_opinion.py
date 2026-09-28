"""Второе мнение нейросети: справочно, только бедро, вердикт и выгрузку не трогает."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.processing import second_opinion as so
from app.processing.base import ImageInput
from app.processing.rulebased import RuleBasedProcessor
from tests.conftest import _make_client, upload, wait_done

FEMUR = "Проксимальный отдел бедра"
HAVE_MODEL = (so.MODEL_DIR / so.ONNX_NAME).exists()


def _task(threshold: float = 0.3, coef: float = 0.0) -> dict:
    return {
        "label": "есть нарушение",
        "region": FEMUR,
        "violation": "",
        "oof_roc_auc": 0.7,
        "threshold": threshold,
        "mean": [0.0, 0.0],
        "scale": [1.0, 1.0],
        "coef": [coef, 0.0],
        "intercept": float(np.log(threshold / (1 - threshold))),
    }


def test_preprocess_matches_training_scale() -> None:
    arr = np.zeros((280, 248), dtype=np.uint8)
    arr[:, 124:] = 255
    x = so.preprocess(arr, 384, 320)
    assert x.shape == (1, 1, 384, 320) and x.dtype == np.float32
    assert x.min() == pytest.approx(-1024.0) and x.max() == pytest.approx(1024.0)


def test_score_puts_the_model_threshold_at_one_half() -> None:
    assert so.score(np.zeros(2), _task(threshold=0.3)) == pytest.approx(0.5)
    assert so.score(np.array([1.0, 0.0]), _task(threshold=0.3, coef=2.0)) > 0.5


def test_check_is_reference_only() -> None:
    c = so.make_check("femur_quality", _task(), 0.8)
    assert c["decides"] is False and c["in_score"] is False
    assert c["fired"] is True and c["source"] == so.SOURCE
    assert "0.70" in c["criterion"]  # рядом с оценкой — её точность на отложенных данных


def test_missing_model_switches_it_off(tmp_path: Path) -> None:
    second = so.SecondOpinion(tmp_path)
    assert second.load() is False
    assert "нет файлов модели" in second.error


class _Fake:
    model = "подмена"

    def checks(self, arr, region):  # noqa: ANN001, ANN201
        return [so.make_check("femur_quality", _task(), 0.99)] if region == FEMUR else []


def _hip(samples: Path) -> ImageInput:
    f = next((samples / "study_hip_02_rotated").glob("*.dcm"))
    return ImageInput(image_id="x", path=f, original_filename=f.name, study_uid=None, image_uid=None)


def test_second_opinion_does_not_move_the_verdict(samples: Path) -> None:
    plain = RuleBasedProcessor(second_opinion=False)
    plain.load()
    with_nn = RuleBasedProcessor(second_opinion=False)
    with_nn.load()
    with_nn._second = _Fake()  # noqa: SLF001
    a, b = plain.predict(_hip(samples)), with_nn.predict(_hip(samples))
    assert (b.quality_class, b.violation_types, b.confidence) == (a.quality_class, a.violation_types, a.confidence)
    nn = [c for c in b.details["checks"] if c["source"] == so.SOURCE]
    assert len(nn) == 1 and nn[0]["decides"] is False
    assert b.details["second_opinion"]["checks"] == ["nn_femur_quality"]

    spine = next((samples / "study_spine_01").glob("*.dcm"))
    s = with_nn.predict(ImageInput("y", spine, spine.name, None, None))
    assert not [c for c in s.details["checks"] if c["source"] == so.SOURCE]


def test_failing_model_does_not_break_processing(samples: Path) -> None:
    class Broken(_Fake):
        def checks(self, arr, region):  # noqa: ANN001, ANN201
            raise RuntimeError("сбой")

    p = RuleBasedProcessor(second_opinion=False)
    p.load()
    p._second = Broken()  # noqa: SLF001
    pred = p.predict(_hip(samples))
    assert pred.anatomical_region == FEMUR
    assert "сбой" in pred.details["second_opinion_error"]


@pytest.mark.skipif(not HAVE_MODEL, reason="ONNX второго мнения не собран (собирается в Docker-образе)")
def test_real_model_through_the_api(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, samples: Path) -> None:
    pytest.importorskip("onnxruntime")
    gen = _make_client(tmp_path, monkeypatch, "rulebased", second_opinion=True)
    client: TestClient = next(gen)
    try:
        f = next((samples / "study_hip_02_rotated").glob("*.dcm"))
        sid = upload(client, (f, f.name)).json()["studies"][0]["id"]
        client.post(f"/api/v1/studies/{sid}/process")
        wait_done(client, sid)
        row = client.get(f"/api/v1/studies/{sid}/result").json()["rows"][0]
        nn = [c for c in row["details"]["checks"] if c["source"] == so.SOURCE]
        assert {c["rule_id"] for c in nn} == {"nn_femur_quality", "nn_femur_position", "nn_femur_roi"}
        assert all(0.0 <= c["value"] <= 1.0 and c["decides"] is False for c in nn)
        csv = client.get(f"/api/v1/studies/{sid}/download?format=csv").text
        assert "нейросет" not in csv  # в выгрузку по ТЗ второе мнение не попадает
    finally:
        gen.close()
