import csv
import io
from pathlib import Path

from fastapi.testclient import TestClient
from openpyxl import load_workbook

from app.services.export import EXPORT_COLUMNS
from tests.conftest import upload, wait_done


def test_health(client: TestClient) -> None:
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["processor"] == "mock"
    assert body["is_mock"] is True
    assert body["database"] == "ok"


def test_openapi_has_required_endpoints(client: TestClient) -> None:
    paths = client.get("/openapi.json").json()["paths"]
    for p in [
        "/health",
        "/api/v1/studies/upload",
        "/api/v1/studies",
        "/api/v1/studies/{study_id}",
        "/api/v1/studies/{study_id}/process",
        "/api/v1/studies/{study_id}/status",
        "/api/v1/studies/{study_id}/result",
        "/api/v1/studies/{study_id}/download",
        "/api/v1/batch/process",
    ]:
        assert p in paths, p


def test_docs_pages(client: TestClient) -> None:
    r = client.get("/docs")
    assert r.status_code == 200
    assert "swagger-ui-bundle.js" in r.text
    r = client.get("/redoc")
    assert r.status_code == 200
    assert "redoc" in r.text.lower()


def test_upload_single_and_get(client: TestClient, samples: Path) -> None:
    r = upload(client, (samples / "study_spine_01/IM0001.dcm", "IM0001.dcm"))
    assert r.status_code == 201, r.text
    body = r.json()
    assert len(body["studies"]) == 1 and body["rejected"] == []
    study = body["studies"][0]
    assert study["status"] == "uploaded"
    assert study["image_count"] == 1
    assert study["modality"] == "BMD"

    detail = client.get(f"/api/v1/studies/{study['id']}").json()
    assert detail["images"][0]["body_part_examined"] == "LSPINE"
    assert detail["images"][0]["preview_url"]
    assert "PatientName" not in str(detail)

    listing = client.get("/api/v1/studies").json()
    assert listing["total"] == 1


def test_upload_groups_by_study_uid(client: TestClient, samples: Path) -> None:
    r = upload(
        client,
        (samples / "study_hip_01/IM0001.dcm", "study_hip_01/IM0001.dcm"),
        (samples / "study_hip_01/IM0002.dcm", "study_hip_01/IM0002.dcm"),
        (samples / "study_spine_01/IM0001.dcm", "study_spine_01/IM0001.dcm"),
    )
    assert r.status_code == 201
    studies = r.json()["studies"]
    assert sorted(s["image_count"] for s in studies) == [1, 2]
    hip = next(s for s in studies if s["image_count"] == 2)
    assert hip["name"] == "study_hip_01"
    assert hip["source_path"] == "study_hip_01"


def test_upload_zip(client: TestClient, samples: Path) -> None:
    r = upload(client, (samples / "demo_studies.zip", "demo_studies.zip"))
    assert r.status_code == 201, r.text
    studies = r.json()["studies"]
    assert len(studies) == 7
    assert sum(s["image_count"] for s in studies) == 10


def test_upload_rejects_non_dicom(client: TestClient, samples: Path) -> None:
    r = upload(client, (samples / "edge_cases/not_a_dicom.dcm", "not_a_dicom.dcm"))
    assert r.status_code == 422
    r = upload(
        client,
        (samples / "edge_cases/not_a_dicom.dcm", "not_a_dicom.dcm"),
        (samples / "study_spine_01/IM0001.dcm", "IM0001.dcm"),
    )
    assert r.status_code == 201
    assert len(r.json()["studies"]) == 1
    assert r.json()["rejected"][0]["filename"] == "not_a_dicom.dcm"


def test_upload_duplicate_sop_rejected(client: TestClient, samples: Path) -> None:
    p = samples / "study_spine_01/IM0001.dcm"
    r = upload(client, (p, "a.dcm"), (p, "b.dcm"))
    assert r.status_code == 201
    assert r.json()["studies"][0]["image_count"] == 1
    assert "Дубликат" in r.json()["rejected"][0]["reason"]


def test_full_processing_flow(client: TestClient, samples: Path) -> None:
    files = [(samples / f"study_combined/IM000{i}.dcm", f"study_combined/IM000{i}.dcm") for i in (1, 2, 3)]
    sid = upload(client, *files).json()["studies"][0]["id"]

    assert client.get(f"/api/v1/studies/{sid}/result").status_code == 409
    assert client.get(f"/api/v1/studies/{sid}/download").status_code == 409

    r = client.post(f"/api/v1/studies/{sid}/process")
    assert r.status_code == 202
    assert r.json()["status"] == "queued"

    st = wait_done(client, sid)
    assert st["status"] == "completed"
    assert st["progress"] == 100
    assert st["processed_images"] == st["total_images"] == 3

    res = client.get(f"/api/v1/studies/{sid}/result").json()
    assert res["is_mock"] is True
    assert len(res["rows"]) == 3
    regions = {row["anatomical_region"] for row in res["rows"]}
    assert regions == {"lumbar_spine", "proximal_femur"}
    for row in res["rows"]:
        assert row["processing_status"] == "success"
        assert row["quality_class"] in ("acceptable", "unacceptable")
        assert (row["quality_class"] == "unacceptable") == bool(row["violation_type"])
        assert row["details"]["checks"]

    csv_resp = client.get(f"/api/v1/studies/{sid}/download?format=csv")
    assert csv_resp.status_code == 200
    assert "attachment" in csv_resp.headers["content-disposition"]
    rows = list(csv.reader(io.StringIO(csv_resp.text)))
    assert rows[0] == EXPORT_COLUMNS
    assert len(rows) == 4  # шапка + 3 снимка исследования

    xlsx_resp = client.get(f"/api/v1/studies/{sid}/download?format=xlsx")
    assert xlsx_resp.status_code == 200
    wb = load_workbook(io.BytesIO(xlsx_resp.content))
    header = [c.value for c in wb["results"][1]]
    assert header == EXPORT_COLUMNS
    assert wb["results"].max_row == 4

    assert client.get(f"/api/v1/studies/{sid}/download?format=pdf").status_code == 422


def test_results_are_reproducible(client: TestClient, samples: Path) -> None:
    sid = upload(client, (samples / "demo_studies.zip", "demo.zip")).json()["studies"][0]["id"]
    client.post(f"/api/v1/studies/{sid}/process")
    wait_done(client, sid)
    first = client.get(f"/api/v1/studies/{sid}/result").json()["rows"]
    client.post(f"/api/v1/studies/{sid}/process")
    wait_done(client, sid)
    second = client.get(f"/api/v1/studies/{sid}/result").json()["rows"]
    key = ("image_uid", "anatomical_region", "quality_class", "violation_type")
    assert [[r[k] for k in key] for r in first] == [[r[k] for k in key] for r in second]


def test_image_without_pixels_fails(client: TestClient, samples: Path) -> None:
    sid = upload(client, (samples / "edge_cases/no_pixel_data.dcm", "nopix.dcm")).json()["studies"][0]["id"]
    client.post(f"/api/v1/studies/{sid}/process")
    st = wait_done(client, sid)
    assert st["status"] == "failed"
    res = client.get(f"/api/v1/studies/{sid}/result").json()
    assert res["rows"][0]["processing_status"] == "error"
    assert res["rows"][0]["error_message"]


def test_batch_process_and_download(client: TestClient, samples: Path) -> None:
    ids = [s["id"] for s in upload(client, (samples / "demo_studies.zip", "demo.zip")).json()["studies"]]

    assert client.post("/api/v1/batch/process", json={}).status_code == 400
    assert client.get("/api/v1/batch/download").status_code == 409

    r = client.post("/api/v1/batch/process", json={"study_ids": [ids[0], "missing"]})
    assert r.status_code == 202
    assert r.json()["accepted"] == [ids[0]]
    assert r.json()["skipped"][0] == {"study_id": "missing", "reason": "not_found"}
    wait_done(client, ids[0])

    r = client.post("/api/v1/batch/process", json={"all_pending": True})
    assert sorted(r.json()["accepted"]) == sorted(ids[1:])
    for sid in ids:
        assert wait_done(client, sid)["status"] == "completed"

    r = client.post("/api/v1/batch/process", json={"study_ids": ids})
    assert r.json()["accepted"] == []
    assert {s["reason"] for s in r.json()["skipped"]} == {"already_completed"}

    r = client.get("/api/v1/batch/download", params={"format": "csv"})
    assert r.status_code == 200
    assert len(r.text.strip().splitlines()) == 1 + 10

    r = client.get("/api/v1/batch/download", params={"format": "xlsx", "study_ids": ids[:1]})
    assert r.status_code == 200

    listing = client.get("/api/v1/studies", params={"status": "completed"}).json()
    assert listing["total"] == 7
    assert listing["items"][0]["summary"]["total"] >= 1


def test_preview_and_delete(client: TestClient, samples: Path) -> None:
    study = upload(client, (samples / "study_spine_01/IM0001.dcm", "IM0001.dcm")).json()["studies"][0]
    detail = client.get(f"/api/v1/studies/{study['id']}").json()
    r = client.get(detail["images"][0]["preview_url"])
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/png"
    assert r.content[:8] == b"\x89PNG\r\n\x1a\n"

    assert client.delete(f"/api/v1/studies/{study['id']}").status_code == 204
    assert client.get(f"/api/v1/studies/{study['id']}").status_code == 404
    assert client.post(f"/api/v1/studies/{study['id']}/process").status_code == 404
