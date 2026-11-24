"""HTTP contract tests use a stub predictor and isolated temporary storage only."""
from contextlib import closing
import asyncio
import io
import json
import sqlite3
import httpx

from fastapi.testclient import TestClient
from PIL import Image
import pytest

from app.app import MAX_FILE_BYTES, create_app, load_metrics


class StubPredictor:
    default = "plantcnn"
    unavailable = {}

    def __init__(self):
        self.models = {"plantcnn": {"title": "Test CNN", "size": 128,
            "version": "sha256:test-checkpoint", "checkpoint_sha256": "test-checkpoint",
            "classes": ["Apple___healthy", "Apple___Black_rot"], "params": 10}}
        self.last_image = None

    def predict(self, image, model=None, topk=5, cam=True):
        self.last_image = image.copy()
        return {"model": "plantcnn", "model_title": "Test CNN", "model_version": "sha256:test-checkpoint",
                "time_ms": 1.25, "timings_ms": {"total": 1.25, "inference": 0.5},
                "top": [{"cls": "Apple___healthy", "prob": 0.8, "index": 0},
                        {"cls": "Apple___Black_rot", "prob": 0.2, "index": 1}],
                "heatmap": "data:image/png;base64,dGVzdA==" if cam else None}


def image_bytes(fmt="PNG", size=(20, 30), exif=None):
    buf = io.BytesIO()
    image = Image.new("RGB", size, (30, 180, 40))
    image.save(buf, format=fmt, **({"exif": exif} if exif else {}))
    return buf.getvalue()


@pytest.fixture
def client(tmp_path):
    pr = StubPredictor()
    web = tmp_path / "dist"
    (web / "assets").mkdir(parents=True)
    (web / "index.html").write_text("<html>LeafCare test SPA</html>", encoding="utf8")
    api = create_app(predictor=pr, db_path=tmp_path / "history.db", uploads_dir=tmp_path / "uploads",
                     reports_dir=tmp_path / "reports", frontend_dir=web)
    with TestClient(api) as test_client:
        yield test_client


def post_image(client, **data):
    return client.post("/predict", files={"image": ("leaf.png", image_bytes(), "image/png")}, data=data)


def test_contract_and_no_persistence_by_default(client):
    response = post_image(client, include_heatmap="false")
    assert response.status_code == 200
    body = response.json()
    assert body["request_id"] == response.headers["X-Request-ID"]
    assert body["predicted_class"] == "Apple___healthy"
    assert body["model_version"] == "sha256:test-checkpoint"
    assert body["confidence"] == 0.8 and body["top_k"][0]["confidence"] == 0.8
    assert body["calibrated"] is False and body["warning"]
    assert body["saved_to_history"] is False and body["heatmap"] is None
    assert body["image_url"] is None
    assert not list(client.app.state.uploads_dir.iterdir())
    assert client.get("/api/history").json()["stats"]["total"] == 0
    assert post_image(client).json()["request_id"] != body["request_id"]


def test_opt_in_history_and_delete_lifecycle(client):
    body = post_image(client, save_history="true").json()
    assert body["saved_to_history"] is True
    assert client.get(body["image_url"]).status_code == 200
    history = client.get("/api/history").json()
    assert history["stats"] == {"total": 1, "healthy": 1, "affected": 0,
                                "average_confidence": 0.8, "average_time_ms": 1.25}
    assert history["items"][0]["model_version"] == body["model_version"]
    assert client.delete(f"/api/history/{body['history_id']}").status_code == 200
    assert client.get(body["image_url"]).status_code == 404
    assert client.get("/api/history").json()["stats"]["total"] == 0
    assert client.delete(f"/api/history/{body['history_id']}").status_code == 404


@pytest.mark.parametrize("name,mime,raw,status,code", [
    ("leaf.png", "image/png", b"", 400, "EMPTY_FILE"),
    ("leaf.txt", "text/plain", b"hello", 415, "UNSUPPORTED_MEDIA_TYPE"),
    ("leaf.png", "image/jpeg", b"bad", 415, "UNSUPPORTED_MEDIA_TYPE"),
    ("leaf.png", "image/png", b"not a PNG", 422, "IMAGE_DECODE_FAILED"),
    ("leaf.jpg", "image/jpeg", image_bytes(), 415, "UNSUPPORTED_MEDIA_TYPE"),
    ("leaf.png", "image/png", b"x" * (MAX_FILE_BYTES + 1), 413, "FILE_TOO_LARGE"),
])
def test_invalid_uploads_are_structured_and_never_saved(client, name, mime, raw, status, code):
    result = client.post("/predict", files={"image": (name, raw, mime)}, data={"save_history": "true"})
    assert result.status_code == status
    assert result.json()["error_code"] == code
    assert result.json()["request_id"] == result.headers["X-Request-ID"]
    assert not list(client.app.state.uploads_dir.iterdir())
    assert client.get("/api/history").json()["stats"]["total"] == 0


def test_no_file_and_oversized_pixels(client, monkeypatch):
    assert client.post("/predict").json()["error_code"] == "EMPTY_FILE"
    monkeypatch.setattr("app.app.MAX_PIXELS", 100)
    result = post_image(client)
    assert result.status_code == 413 and result.json()["error_code"] == "PIXEL_LIMIT_EXCEEDED"


@pytest.mark.parametrize("declared_length", [None, "1"])
def test_chunked_body_limit_stops_reading_early(client, declared_length):
    chunks_read = 0

    async def run():
        async def chunks():
            nonlocal chunks_read
            yield b'--leafcare\r\nContent-Disposition: form-data; name="image"; filename="leaf.png"\r\nContent-Type: image/png\r\n\r\n'
            for _ in range(20):
                chunks_read += 1
                yield b"x" * (1024 * 1024)
            yield b"\r\n--leafcare--\r\n"
        headers = {"Content-Type": "multipart/form-data; boundary=leafcare"}
        if declared_length:
            headers["Content-Length"] = declared_length
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=client.app), base_url="http://test") as async_client:
            return await async_client.post("/predict", content=chunks(), headers=headers)

    response = asyncio.run(run())
    assert response.status_code == 413 and response.json()["error_code"] == "FILE_TOO_LARGE"
    assert chunks_read < 20
    assert not list(client.app.state.uploads_dir.iterdir())


def test_exif_orientation_and_metadata_removal(client):
    exif = Image.Exif()
    exif[274] = 6
    exif[270] = "private description"
    raw = image_bytes("JPEG", size=(20, 30), exif=exif)
    result = client.post("/predict", files={"image": ("photo.jpg", raw, "image/jpeg")}, data={"save_history": "true"})
    assert result.status_code == 200
    assert client.app.state.predictor.last_image.size == (30, 20)
    saved = Image.open(io.BytesIO(client.get(result.json()["image_url"]).content))
    assert saved.size == (30, 20) and not saved.getexif()


def test_missing_models_and_explicit_invalid_model(client):
    response = post_image(client, model="not-installed")
    assert response.status_code == 503 and response.json()["error_code"] == "MODEL_UNAVAILABLE"
    client.app.state.predictor.models = {}
    assert client.get("/health").status_code == 503
    assert client.get("/health").json()["model_loaded"] is False
    assert client.get("/model-info").status_code == 503
    assert post_image(client).status_code == 503


def test_model_information_and_openapi_schema(client):
    assert client.get("/health").json()["model_loaded"] is True
    metadata = client.get("/model-info").json()
    assert metadata["models"][0]["classes"][0]["index"] == 0
    assert metadata["models"][0]["metrics"] is None
    assert metadata["calibrated"] is False and metadata["retention"]["default_save"] is False
    schema = client.get("/openapi.json").json()
    prediction_schema = schema["components"]["schemas"]["PredictionResponse"]
    assert {"request_id", "predicted_class", "confidence", "model_version", "warning"} <= set(prediction_schema["required"])
    assert schema["paths"]["/predict"]["post"]["responses"]["415"]["content"]["application/json"]["schema"]["$ref"].endswith("ErrorResponse")
    for route in ("/", "/history", "/diseases", "/about"):
        assert "LeafCare test SPA" in client.get(route).text


def test_history_aggregates_cover_all_pages(client):
    with closing(sqlite3.connect(client.app.state.db_path)) as db:
        db.executemany("INSERT INTO predictions(created_at,filename,model,cls,confidence,time_ms) VALUES (?,?,?,?,?,?)",
            [("2026-01-01T00:00:00+00:00", f"{i:012x}.jpg", "plantcnn", "Apple___healthy" if i < 220 else "Apple___Black_rot", .8, 2.)
             for i in range(250)])
        db.commit()
    result = client.get("/api/history?limit=50&offset=0").json()
    assert len(result["items"]) == 50 and result["has_more"] is True
    assert result["stats"]["total"] == 250
    assert result["stats"]["healthy"] == 220 and result["stats"]["affected"] == 30
    assert client.get("/api/history?limit=50&offset=200").json()["has_more"] is False


def test_delete_cannot_remove_files_outside_uploads(client, tmp_path):
    outside = tmp_path / "important.txt"
    outside.write_text("keep", encoding="utf8")
    with closing(sqlite3.connect(client.app.state.db_path)) as db:
        cursor = db.execute("INSERT INTO predictions(created_at,filename,model,cls,confidence,time_ms) VALUES (?,?,?,?,?,?)",
                            ("2026", "../important.txt", "plantcnn", "Apple___healthy", .8, 2.))
        db.commit()
        pid = cursor.lastrowid
    assert client.delete(f"/api/history/{pid}").status_code == 200
    assert outside.read_text(encoding="utf8") == "keep"


def test_only_matching_evaluation_is_reported_as_verified(tmp_path):
    evaluation = tmp_path / "evaluation_current"
    evaluation.mkdir()
    (evaluation / "plantcnn_evaluation.json").write_text(json.dumps({"checkpoint_sha256": "abc", "test_acc": .25}))
    (tmp_path / "plantcnn_results.json").write_text(json.dumps({"test_acc": .9}))
    assert load_metrics(tmp_path, "plantcnn", "abc")["provenance"] == "verified_current_evaluation"
    assert load_metrics(tmp_path, "plantcnn", "different")["provenance"] == "legacy_unverified"
