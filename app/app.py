"""LeafCare AI FastAPI service. Run locally with ``python -m app.app``."""
from __future__ import annotations
from contextlib import asynccontextmanager, closing
from datetime import datetime, timezone
import io
import json
import logging
from pathlib import Path
import re
import sqlite3
import time
import uuid
import warnings
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps, UnidentifiedImageError
from pydantic import BaseModel, ConfigDict, Field
from starlette.concurrency import run_in_threadpool
from starlette.exceptions import HTTPException
from starlette.types import ASGIApp, Receive, Scope, Send
from app.disease_info import INFO, get_info
from app.predictor import ModelUnavailableError, Predictor

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "history.db"
UPLOADS = ROOT / "uploads"
REPORTS = ROOT / "reports"
VERSION = "2.0.0"
MAX_FILE_BYTES = 10 * 1024 * 1024
MAX_PIXELS = 25_000_000
LOW_CONF = 0.50  # Provisional UI threshold, not a calibrated rejection rule.
WARNING = "Бұл ықтимал класс, агрономиялық диагноз емес. Softmax мәні калибрленбеген; нәтижені маманмен тексеріңіз."
RETENTION = {"default_save": False, "policy": "explicit_opt_in_until_deleted",
             "description": "Сурет пен нәтиже тек келісіммен жергілікті тарихқа сақталады және қолмен жойылғанға дейін қалады."}
LOGGER = logging.getLogger("leafcare.requests")


class ErrorResponse(BaseModel):
    error: str
    error_code: str
    request_id: str


class TopKPrediction(BaseModel):
    class_label: str = Field(alias="class")
    label: str
    display_name_kk: str
    confidence: float = Field(ge=0, le=1)


class PredictionResponse(BaseModel):
    model_config = ConfigDict(extra="allow")
    request_id: str
    predicted_class: str
    display_name_kk: str
    confidence: float = Field(ge=0, le=1)
    top_k: list[TopKPrediction]
    model_version: str
    warning: str
    calibrated: bool
    saved_to_history: bool
    low_confidence: bool
    model: str
    model_title: str
    time_ms: float
    timings_ms: dict[str, float]
    heatmap: str | None
    image_url: str | None = None


PREDICT_ERRORS = {status: {"model": ErrorResponse, "description": description} for status, description in (
    (400, "Empty image or invalid parameters"), (413, "File or image dimensions exceed limit"),
    (415, "Extension, MIME or decoded format is unsupported"), (422, "Image decoding failed"),
    (503, "Requested model unavailable"), (500, "Internal processing error"))}


class PredictionBodyLimit:
    """Bound actual streamed multipart bytes, including absent/false Content-Length."""
    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send):
        if scope["type"] != "http" or scope.get("path") not in ("/predict", "/api/predict"):
            return await self.app(scope, receive, send)
        received = 0

        async def bounded_receive():
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                # Permit a small multipart envelope; each actual file is checked separately.
                if received > MAX_FILE_BYTES + 65536:
                    raise HTTPException(status_code=413, detail="Upload body exceeds limit")
            return message

        await self.app(scope, bounded_receive, send)


class APIError(Exception):
    def __init__(self, status, code, message):
        self.status, self.code, self.message = status, code, message


def connect(db_path):
    connection = sqlite3.connect(str(db_path), timeout=20)
    connection.row_factory = sqlite3.Row
    return connection


def init_db(db_path):
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    with closing(connect(db_path)) as db:
        db.execute("""CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT NOT NULL,
            filename TEXT NOT NULL, model TEXT NOT NULL, cls TEXT NOT NULL,
            confidence REAL NOT NULL, time_ms REAL NOT NULL)""")
        existing = {r[1] for r in db.execute("PRAGMA table_info(predictions)")}
        for column in ("model_version", "request_id"):
            if column not in existing:
                db.execute(f"ALTER TABLE predictions ADD COLUMN {column} TEXT")
        db.commit()


def decode_image(raw, filename, content_type):
    if not raw:
        raise APIError(400, "EMPTY_FILE", "Сурет бос. JPG немесе PNG файлын таңдаңыз.")
    if len(raw) > MAX_FILE_BYTES:
        raise APIError(413, "FILE_TOO_LARGE", "Файл көлемі 10 МБ-тан аспауы керек.")
    ext = Path(filename or "").suffix.lower()
    expected = {".jpg": ("image/jpeg", "JPEG"), ".jpeg": ("image/jpeg", "JPEG"), ".png": ("image/png", "PNG")}
    if ext not in expected or content_type != expected[ext][0]:
        raise APIError(415, "UNSUPPORTED_MEDIA_TYPE", "Тек JPG, JPEG және PNG суреттері қабылданады. Файл түрін тексеріңіз.")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(raw)) as source:
                if source.format != expected[ext][1]:
                    raise APIError(415, "UNSUPPORTED_MEDIA_TYPE", "Файл кеңейтімі мен нақты сурет форматы сәйкес емес.")
                if source.width * source.height > MAX_PIXELS:
                    raise APIError(413, "PIXEL_LIMIT_EXCEEDED", "Сурет 25 миллион пиксельден аспауы керек. Өлшемін кішірейтіңіз.")
                source.load()
                oriented = ImageOps.exif_transpose(source).convert("RGB")
                clean = Image.new("RGB", oriented.size)
                clean.paste(oriented)
                return clean
    except (Image.DecompressionBombError, Image.DecompressionBombWarning):
        raise APIError(413, "PIXEL_LIMIT_EXCEEDED", "Сурет өлшемі тым үлкен. Өлшемін кішірейтіңіз.")
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError):
        raise APIError(422, "IMAGE_DECODE_FAILED", "Суретті оқу мүмкін болмады. Бүтін JPG немесе PNG файлын таңдаңыз.")


def load_metrics(reports_dir, model, checkpoint_sha256=None):
    verified = Path(reports_dir) / "evaluation_current" / f"{model}_evaluation.json"
    legacy = Path(reports_dir) / f"{model}_results.json"
    keys = ("test_acc", "test_top3", "f1_macro", "precision_macro", "recall_macro", "params",
            "infer_ms_cpu", "img_size", "epochs", "model_size_mb", "train_samples", "val_samples", "test_samples",
            "latency_ms", "checkpoint_sha256", "dataset_manifest_sha256", "split_policy",
            "split_reconstruction", "limitations", "training_history_status", "calibrated", "external_evaluation")
    for path in (verified, legacy):
        if not path.is_file():
            continue
        try:
            result = json.loads(path.read_text(encoding="utf8"))
            if path == verified and (not checkpoint_sha256 or result.get("checkpoint_sha256") != checkpoint_sha256):
                continue
            return {**{k: result[k] for k in keys if k in result},
                    "provenance": "verified_current_evaluation" if path == verified else "legacy_unverified",
                    "note": ("Ағымдағы checkpoint хэшімен байланыстырылған өлшем. Деректер аудитінің шектеулерін ескеріңіз."
                             if path == verified else "Алдыңғы эксперимент нәтижесі; checkpoint хэшімен және leakage аудитімен байланысы расталмаған.")}
        except (ValueError, OSError, TypeError):
            continue
    return None


def create_app(predictor=None, db_path=DB_PATH, uploads_dir=UPLOADS, reports_dir=REPORTS, frontend_dir=None):
    db_path, uploads_dir, reports_dir = Path(db_path), Path(uploads_dir), Path(reports_dir)
    frontend_dir = Path(frontend_dir or ROOT / "frontend" / "dist")

    @asynccontextmanager
    async def lifespan(api):
        uploads_dir.mkdir(parents=True, exist_ok=True)
        init_db(db_path)
        if api.state.predictor is None:
            api.state.predictor = await run_in_threadpool(Predictor)
        yield

    api = FastAPI(title="LeafCare AI", version=VERSION, lifespan=lifespan)
    api.add_middleware(PredictionBodyLimit)
    api.state.predictor = predictor
    api.state.db_path = db_path
    api.state.uploads_dir = uploads_dir

    def error_response(request, status, code, message):
        request.state.error_code = code
        return JSONResponse({"error": message, "error_code": code,
                             "request_id": request.state.request_id}, status_code=status)

    @api.middleware("http")
    async def request_context(request: Request, call_next):
        request.state.request_id = uuid.uuid4().hex
        request.state.error_code = None
        request.state.model_version = None
        start = time.perf_counter()
        try:
            length = request.headers.get("content-length", "")
            if request.url.path in ("/predict", "/api/predict") and length.isdigit() and int(length) > MAX_FILE_BYTES + 65536:
                response = error_response(request, 413, "FILE_TOO_LARGE", "Файл көлемі 10 МБ-тан аспауы керек.")
            else:
                response = await call_next(request)
        except Exception as exc:
            LOGGER.error(json.dumps({"event": "request_failed", "request_id": request.state.request_id,
                                     "error_type": type(exc).__name__}))
            response = error_response(request, 500, "INTERNAL_ERROR", "Сервер қатесі. Сәлден кейін қайталап көріңіз.")
        response.headers["X-Request-ID"] = request.state.request_id
        response.headers["X-Content-Type-Options"] = "nosniff"
        LOGGER.info(json.dumps({"request_id": request.state.request_id, "model_version": request.state.model_version,
                                "latency_ms": round((time.perf_counter() - start) * 1000, 2),
                                "status": response.status_code, "error_code": request.state.error_code}))
        return response

    @api.exception_handler(APIError)
    async def api_error(request, exc):
        return error_response(request, exc.status, exc.code, exc.message)

    @api.exception_handler(RequestValidationError)
    async def invalid_request(request, exc):
        return error_response(request, 400, "INVALID_REQUEST", "Сұрау параметрлері қате. Форманы тексеріңіз.")

    @api.exception_handler(HTTPException)
    async def http_error(request, exc):
        if exc.status_code == 413:
            return error_response(request, 413, "FILE_TOO_LARGE", "Файл көлемі 10 МБ-тан аспауы керек.")
        return error_response(request, exc.status_code, "NOT_FOUND" if exc.status_code == 404 else "INVALID_REQUEST",
                              "Сұралған ресурс табылмады." if exc.status_code == 404 else "Сұрауды орындау мүмкін болмады.")

    @api.get("/health")
    def health():
        pr = api.state.predictor
        available = bool(pr and pr.models)
        unavailable = getattr(pr, "unavailable", {}) if pr else {}
        return JSONResponse({"status": ("degraded" if unavailable else "ok") if available else "unavailable",
                             "model_loaded": available, "version": VERSION,
                             "models": list(pr.models) if pr else [], "unavailable_models": unavailable},
                            status_code=200 if available else 503)

    @api.get("/model-info")
    def model_info():
        pr = api.state.predictor
        if not pr or not pr.models:
            raise APIError(503, "MODEL_UNAVAILABLE", "Модель жүктелмеген. Сервердегі checkpoint файлдарын тексеріңіз.")
        models = []
        for name, spec in pr.models.items():
            metrics = load_metrics(reports_dir, name, spec.get("checkpoint_sha256"))
            models.append({"id": name, "title": spec["title"], "version": spec["version"],
                           "input_size": [spec["size"], spec["size"]], "params": spec.get("params"),
                           "classes": [{"index": i, "label": cls, **{k: get_info(cls)[k] for k in ("plant", "name", "healthy")}}
                                       for i, cls in enumerate(spec["classes"])],
                           "preprocessing": spec.get("preprocessing"),
                           "preprocessing_provenance": spec.get("preprocessing_provenance", "unknown"),
                           "metrics": metrics, "metrics_provenance": metrics["provenance"] if metrics else None})
        default = next(m for m in models if m["id"] == pr.default)
        return {"default_model": pr.default, "models": models, "unavailable_models": getattr(pr, "unavailable", {}),
                "classes": default["classes"], "input_size": default["input_size"], "version": default["version"],
                "warning": WARNING, "retention": RETENTION, "calibrated": False,
                "confidence_threshold": LOW_CONF, "threshold_status": "provisional_not_calibrated", "ood_detection": False}

    @api.get("/api/diseases")
    def diseases():
        return {"items": [{"cls": cls, **get_info(cls)} for cls in INFO]}

    @api.get("/api/history")
    def history(limit: int = 50, offset: int = 0):
        if not 1 <= limit <= 200 or offset < 0:
            raise APIError(400, "INVALID_REQUEST", "Тарих параметрлері қате.")
        with closing(connect(db_path)) as db:
            rows = db.execute("SELECT * FROM predictions ORDER BY id DESC LIMIT ? OFFSET ?", (limit, offset)).fetchall()
            stats = dict(db.execute("SELECT COUNT(*) total, AVG(confidence) average_confidence, AVG(time_ms) average_time_ms FROM predictions").fetchone())
            healthy = sum(r["n"] for r in db.execute("SELECT cls, COUNT(*) n FROM predictions GROUP BY cls") if get_info(r["cls"])["healthy"])
        stats.update(healthy=healthy, affected=stats["total"] - healthy)
        stats["average_confidence"] = stats["average_confidence"] or 0.0
        stats["average_time_ms"] = stats["average_time_ms"] or 0.0
        items = [{**dict(r), "info": get_info(r["cls"]), "image_url": f"/uploads/{r['filename']}"} for r in rows]
        return {"items": items, "stats": stats, "limit": limit, "offset": offset,
                "has_more": offset + len(items) < stats["total"], "retention": RETENTION}

    @api.delete("/api/history/{prediction_id}")
    def delete_history(prediction_id: int):
        with closing(connect(db_path)) as db:
            row = db.execute("SELECT filename FROM predictions WHERE id=?", (prediction_id,)).fetchone()
            if row is None:
                raise APIError(404, "NOT_FOUND", "Жазба табылмады.")
            filename = row["filename"]
            db.execute("DELETE FROM predictions WHERE id=?", (prediction_id,))
            referenced = db.execute("SELECT 1 FROM predictions WHERE filename=? LIMIT 1", (filename,)).fetchone()
            db.commit()
        if not referenced and re.fullmatch(r"[0-9a-f]{12,32}\.(?:jpg|jpeg|png|webp|bmp)", filename):
            target = uploads_dir / filename
            if not target.is_symlink() and target.resolve().parent == uploads_dir.resolve():
                target.unlink(missing_ok=True)
        return {"ok": True, "deleted_id": prediction_id}

    @api.post("/predict", response_model=PredictionResponse, responses=PREDICT_ERRORS)
    @api.post("/api/predict", response_model=PredictionResponse, responses=PREDICT_ERRORS)
    async def predict(request: Request, image: UploadFile | None = File(default=None),
                      model: str | None = Form(default=None), save_history: bool = Form(default=False),
                      include_heatmap: bool = Form(default=True)):
        started = time.perf_counter()
        if image is None or not image.filename:
            raise APIError(400, "EMPTY_FILE", "Суретті таңдаңыз.")
        try:
            raw = await image.read(MAX_FILE_BYTES + 1)
            decoded_start = time.perf_counter()
            clean = await run_in_threadpool(decode_image, raw, image.filename, image.content_type)
            decode_ms = (time.perf_counter() - decoded_start) * 1000
        finally:
            await image.close()
        pr = api.state.predictor
        if not pr or not pr.models or (model and model not in pr.models):
            raise APIError(503, "MODEL_UNAVAILABLE", "Сұралған модель қолжетімсіз. Қолжетімді модельді таңдаңыз.")
        try:
            result = await run_in_threadpool(pr.predict, clean, model, 5, include_heatmap)
        except ModelUnavailableError as exc:
            raise APIError(503, "MODEL_UNAVAILABLE", str(exc))
        request.state.model_version = result["model_version"]
        for item in result["top"]:
            info = get_info(item["cls"])
            item.update({k: info[k] for k in ("plant", "name", "healthy")})
        best = result["top"][0]
        low = best["prob"] < LOW_CONF
        result.update(request_id=request.state.request_id, predicted_class=best["cls"],
                      display_name_kk=f"{best['plant']} — {best['name']}", confidence=best["prob"],
                      top_k=[{"class": t["cls"], "label": t["cls"], "confidence": t["prob"],
                              "display_name_kk": f"{t['plant']} — {t['name']}"} for t in result["top"]],
                      info=get_info(best["cls"]), warning=("Сенімді жіктеу мүмкін болмады. " if low else "") + WARNING,
                      low_confidence=low, calibrated=False, threshold_status="provisional_not_calibrated",
                      confidence_threshold=LOW_CONF, ood_detection=False, saved_to_history=False)
        if save_history:
            filename = uuid.uuid4().hex + ".jpg"
            destination = uploads_dir / filename
            try:
                await run_in_threadpool(clean.save, destination, format="JPEG", quality=95)
                with closing(connect(db_path)) as db:
                    cursor = db.execute("""INSERT INTO predictions(created_at,filename,model,cls,confidence,time_ms,model_version,request_id)
                        VALUES (?,?,?,?,?,?,?,?)""", (datetime.now(timezone.utc).isoformat(timespec="seconds"), filename,
                        result["model"], best["cls"], best["prob"], result["time_ms"], result["model_version"], request.state.request_id))
                    db.commit()
                    result["history_id"] = cursor.lastrowid
            except Exception:
                destination.unlink(missing_ok=True)
                raise
            result.update(saved_to_history=True, image_url=f"/uploads/{filename}")
        result.setdefault("timings_ms", {})["decode"] = round(decode_ms, 2)
        result["timings_ms"]["request"] = round((time.perf_counter() - started) * 1000, 2)
        return result

    api.mount("/uploads", StaticFiles(directory=uploads_dir, check_dir=False), name="uploads")
    api.mount("/reports", StaticFiles(directory=reports_dir, check_dir=False), name="reports")
    api.mount("/assets", StaticFiles(directory=frontend_dir / "assets", check_dir=False), name="assets")

    @api.get("/", response_class=HTMLResponse)
    @api.get("/history", response_class=HTMLResponse)
    @api.get("/about", response_class=HTMLResponse)
    @api.get("/diseases", response_class=HTMLResponse)
    def frontend():
        index = frontend_dir / "index.html"
        if index.is_file():
            return FileResponse(index)
        return HTMLResponse("<!doctype html><html lang='kk'><meta charset='utf-8'><title>LeafCare AI</title>"
                            "<h1>LeafCare AI</h1><p>Интерфейс әлі құрастырылмаған.</p>"
                            "<p><a href='/docs'>API құжаттамасы</a></p></html>", status_code=503)

    return api


app = create_app()

if __name__ == "__main__":
    import uvicorn
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    uvicorn.run(app, host="127.0.0.1", port=5000, access_log=False)
