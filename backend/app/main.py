"""HTTP API. The browser owns the editing; the server owns images and models."""
from __future__ import annotations

import base64
import hmac
import io
import os
import threading
import time
import uuid
from collections import OrderedDict
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps, UnidentifiedImageError
from pydantic import BaseModel, Field

from . import pipeline, text, vector
from .models import engine

MAX_UPLOAD_BYTES = 40 * 1024 * 1024
MAX_SIDE = 4096
MAX_SESSIONS = 6
ALLOWED_FORMATS = {"PNG", "JPEG", "WEBP", "BMP"}
Image.MAX_IMAGE_PIXELS = 80_000_000
# When the API is reachable from the Internet these two must be set.
ACCESS_KEY = os.environ.get("PROLAYER_ACCESS_KEY", "")
ALLOWED_ORIGINS = [origin for origin in os.environ.get("PROLAYER_ORIGINS", "").split(",") if origin]


@dataclass
class Session:
    id: str
    rgb: np.ndarray
    background: np.ndarray
    embeddings: object = None
    layers: dict[str, pipeline.Layer] = field(default_factory=dict)
    masks: OrderedDict[str, np.ndarray] = field(default_factory=OrderedDict)
    background_version: int = 0
    progress: dict = field(default_factory=lambda: {"stage": "", "value": 0.0})
    last_used: float = field(default_factory=time.monotonic)
    lock: threading.Lock = field(default_factory=threading.Lock)

    @property
    def size(self) -> tuple[int, int]:
        return self.rgb.shape[1], self.rgb.shape[0]


sessions: OrderedDict[str, Session] = OrderedDict()
sessions_lock = threading.Lock()


@asynccontextmanager
async def lifespan(_: FastAPI):
    # In the background so the page can load and show progress meanwhile.
    threading.Thread(target=engine.preload, daemon=True).start()
    yield


app = FastAPI(title="ProLayer", lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)


@app.middleware("http")
async def require_access_key(request: Request, call_next):
    # GETs are covered by the unguessable session id in their URL, which also
    # lets plain <img> tags load layers. Everything that costs GPU needs the key.
    if ACCESS_KEY and request.url.path.startswith("/api/") and request.method not in ("GET", "HEAD", "OPTIONS"):
        supplied = request.headers.get("x-prolayer-key", "")
        if not hmac.compare_digest(supplied.encode(), ACCESS_KEY.encode()):
            return JSONResponse({"detail": "Mã truy cập không đúng."}, status_code=401)
    return await call_next(request)


# Added last so it wraps the key check: browsers need CORS headers on a 401 too.
app.add_middleware(CORSMiddleware, allow_origins=ALLOWED_ORIGINS, allow_methods=["*"], allow_headers=["*"])


def get_session(session_id: str) -> Session:
    with sessions_lock:
        session = sessions.get(session_id)
        if session is None:
            raise HTTPException(404, "Phiên làm việc đã hết hạn, hãy tải ảnh lên lại.")
        sessions.move_to_end(session_id)
        session.last_used = time.monotonic()
        return session


def png_response(array: np.ndarray) -> Response:
    return Response(encode_png(array), media_type="image/png", headers={"Cache-Control": "no-store"})


def encode_png(array: np.ndarray) -> bytes:
    if array.ndim == 2:
        converted = array
    elif array.shape[2] == 4:
        converted = cv2.cvtColor(array, cv2.COLOR_RGBA2BGRA)
    else:
        converted = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
    ok, data = cv2.imencode(".png", converted, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    if not ok:
        raise HTTPException(500, "Không mã hóa được ảnh.")
    return data.tobytes()


def layer_json(session: Session, layer: pipeline.Layer) -> dict:
    return {
        "id": layer.id,
        "name": layer.name,
        "left": layer.left,
        "top": layer.top,
        "width": layer.rgba.shape[1],
        "height": layer.rgba.shape[0],
        "blendMode": layer.blend_mode,
        "kind": layer.kind,
        "text": layer.text,
        "url": f"/api/sessions/{session.id}/layers/{layer.id}.png",
    }


def read_image(data: bytes) -> np.ndarray:
    try:
        with Image.open(io.BytesIO(data)) as image:
            if image.format not in ALLOWED_FORMATS:
                raise HTTPException(415, "Chỉ hỗ trợ PNG, JPEG, WEBP hoặc BMP.")
            image = ImageOps.exif_transpose(image)
            if image.mode in ("RGBA", "LA", "P"):
                # Transparent uploads are flattened: SAM expects opaque RGB.
                rgba = image.convert("RGBA")
                flat = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
                flat.alpha_composite(rgba)
                image = flat
            image = image.convert("RGB")
            image.thumbnail((MAX_SIDE, MAX_SIDE), Image.LANCZOS)
            return np.array(image)
    except (UnidentifiedImageError, Image.DecompressionBombError, OSError) as error:
        raise HTTPException(415, "Tệp không phải ảnh hợp lệ hoặc ảnh quá lớn.") from error


@app.get("/api/health")
def health() -> dict:
    status = engine.status()
    if ACCESS_KEY and status["error"]:
        status["error"] = "Xem nhật ký trên máy chủ."  # raw errors contain local paths
    return status | {"keyRequired": bool(ACCESS_KEY)}


@app.post("/api/sessions")
async def create_session(file: UploadFile = File(...)) -> dict:
    data = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "Ảnh vượt quá 40MB.")
    rgb = read_image(data)
    session = Session(id=uuid.uuid4().hex, rgb=rgb, background=rgb.copy())
    with sessions_lock:
        sessions[session.id] = session
        while len(sessions) > MAX_SESSIONS:
            sessions.popitem(last=False)
    width, height = session.size
    return {"id": session.id, "width": width, "height": height}


@app.get("/api/sessions/{session_id}/image.png")
def get_image(session_id: str) -> Response:
    return png_response(get_session(session_id).rgb)


@app.get("/api/sessions/{session_id}/background.png")
def get_background(session_id: str) -> Response:
    return png_response(get_session(session_id).background)


@app.get("/api/sessions/{session_id}/layers/{layer_id}.png")
def get_layer(session_id: str, layer_id: str) -> Response:
    layer = get_session(session_id).layers.get(layer_id)
    if layer is None:
        raise HTTPException(404, "Không tìm thấy layer.")
    return png_response(layer.rgba)


@app.get("/api/sessions/{session_id}/layers/{layer_id}/svg")
def get_layer_svg(session_id: str, layer_id: str) -> dict:
    layer = get_session(session_id).layers.get(layer_id)
    if layer is None:
        raise HTTPException(404, "Không tìm thấy layer.")
    return {"svg": vector.layer_svg(layer)}


@app.get("/api/sessions/{session_id}/progress")
def get_progress(session_id: str) -> dict:
    return get_session(session_id).progress


class Point(BaseModel):
    x: float
    y: float
    positive: bool = True


class SegmentRequest(BaseModel):
    points: list[Point] = Field(default_factory=list, max_length=32)
    box: tuple[float, float, float, float] | None = None
    # None picks the mask SAM is most confident in; 0-2 go from part to whole.
    level: Literal[0, 1, 2] | None = None


def ensure_embeddings(session: Session) -> None:
    if session.embeddings is None:
        session.embeddings = engine.embed(session.rgb)


def preview_png(mask: np.ndarray) -> str:
    """Tinted overlay with a crisp outline, small enough to send on every click."""
    height, width = mask.shape
    scale = min(1.0, 1280 / max(width, height))
    small = cv2.resize(
        mask.astype(np.uint8), (max(1, round(width * scale)), max(1, round(height * scale))), interpolation=cv2.INTER_NEAREST
    )
    outline = cv2.morphologyEx(small, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0
    overlay = np.zeros((*small.shape, 4), np.uint8)
    overlay[small > 0] = (201, 18, 107, 120)
    overlay[outline] = (255, 255, 255, 255)
    return "data:image/png;base64," + base64.b64encode(encode_png(overlay)).decode()


@app.post("/api/sessions/{session_id}/segment")
def segment(session_id: str, request: SegmentRequest) -> dict:
    session = get_session(session_id)
    if not request.points and request.box is None:
        raise HTTPException(422, "Cần ít nhất một điểm hoặc một khung.")
    width, height = session.size
    for point in request.points:
        if not (0 <= point.x < width and 0 <= point.y < height):
            raise HTTPException(422, "Điểm nằm ngoài ảnh.")
    with session.lock:
        ensure_embeddings(session)
        mask, score = engine.predict_mask(
            session.rgb,
            session.embeddings,
            [(p.x, p.y) for p in request.points],
            [int(p.positive) for p in request.points],
            request.box,
            request.level,
        )
        mask_id = uuid.uuid4().hex[:12]
        session.masks[mask_id] = mask
        while len(session.masks) > 8:
            session.masks.popitem(last=False)
    return {"maskId": mask_id, "score": score, "empty": not mask.any(), "preview": preview_png(mask)}


class ExtractRequest(BaseModel):
    mask_id: str
    edge: float = Field(1.0, ge=0.2, le=4.0)
    transparent: bool = False
    shadow: bool = False


@app.post("/api/sessions/{session_id}/extract")
def extract(session_id: str, request: ExtractRequest) -> dict:
    session = get_session(session_id)
    with session.lock:
        mask = session.masks.pop(request.mask_id, None)
        if mask is None:
            raise HTTPException(404, "Vùng chọn không còn, hãy chọn lại.")
        if not mask.any():
            raise HTTPException(422, "Vùng chọn trống.")
        layers, session.background = pipeline.extract_element(
            session.rgb,
            session.background,
            mask,
            f"Phần tử {len(session.layers) + 1}",
            request.edge,
            request.transparent,
            request.shadow,
        )
        session.layers.update({layer.id: layer for layer in layers})
        session.background_version += 1
    return {"layers": [layer_json(session, layer) for layer in layers], "backgroundVersion": session.background_version}


class DecomposeRequest(BaseModel):
    detail: pipeline.Detail = "medium"
    max_layers: int = Field(24, ge=1, le=60)
    edge: float = Field(1.0, ge=0.2, le=4.0)
    text: bool = True
    shadow: bool = True


@app.post("/api/sessions/{session_id}/decompose")
def decompose(session_id: str, request: DecomposeRequest) -> dict:
    session = get_session(session_id)

    def report(stage: str, value: float) -> None:
        session.progress = {"stage": stage, "value": round(value, 3)}

    with session.lock:
        report("Đang phân tích ảnh", 0.0)
        try:
            text_layers: list[pipeline.Layer] = []
            source = session.rgb
            if request.text:
                report("Đang đọc chữ", 0.01)
                text_layers, source = text.extract_text(session.rgb)
            # Letters would fragment the masks, so segmentation sees the erased picture.
            embeddings = engine.embed(source) if text_layers else None
            if embeddings is None:
                ensure_embeddings(session)
                embeddings = session.embeddings
            layers, background = pipeline.decompose(
                source, embeddings, request.detail, request.max_layers, request.edge, request.shadow, report
            )
            layers += text_layers
        finally:
            report("", 0.0)
        session.layers = {layer.id: layer for layer in layers}
        session.background = background
        session.background_version += 1
        flat = pipeline.flatten(background, layers)
        mse = float(np.mean((flat.astype(np.float32) - session.rgb.astype(np.float32)) ** 2))
    return {
        "layers": [layer_json(session, layer) for layer in layers],
        "backgroundVersion": session.background_version,
        "fidelityPsnr": round(10 * np.log10(255**2 / max(mse, 1e-6)), 2),
    }


# Registered last so that /api routes win over the static catch-all.
_dist = Path(__file__).resolve().parents[2] / "frontend" / "dist"
if _dist.is_dir():
    app.mount("/", StaticFiles(directory=_dist, html=True), name="frontend")
