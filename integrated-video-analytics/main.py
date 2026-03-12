from __future__ import annotations

import asyncio
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict

import uvicorn
from fastapi import FastAPI, File, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
from fastapi.templating import Jinja2Templates
from fpdf import FPDF

from database import get_face_records, get_plate_reads, get_recent_events, get_vehicle_records
from pipeline import warm_shared_resources
from runtime import CameraRuntime

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"

runtimes: Dict[str, CameraRuntime] = {}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".webm", ".ts"}
UPLOAD_CHUNK_SIZE = 1024 * 1024


def env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def parse_size_bytes(value: str | None, default: int) -> int:
    if value is None:
        return default
    normalized = value.strip().upper()
    if not normalized:
        return default
    suffixes = {
        "GB": 1024 * 1024 * 1024,
        "MB": 1024 * 1024,
        "KB": 1024,
        "B": 1,
    }
    for suffix, multiplier in suffixes.items():
        if normalized.endswith(suffix):
            number = normalized[: -len(suffix)].strip()
            try:
                return max(int(float(number) * multiplier), 1)
            except ValueError:
                return default
    try:
        return max(int(normalized), 1)
    except ValueError:
        return default


def sanitize_upload_name(filename: str | None) -> str:
    original = Path(filename or "video.mp4").name
    suffix = Path(original).suffix.lower() or ".mp4"
    stem = Path(original).stem
    safe_stem = "".join(ch if ch.isalnum() or ch in {"_", "-", "."} else "_" for ch in stem).strip("._")
    return f"{safe_stem or 'video'}{suffix}"


MAX_UPLOAD_SIZE_BYTES = parse_size_bytes(os.getenv("CYBERSHIELD_MAX_UPLOAD_SIZE"), 512 * 1024 * 1024)
PRELOAD_SHARED_MODELS = env_flag("CYBERSHIELD_PRELOAD_MODELS", True)
WS_UPDATE_INTERVAL_SECONDS = max(float(os.getenv("CYBERSHIELD_WS_INTERVAL", "1.0")), 0.25)


def sanitize_camera_id(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in value.strip())
    return cleaned or f"camera_{int(time.time())}"


def next_camera_id() -> str:
    index = 1
    while f"camera_{index}" in runtimes:
        index += 1
    return f"camera_{index}"


async def persist_upload(file: UploadFile, target_path: Path) -> None:
    total_written = 0
    try:
        with target_path.open("wb") as output_file:
            while True:
                chunk = await file.read(UPLOAD_CHUNK_SIZE)
                if not chunk:
                    break
                total_written += len(chunk)
                if total_written > MAX_UPLOAD_SIZE_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail=f"Upload exceeds the {MAX_UPLOAD_SIZE_BYTES} byte limit.",
                    )
                output_file.write(chunk)
    except Exception:
        target_path.unlink(missing_ok=True)
        raise
    finally:
        await file.close()


def get_initial_state(camera_id: str, source: str = "") -> dict:
    return {
        "camera_id": camera_id,
        "source": source,
        "vehicle_count": 0,
        "vehicle_total_count": 0,
        "vehicle_types": {"car": 0, "motorcycle": 0, "bus": 0, "truck": 0},
        "vehicle_current_types": {"car": 0, "motorcycle": 0, "bus": 0, "truck": 0},
        "people_count": 0,
        "people_total_count": 0,
        "gender_stats": {"Man": 0, "Woman": 0, "Unknown": 0},
        "crowd_density": "Low",
        "faces_detected": 0,
        "plates_detected": 0,
        "zone_count": 0,
        "recent_vehicles": [],
        "recent_plates": [],
        "recent_faces": [],
        "event_logs": [],
        "last_updated": None,
        "is_processing": False,
        "stream_fps": 0.0,
        "analytics_fps": 0.0,
        "inference_latency_ms": 0.0,
        "plate_detector_ready": False,
        "detector_model": None,
        "plate_model": None,
        "device": "cpu",
    }


def get_state_snapshot(camera_id: str) -> dict:
    if camera_id not in runtimes:
        return get_initial_state(camera_id)
    return runtimes[camera_id].snapshot_state()


def release_camera(camera_id: str, drop_state: bool = False) -> None:
    runtime = runtimes.pop(camera_id, None)
    if runtime is not None:
        runtime.release()


def mount_camera(camera_id: str, source: str) -> None:
    release_camera(camera_id)
    state = get_initial_state(camera_id, str(source))
    runtimes[camera_id] = CameraRuntime(camera_id, str(source), state)


@asynccontextmanager
async def lifespan(_: FastAPI):
    UPLOAD_DIR.mkdir(exist_ok=True)
    (BASE_DIR / "watchlist").mkdir(exist_ok=True)
    if PRELOAD_SHARED_MODELS:
        await asyncio.to_thread(warm_shared_resources)
    yield
    for camera_id in list(runtimes.keys()):
        release_camera(camera_id, drop_state=False)


app = FastAPI(title="CyberShield AI Video Analytics", lifespan=lifespan)
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def enforce_upload_limits(request: Request, call_next):
    if request.url.path == "/api/video/upload":
        content_length = request.headers.get("content-length")
        if content_length:
            try:
                announced_size = int(content_length)
            except ValueError:
                return JSONResponse(status_code=400, content={"detail": "Invalid Content-Length header."})
            if announced_size > MAX_UPLOAD_SIZE_BYTES:
                return JSONResponse(
                    status_code=413,
                    content={"detail": f"Upload exceeds the {MAX_UPLOAD_SIZE_BYTES} byte limit."},
                )
    return await call_next(request)


@app.get("/", response_class=HTMLResponse)
async def read_dashboard(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/api/cameras")
async def list_cameras():
    return {
        "cameras": [
            {
                "camera_id": camera_id,
                "source": runtimes[camera_id].source,
                "running": runtimes[camera_id].running,
            }
            for camera_id in runtimes
        ]
    }


@app.post("/api/cameras/add")
async def add_camera(camera_id: str, source: str):
    camera_id = sanitize_camera_id(camera_id)
    try:
        mount_camera(camera_id, source)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"status": "success", "camera_id": camera_id, "info": f"Camera {camera_id} added."}


@app.delete("/api/cameras/{camera_id}")
async def remove_camera(camera_id: str):
    if camera_id not in runtimes:
        raise HTTPException(status_code=404, detail="Camera not found")
    release_camera(camera_id, drop_state=True)
    return {"status": "success", "camera_id": camera_id}


@app.post("/api/video/upload")
async def upload_video(request: Request, file: UploadFile = File(...), camera_id: str | None = None):
    if camera_id is None:
        form = await request.form()
        raw_camera_id = form.get("camera_id")
        if isinstance(raw_camera_id, str) and raw_camera_id.strip():
            camera_id = raw_camera_id
    camera_id = sanitize_camera_id(camera_id) if camera_id else next_camera_id()
    safe_name = sanitize_upload_name(file.filename)
    if Path(safe_name).suffix.lower() not in VIDEO_EXTENSIONS:
        raise HTTPException(status_code=415, detail="Unsupported video format.")
    target_path = UPLOAD_DIR / f"{camera_id}_{int(time.time())}_{safe_name}"

    await persist_upload(file, target_path)

    try:
        mount_camera(camera_id, str(target_path))
    except Exception as exc:
        target_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "status": "success",
        "camera_id": camera_id,
        "filename": safe_name,
        "info": f"File '{safe_name}' mounted as {camera_id}",
    }


def generate_frames(camera_id: str):
    runtime = runtimes.get(camera_id)
    if runtime is None:
        return
    yield from runtime.frame_generator()


@app.get("/api/video/stream")
def video_feed_default():
    if not runtimes:
        raise HTTPException(status_code=404, detail="No camera available")
    camera_id = next(iter(runtimes.keys()))
    return StreamingResponse(
        generate_frames(camera_id),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/api/video/stream/{camera_id}")
def video_feed(camera_id: str):
    if camera_id not in runtimes:
        raise HTTPException(status_code=404, detail="Camera not found")
    return StreamingResponse(
        generate_frames(camera_id),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/api/analytics/status")
def get_analytics_status(camera_id: str):
    return get_state_snapshot(camera_id)


@app.get("/api/logs/history")
def get_logs_history(limit: int = 50, query: str | None = None, camera_id: str | None = None):
    return {"logs": get_recent_events(limit=limit, query=query, camera_id=camera_id)}


@app.get("/api/records/plates")
def get_plate_history(limit: int = 25, query: str | None = None, camera_id: str | None = None):
    return {"records": get_plate_reads(limit=limit, query=query, camera_id=camera_id)}


@app.get("/api/records/vehicles")
def get_vehicle_history(limit: int = 25, query: str | None = None, camera_id: str | None = None):
    return {"records": get_vehicle_records(limit=limit, query=query, camera_id=camera_id)}


@app.get("/api/records/faces")
def get_face_history(limit: int = 25, query: str | None = None, camera_id: str | None = None):
    return {"records": get_face_records(limit=limit, query=query, camera_id=camera_id)}


@app.websocket("/ws/analytics/{camera_id}")
async def websocket_endpoint(websocket: WebSocket, camera_id: str):
    await websocket.accept()
    try:
        while True:
            await websocket.send_json(get_state_snapshot(camera_id))
            await asyncio.sleep(WS_UPDATE_INTERVAL_SECONDS)
    except WebSocketDisconnect:
        return


@app.get("/api/reports/download")
async def download_report(camera_id: str):
    state = get_state_snapshot(camera_id)
    vehicle_records = get_vehicle_records(limit=10, camera_id=camera_id)
    plate_records = get_plate_reads(limit=10, camera_id=camera_id)
    face_records = get_face_records(limit=10, camera_id=camera_id)

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=12)
    pdf.add_page()
    pdf.set_font("Arial", size=15)
    pdf.cell(190, 10, txt="CyberShield AI Video Analytics Report", ln=1, align="C")

    pdf.set_font("Arial", size=11)
    pdf.cell(190, 8, txt=f"Camera ID: {camera_id}", ln=1)
    pdf.cell(190, 8, txt=f"Source: {state.get('source', 'N/A')}", ln=1)
    pdf.cell(190, 8, txt=f"Last Updated: {state.get('last_updated', 'N/A')}", ln=1)
    pdf.ln(4)
    pdf.cell(190, 8, txt=f"Vehicles in frame: {state.get('vehicle_count', 0)}", ln=1)
    pdf.cell(190, 8, txt=f"Stable vehicles recorded over session: {state.get('vehicle_total_count', 0)}", ln=1)
    pdf.cell(190, 8, txt=f"People in frame: {state.get('people_count', 0)}", ln=1)
    pdf.cell(190, 8, txt=f"People tracked over session: {state.get('people_total_count', 0)}", ln=1)
    pdf.cell(190, 8, txt=f"Crowd density: {state.get('crowd_density', 'Low')}", ln=1)
    pdf.cell(190, 8, txt=f"Faces analyzed: {state.get('faces_detected', 0)}", ln=1)
    pdf.cell(190, 8, txt=f"Plates logged: {state.get('plates_detected', 0)}", ln=1)
    pdf.cell(190, 8, txt=f"Zone occupancy: {state.get('zone_count', 0)}", ln=1)
    pdf.cell(190, 8, txt=f"Output stream FPS: {state.get('stream_fps', 0.0)}", ln=1)
    pdf.cell(190, 8, txt=f"Analytics FPS: {state.get('analytics_fps', 0.0)}", ln=1)
    pdf.cell(190, 8, txt=f"Inference latency: {state.get('inference_latency_ms', 0.0)} ms", ln=1)
    pdf.cell(190, 8, txt=f"Device: {state.get('device', 'cpu')}", ln=1)
    pdf.ln(4)

    pdf.set_font("Arial", size=12)
    pdf.cell(190, 8, txt="Vehicle classification totals", ln=1)
    pdf.set_font("Arial", size=11)
    for vehicle_type, count in state.get("vehicle_types", {}).items():
        pdf.cell(190, 7, txt=f"{vehicle_type.capitalize()}: {count}", ln=1)

    pdf.ln(4)
    pdf.set_font("Arial", size=12)
    pdf.cell(190, 8, txt="Gender analytics", ln=1)
    pdf.set_font("Arial", size=11)
    for gender, count in state.get("gender_stats", {}).items():
        pdf.cell(190, 7, txt=f"{gender}: {count}", ln=1)

    if vehicle_records:
        pdf.ln(4)
        pdf.set_font("Arial", size=12)
        pdf.cell(190, 8, txt="Recent vehicle records", ln=1)
        pdf.set_font("Arial", size=10)
        for record in vehicle_records:
            plate_value = record["plate_text"] or "Plate pending"
            pdf.multi_cell(
                190,
                6,
                txt=(
                    f"tracker #{record['tracker_id']} | {record['vehicle_type']} | {plate_value} | "
                    f"first seen {record['first_seen']} | last seen {record['last_seen']}"
                ),
            )

    if plate_records:
        pdf.ln(4)
        pdf.set_font("Arial", size=12)
        pdf.cell(190, 8, txt="Recent ANPR records", ln=1)
        pdf.set_font("Arial", size=10)
        for record in plate_records:
            pdf.multi_cell(
                190,
                6,
                txt=(
                    f"{record['plate_text']} | {record['vehicle_type']} | "
                    f"first seen {record['first_seen']} | last seen {record['last_seen']}"
                ),
            )

    if face_records:
        pdf.ln(4)
        pdf.set_font("Arial", size=12)
        pdf.cell(190, 8, txt="Recent face records", ln=1)
        pdf.set_font("Arial", size=10)
        for record in face_records:
            identity = record["identity"] or "Anonymous"
            status = "Watchlist hit" if record["watchlist_hit"] else "No watchlist hit"
            pdf.multi_cell(
                190,
                6,
                txt=(
                    f"{identity} | {record['gender'] or 'Unknown'} | {status} | "
                    f"first seen {record['first_seen']} | last seen {record['last_seen']}"
                ),
            )

    payload = pdf.output(dest="S")
    if isinstance(payload, str):
        payload = payload.encode("latin-1")
    elif isinstance(payload, bytearray):
        payload = bytes(payload)
    return Response(
        content=payload,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'attachment; filename="CyberShield_{camera_id}_Analytics_Report.pdf"'
        },
    )


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True)
