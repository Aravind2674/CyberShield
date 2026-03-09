from __future__ import annotations

import asyncio
import shutil
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict

import cv2
import uvicorn
from fastapi import FastAPI, File, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from fpdf import FPDF

from camera import CameraStream
from database import get_face_records, get_plate_reads, get_recent_events
from pipeline import VideoPipeline

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
REPORT_PATH = BASE_DIR / "analytics_report.pdf"

cameras: Dict[str, CameraStream] = {}
pipelines: Dict[str, VideoPipeline] = {}
camera_states: Dict[str, dict] = {}
camera_sources: Dict[str, str] = {}


def sanitize_camera_id(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in value.strip())
    return cleaned or f"camera_{int(time.time())}"


def next_camera_id() -> str:
    index = 1
    while f"camera_{index}" in cameras:
        index += 1
    return f"camera_{index}"


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
        "recent_plates": [],
        "recent_faces": [],
        "event_logs": [],
        "last_updated": None,
        "is_processing": False,
    }


def get_state_snapshot(camera_id: str) -> dict:
    if camera_id not in camera_states:
        return get_initial_state(camera_id)

    pipeline = pipelines.get(camera_id)
    state = camera_states[camera_id]
    if pipeline is None:
        return state.copy()
    return pipeline.snapshot_state(state)


def release_camera(camera_id: str, drop_state: bool = False) -> None:
    stream = cameras.pop(camera_id, None)
    if stream is not None:
        stream.release()

    pipeline = pipelines.pop(camera_id, None)
    if pipeline is not None:
        pipeline.shutdown()

    camera_sources.pop(camera_id, None)
    if drop_state:
        camera_states.pop(camera_id, None)


def mount_camera(camera_id: str, source: str) -> None:
    release_camera(camera_id)
    cameras[camera_id] = CameraStream(source)
    pipelines[camera_id] = VideoPipeline(camera_id)
    camera_states[camera_id] = get_initial_state(camera_id, str(source))
    camera_sources[camera_id] = str(source)


@asynccontextmanager
async def lifespan(_: FastAPI):
    UPLOAD_DIR.mkdir(exist_ok=True)
    (BASE_DIR / "watchlist").mkdir(exist_ok=True)
    yield
    for camera_id in list(cameras.keys()):
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


@app.get("/", response_class=HTMLResponse)
async def read_dashboard(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/api/cameras")
async def list_cameras():
    return {
        "cameras": [
            {
                "camera_id": camera_id,
                "source": camera_sources.get(camera_id, ""),
                "running": cameras[camera_id].running,
            }
            for camera_id in cameras
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
    if camera_id not in cameras:
        raise HTTPException(status_code=404, detail="Camera not found")
    release_camera(camera_id, drop_state=True)
    return {"status": "success", "camera_id": camera_id}


@app.post("/api/video/upload")
async def upload_video(file: UploadFile = File(...), camera_id: str | None = None):
    camera_id = sanitize_camera_id(camera_id) if camera_id else next_camera_id()
    safe_name = Path(file.filename or "video.mp4").name
    target_path = UPLOAD_DIR / f"{camera_id}_{int(time.time())}_{safe_name}"

    with target_path.open("wb") as output_file:
        shutil.copyfileobj(file.file, output_file)

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
    if camera_id not in cameras:
        return

    stream = cameras[camera_id]
    pipeline = pipelines[camera_id]
    state = camera_states[camera_id]

    try:
        while stream.running:
            success, frame = stream.read()
            if not success or frame is None:
                break

            state["is_processing"] = True
            processed_frame = pipeline.process_frame(frame, state)
            ok, buffer = cv2.imencode(".jpg", processed_frame)
            if not ok:
                continue

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
            )
    finally:
        state["is_processing"] = False


@app.get("/api/video/stream")
def video_feed_default():
    if not cameras:
        raise HTTPException(status_code=404, detail="No camera available")
    camera_id = next(iter(cameras.keys()))
    return StreamingResponse(
        generate_frames(camera_id),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/api/video/stream/{camera_id}")
def video_feed(camera_id: str):
    if camera_id not in cameras:
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


@app.get("/api/records/faces")
def get_face_history(limit: int = 25, query: str | None = None, camera_id: str | None = None):
    return {"records": get_face_records(limit=limit, query=query, camera_id=camera_id)}


@app.websocket("/ws/analytics/{camera_id}")
async def websocket_endpoint(websocket: WebSocket, camera_id: str):
    await websocket.accept()
    try:
        while True:
            await websocket.send_json(get_state_snapshot(camera_id))
            await asyncio.sleep(0.5)
    except WebSocketDisconnect:
        return


@app.get("/api/reports/download")
async def download_report(camera_id: str):
    state = get_state_snapshot(camera_id)
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

    pdf.output(str(REPORT_PATH))
    return FileResponse(
        path=str(REPORT_PATH),
        filename=f"CyberShield_{camera_id}_Analytics_Report.pdf",
        media_type="application/pdf",
    )


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True)
