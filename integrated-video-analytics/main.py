from fastapi import FastAPI, BackgroundTasks, UploadFile, File, Request, WebSocket, WebSocketDisconnect
import asyncio
from fastapi.responses import JSONResponse, StreamingResponse, HTMLResponse, FileResponse
from fpdf import FPDF
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
import uvicorn
import cv2
import threading
import time
import os
from database import log_event, get_recent_events
from camera import CameraStream

from pipeline import VideoPipeline

app = FastAPI(title="AI Video Analytics System")
templates = Jinja2Templates(directory="templates")

# Initialize Pipeline
pipeline = VideoPipeline()

# Global variables
current_video_path = None
is_streaming = False

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state for analytics
analytics_state = {
    "vehicle_count": 0,
    "vehicle_types": {"car": 0, "motorcycle": 0, "bus": 0, "truck": 0},
    "people_count": 0,
    "gender_stats": {"Man": 0, "Woman": 0},
    "faces_detected": 0,
    "plates_detected": 0,
    "event_logs": [], # To store searchable records [{time, type, detail}]
    "is_processing": False
}

@app.get("/", response_class=HTMLResponse)
async def read_dashboard(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# --- Multi-Camera Setup ---

# Maps camera_id -> CameraStream object
cameras = {}
# Maps camera_id -> Pipeline Instance (each needs its own tracker state)
pipelines = {}

@app.post("/api/cameras/add")
async def add_camera(camera_id: str, source: str):
    """Dynamically add an RTSP source or uploaded video source"""
    cameras[camera_id] = CameraStream(source)
    pipelines[camera_id] = VideoPipeline() # Separate tracking IDs per camera
    return {"status": "success", "info": f"Camera {camera_id} added."}

@app.post("/api/video/upload")
async def upload_video(file: UploadFile = File(...)):
    global is_streaming
    
    file_location = f"temp_{file.filename}"
    with open(file_location, "wb+") as file_object:
        file_object.write(file.file.read())
        
    # Treat the uploaded file as physical "camera_1"
    cameras["camera_1"] = CameraStream(file_location)
    pipelines["camera_1"] = VideoPipeline()
    is_streaming = True
    
    return {"info": f"file '{file.filename}' mounted as camera_1 stream"}

def generate_frames(camera_id: str):
    global is_streaming, analytics_state
    
    if camera_id not in cameras:
        return
        
    stream = cameras[camera_id]
    pipe = pipelines[camera_id]
    
    while is_streaming and stream.running:
        success, frame = stream.read()
        if not success:
            break
            
        analytics_state['is_processing'] = True
        processed_frame = pipe.process_frame(frame, analytics_state)
        
        ret, buffer = cv2.imencode('.jpg', processed_frame)
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
               
    analytics_state['is_processing'] = False

@app.get("/api/video/stream")
def video_feed_default():
    """Fallback feed for current architecture (always routes to camera_1)"""
    return StreamingResponse(generate_frames("camera_1"), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/api/video/stream/{camera_id}")
def video_feed(camera_id: str):
    if camera_id not in cameras:
        return JSONResponse(status_code=404, content={"message": "Camera not found"})
    return StreamingResponse(generate_frames(camera_id), media_type="multipart/x-mixed-replace; boundary=frame")

# --- Analytics endpoints ---

@app.get("/api/analytics/status")
def get_analytics_status():
    return analytics_state

@app.get("/api/logs/history")
def get_logs_history(limit: int = 50):
    """Retrieve historical logs from SQLite rather than memory."""
    logs = get_recent_events(limit)
    return {"logs": logs}

# Active websocket connections
connected_clients = []

@app.websocket("/ws/analytics")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    connected_clients.append(websocket)
    try:
        while True:
            # Send current state periodically
            await websocket.send_json(analytics_state)
            await asyncio.sleep(0.5)
    except WebSocketDisconnect:
        connected_clients.remove(websocket)

@app.get("/api/reports/download")
async def download_report():
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=15)
    pdf.cell(200, 10, txt="CyberShield AI Video Analytics Report", ln=1, align="C")
    
    pdf.set_font("Arial", size=12)
    pdf.ln(10)
    pdf.cell(200, 10, txt=f"Total Vehicles Detected: {analytics_state.get('vehicle_count', 0)}", ln=1)
    pdf.cell(200, 10, txt=f"Total People Detected: {analytics_state.get('people_count', 0)}", ln=1)
    pdf.cell(200, 10, txt=f"Zone Trigger Count: {analytics_state.get('zone_count', 0)}", ln=1)
    pdf.cell(200, 10, txt=f"Faces Detected: {analytics_state.get('faces_detected', 0)}", ln=1)
    pdf.cell(200, 10, txt=f"Plates Read: {analytics_state.get('plates_detected', 0)}", ln=1)
    
    pdf.ln(10)
    pdf.cell(200, 10, txt="Vehicle Types:", ln=1)
    for v_type, count in analytics_state.get('vehicle_types', {}).items():
        pdf.cell(200, 10, txt=f"  - {v_type.capitalize()}: {count}", ln=1)
        
    pdf.ln(10)
    pdf.cell(200, 10, txt="Gender Stats:", ln=1)
    for g_type, count in analytics_state.get('gender_stats', {}).items():
        pdf.cell(200, 10, txt=f"  - {g_type}: {count}", ln=1)

    # Save to temp file
    report_path = "analytics_report.pdf"
    pdf.output(report_path)
    
    return FileResponse(path=report_path, filename="CyberShield_Analytics_Report.pdf", media_type="application/pdf")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True)
