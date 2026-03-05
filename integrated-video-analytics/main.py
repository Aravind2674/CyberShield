from fastapi import FastAPI, BackgroundTasks, UploadFile, File, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
import uvicorn
import cv2
import threading
import time
import os

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

@app.post("/api/video/upload")
async def upload_video(file: UploadFile = File(...)):
    global current_video_path, is_streaming
    
    # Save the file temporarily
    file_location = f"temp_{file.filename}"
    with open(file_location, "wb+") as file_object:
        file_object.write(file.file.read())
        
    current_video_path = file_location
    is_streaming = True
    
    return {"info": f"file '{file.filename}' saved at '{file_location}'"}

def generate_frames():
    global current_video_path, is_streaming, analytics_state
    
    if not current_video_path:
        return
        
    cap = cv2.VideoCapture(current_video_path)
    
    while is_streaming and cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
            
        # Process the frame through our YOLOv5 AI pipeline
        analytics_state['is_processing'] = True
        processed_frame = pipeline.process_frame(frame, analytics_state)
        
        # Encode the frame in JPEG format
        ret, buffer = cv2.imencode('.jpg', processed_frame)
        frame_bytes = buffer.tobytes()
        
        # Yield the frame in the byte format
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
               
    cap.release()
    analytics_state['is_processing'] = False

@app.get("/api/video/stream")
def video_feed():
    # Return the response generated along with the specific media type (mime type)
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/api/analytics/status")
def get_analytics_status():
    return analytics_state

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True)
