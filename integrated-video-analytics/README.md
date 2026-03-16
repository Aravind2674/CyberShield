# CyberShield - Integrated Video Analytics

CyberShield is a unified AI video analytics platform for vehicle counting, ANPR, facial recognition, people counting, gender analytics, searchable records, and live reporting.

## Core modules

- Vehicle counting and classification with `YOLOv8` object detection plus `ByteTrack` tracking
- Automatic number plate recognition with a dedicated `YOLOv8` plate detector and `EasyOCR`
- Facial recognition and gender analytics with `DeepFace`
- People counting and crowd density estimation
- Searchable SQLite-backed event, plate, face, and vehicle records
- FastAPI dashboard with live MJPEG streaming, charts, searchable records, and PDF report export

## Runtime behavior

- On CPU, the app defaults to `yolov8n.pt` for lower latency.
- On CUDA-enabled systems, the detector defaults to `yolov8s.pt`.
- The first run downloads detector weights, OCR assets, and DeepFace dependencies automatically.
- Plate recognition uses `https://huggingface.co/yasirfaizahmed/license-plate-object-detection/resolve/main/best.pt` by default and can be overridden with `CYBERSHIELD_PLATE_MODEL`.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

Open `http://localhost:8080`.

### Optional face analytics dependencies

`insightface` and `tf-keras` are optional and may not provide wheels on newer Python releases.

If you want watchlist face matching, use Python 3.11 and install:

```bash
pip install -r requirements-face.txt
```

## Notes

- Uploaded videos are stored in `uploads/`.
- Watchlist images should be placed in `watchlist/`.
- SQLite data is stored in `analytics.db`.
- The included sample videos can be used for smoke testing.
