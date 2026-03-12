from __future__ import annotations

import concurrent.futures
import copy
import os
import re
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import easyocr
import numpy as np
try:
    import insightface
    from insightface.app import FaceAnalysis
    HAS_INSIGHTFACE = True
except ImportError:
    insightface = None
    FaceAnalysis = None
    HAS_INSIGHTFACE = False
import requests
import supervision as sv
import torch
from ultralytics import YOLO

from database import (
    log_event,
    store_metric,
    upsert_face_record,
    upsert_plate_read,
    upsert_vehicle_record,
)

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_PLATE_MODEL_URL = (
    "https://huggingface.co/yasirfaizahmed/license-plate-object-detection/resolve/main/best.pt"
)

TARGET_CLASSES = {
    0: "person",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}
VEHICLE_CLASSES = {"car", "motorcycle", "bus", "truck"}
WATCHLIST_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
PLATE_ALLOWLIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
PLATE_TEXT_PATTERN = re.compile(r"^[A-Z]{2}[0-9]{1,2}[A-Z]{0,3}[0-9]{3,4}$")


def resolve_model_path(explicit: Optional[str], *local_candidates: Path, fallback: str) -> str:
    if explicit:
        return explicit
    for candidate in local_candidates:
        if candidate.exists():
            return str(candidate)
    return fallback


def normalize_plate_text(text: str) -> Optional[str]:
    cleaned = "".join(ch for ch in text.upper() if ch.isalnum())
    if not PLATE_TEXT_PATTERN.match(cleaned):
        return None
    return cleaned


def detector_fallback_name() -> str:
    return "yolov8m.pt" if torch.cuda.is_available() else "yolov8s.pt"


def read_env_float(
    name: str,
    default: float,
    minimum: Optional[float] = None,
    maximum: Optional[float] = None,
) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = float(raw.strip())
    except (AttributeError, ValueError):
        raise ValueError(f"Invalid {name}={raw!r}; floating point required.")
    if minimum is not None:
        value = max(value, minimum)
    if maximum is not None:
        value = min(value, maximum)
    return value


def read_env_int(
    name: str,
    default: int,
    minimum: Optional[int] = None,
    maximum: Optional[int] = None,
) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = int(raw.strip())
    except (AttributeError, ValueError):
        raise ValueError(f"Invalid {name}={raw!r}; integer required.")
    if minimum is not None:
        value = max(value, minimum)
    if maximum is not None:
        value = min(value, maximum)
    return value


def touch_timestamp_cache(cache: Dict[Any, float], key: Any, timestamp: float) -> None:
    cache.pop(key, None)
    cache[key] = timestamp


def trim_timestamp_cache(
    cache: Dict[Any, float],
    now: float,
    ttl_seconds: float,
    max_items: int,
) -> None:
    cutoff = now - ttl_seconds
    stale_keys = [key for key, seen_at in cache.items() if seen_at < cutoff]
    for key in stale_keys:
        cache.pop(key, None)
    while len(cache) > max_items:
        oldest_key = next(iter(cache))
        cache.pop(oldest_key, None)


DETECTION_CONFIDENCE = read_env_float("CYBERSHIELD_DETECT_CONFIDENCE", 0.30, minimum=0.05, maximum=0.95)
PLATE_CONFIDENCE = read_env_float("CYBERSHIELD_PLATE_CONFIDENCE", 0.25, minimum=0.05, maximum=0.95)
DETECTION_IMAGE_SIZE = read_env_int(
    "CYBERSHIELD_DETECT_IMGSZ",
    768 if torch.cuda.is_available() else 640,
    minimum=320,
    maximum=1600,
)
PLATE_IMAGE_SIZE = read_env_int("CYBERSHIELD_PLATE_IMGSZ", 640, minimum=256, maximum=1280)
MIN_STABLE_FRAMES = read_env_int("CYBERSHIELD_MIN_STABLE_FRAMES", 2, minimum=1, maximum=12)
PLATE_SCAN_INTERVAL_SECONDS = read_env_float("CYBERSHIELD_PLATE_SCAN_INTERVAL", 5.0, minimum=0.2)
PLATE_REFRESH_INTERVAL_SECONDS = read_env_float("CYBERSHIELD_PLATE_REFRESH_INTERVAL", 15.0, minimum=0.5)
FACE_SCAN_INTERVAL_SECONDS = read_env_float("CYBERSHIELD_FACE_SCAN_INTERVAL", 3.0, minimum=0.5)
PLATE_OCR_TARGET_WIDTH = read_env_int("CYBERSHIELD_PLATE_TARGET_WIDTH", 480, minimum=240, maximum=1280)
PLATE_HEURISTIC_MIN_VEHICLE_WIDTH = read_env_int("CYBERSHIELD_PLATE_HEURISTIC_MIN_WIDTH", 150, minimum=80, maximum=1000)
PLATE_HEURISTIC_MIN_VEHICLE_HEIGHT = read_env_int("CYBERSHIELD_PLATE_HEURISTIC_MIN_HEIGHT", 80, minimum=40, maximum=1000)
RENDER_TRACK_TTL_SECONDS = read_env_float(
    "CYBERSHIELD_RENDER_TRACK_TTL",
    1.0 if torch.cuda.is_available() else 1.75,
    minimum=0.25,
)
RENDER_TRACK_MAX_TTL_SECONDS = read_env_float("CYBERSHIELD_RENDER_TRACK_MAX_TTL", 4.0, minimum=1.0)
RENDER_TRACK_LATENCY_MULTIPLIER = read_env_float(
    "CYBERSHIELD_RENDER_TRACK_LATENCY_MULTIPLIER",
    2.5,
    minimum=1.0,
    maximum=6.0,
)
TRACK_STALE_SECONDS = read_env_float("CYBERSHIELD_TRACK_STALE_SECONDS", 15.0, minimum=1.0)
METRIC_WRITE_INTERVAL_SECONDS = read_env_float("CYBERSHIELD_METRIC_WRITE_INTERVAL", 2.0, minimum=0.5)
PLATE_CONFIRMATION_HITS = read_env_int("CYBERSHIELD_PLATE_CONFIRMATION_HITS", 2, minimum=1, maximum=8)
PLATE_DIRECT_ACCEPT_CONFIDENCE = read_env_float(
    "CYBERSHIELD_PLATE_DIRECT_CONFIDENCE",
    0.82,
    minimum=0.1,
    maximum=0.99,
)
PLATE_MIN_AGGREGATE_SCORE = read_env_float("CYBERSHIELD_PLATE_MIN_SCORE", 1.2, minimum=0.1)
PLATE_EXECUTOR_WORKERS = read_env_int(
    "CYBERSHIELD_PLATE_WORKERS",
    max(2, min(4, os.cpu_count() or 4)),
    minimum=1,
    maximum=8,
)
PLATE_QUEUE_LIMIT = read_env_int("CYBERSHIELD_PLATE_QUEUE_LIMIT", 8, minimum=1, maximum=64)
FACE_QUEUE_LIMIT = read_env_int("CYBERSHIELD_FACE_QUEUE_LIMIT", 4, minimum=1, maximum=32)
DB_QUEUE_LIMIT = read_env_int("CYBERSHIELD_DB_QUEUE_LIMIT", 256, minimum=8, maximum=4096)
UNIQUE_CACHE_TTL_SECONDS = read_env_float("CYBERSHIELD_UNIQUE_CACHE_TTL", 3600.0, minimum=60.0)
PLATE_CACHE_LIMIT = read_env_int("CYBERSHIELD_PLATE_CACHE_LIMIT", 4096, minimum=100, maximum=200000)
FACE_TRACK_CACHE_LIMIT = read_env_int("CYBERSHIELD_FACE_TRACK_CACHE_LIMIT", 4096, minimum=100, maximum=200000)

PLATE_RECOGNIZER_API_TOKEN = os.getenv("PLATE_RECOGNIZER_API_TOKEN", "054683afe69ddd84b348aec6c88555d963301842")

DETECTION_MODEL_NAME = resolve_model_path(
    os.getenv("CYBERSHIELD_DETECT_MODEL"),
    BASE_DIR / detector_fallback_name(),
    BASE_DIR / "yolov8n.pt",
    fallback=detector_fallback_name(),
)
PLATE_MODEL_NAME = resolve_model_path(
    os.getenv("CYBERSHIELD_PLATE_MODEL"),
    BASE_DIR / "weights" / "best.pt",
    fallback=DEFAULT_PLATE_MODEL_URL,
)


class SharedResources:
    _detector: Optional[YOLO] = None
    _plate_detector: Optional[YOLO] = None
    _face_analyzer = None

    detector_lock = threading.Lock()
    plate_lock = threading.Lock()
    face_lock = threading.Lock()
    init_lock = threading.Lock()

    @classmethod
    def get_detector(cls) -> YOLO:
        with cls.init_lock:
            if cls._detector is None:
                print(f"Loading primary detector: {DETECTION_MODEL_NAME}")
                cls._detector = YOLO(DETECTION_MODEL_NAME)
            return cls._detector

    @classmethod
    def get_plate_detector(cls) -> Optional[YOLO]:
        with cls.init_lock:
            if cls._plate_detector is None:
                try:
                    print(f"Loading plate detector: {PLATE_MODEL_NAME}")
                    cls._plate_detector = YOLO(PLATE_MODEL_NAME)
                except Exception as exc:
                    print(f"Plate detector unavailable: {exc}")
                    cls._plate_detector = None
            return cls._plate_detector

    @classmethod
    def get_face_analyzer(cls):
        if not HAS_INSIGHTFACE:
            return None
        with cls.init_lock:
            if cls._face_analyzer is None:
                try:
                    print("Loading ArcFace analyzer (buffalo_l)")
                    cls._face_analyzer = FaceAnalysis(name='buffalo_l')
                    cls._face_analyzer.prepare(ctx_id=0 if torch.cuda.is_available() else -1, det_size=(640, 640))
                except Exception as exc:
                    print(f"ArcFace unavailable: {exc}")
                    cls._face_analyzer = None
            return cls._face_analyzer

def warm_shared_resources() -> None:
    SharedResources.get_detector()
    SharedResources.get_plate_detector()
    SharedResources.get_face_analyzer()


class VideoPipeline:
    def __init__(self, camera_id: str):
        self.camera_id = camera_id
        self.device = "cuda:0" if self.gpu_available() else "cpu"
        self.detector = SharedResources.get_detector()
        self.plate_detector = SharedResources.get_plate_detector()
        self.face_analyzer = SharedResources.get_face_analyzer()

        self.tracker = sv.ByteTrack()
        self.state_lock = threading.RLock()
        self.plate_executor = concurrent.futures.ThreadPoolExecutor(max_workers=PLATE_EXECUTOR_WORKERS)
        self.face_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self.db_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self.watchlist_dir = BASE_DIR / "watchlist"

        self.track_states: Dict[int, Dict[str, Any]] = {}
        self.render_tracks: Dict[int, Dict[str, Any]] = {}
        self.face_results: Dict[int, Dict[str, Any]] = {}
        self.plate_results: Dict[int, Dict[str, Any]] = {}
        self.plate_votes: Dict[int, Dict[str, Dict[str, float]]] = {}
        self.mmc_votes: Dict[int, Dict[str, Dict[str, float]]] = {}
        self.detected_plate_texts: Dict[str, float] = {}
        self.analyzed_face_track_ids: Dict[int, float] = {}
        self.last_seen: Dict[int, float] = {}
        self.last_face_attempt: Dict[int, float] = {}
        self.last_plate_attempt: Dict[int, float] = {}
        self.pending_tasks: set[str] = set()
        self.pending_plate_futures: set[concurrent.futures.Future[Any]] = set()
        self.pending_face_futures: set[concurrent.futures.Future[Any]] = set()
        self.pending_db_futures: set[concurrent.futures.Future[Any]] = set()
        self.last_metric_write = 0.0

    @staticmethod
    def gpu_available() -> bool:
        return torch.cuda.is_available()

    def shutdown(self) -> None:
        self.plate_executor.shutdown(wait=False, cancel_futures=True)
        self.face_executor.shutdown(wait=False, cancel_futures=True)
        self.db_executor.shutdown(wait=False, cancel_futures=True)

    def snapshot_state(self, state: Dict[str, Any]) -> Dict[str, Any]:
        with self.state_lock:
            return copy.deepcopy(state)

    @staticmethod
    def _normalize_gender(value: Optional[str]) -> str:
        if not value:
            return "Unknown"
        normalized = value.strip().lower()
        if normalized in {"man", "male"}:
            return "Man"
        if normalized in {"woman", "female"}:
            return "Woman"
        return "Unknown"

    @staticmethod
    def _estimate_density(people_count: int, frame_width: int, frame_height: int) -> str:
        area_megapixels = max((frame_width * frame_height) / 1_000_000.0, 0.5)
        density_score = people_count / area_megapixels
        if density_score < 2.5:
            return "Low"
        if density_score < 5.5:
            return "Medium"
        return "High"

    @staticmethod
    def _format_clock(timestamp: Optional[float] = None) -> str:
        return time.strftime("%H:%M:%S", time.localtime(timestamp or time.time()))

    @staticmethod
    def _render_track_ttl(state: Dict[str, Any]) -> float:
        latency_seconds = max(float(state.get("inference_latency_ms") or 0.0) / 1000.0, 0.0)
        analytics_fps = max(float(state.get("analytics_fps") or 0.0), 0.0)
        analytics_interval = (1.0 / analytics_fps) if analytics_fps > 0.0 else 0.0
        ttl = max(
            RENDER_TRACK_TTL_SECONDS,
            latency_seconds * RENDER_TRACK_LATENCY_MULTIPLIER,
            analytics_interval * 2.0,
        )
        return min(ttl, RENDER_TRACK_MAX_TTL_SECONDS)

    @staticmethod
    def _watchlist_has_images(watchlist_dir: Path) -> bool:
        return watchlist_dir.exists() and any(
            path.suffix.lower() in WATCHLIST_SUFFIXES for path in watchlist_dir.iterdir() if path.is_file()
        )

    def _drop_future(self, bucket: set[concurrent.futures.Future[Any]], future: concurrent.futures.Future[Any]) -> None:
        with self.state_lock:
            bucket.discard(future)

    def _submit_executor_task(
        self,
        executor: concurrent.futures.ThreadPoolExecutor,
        future_bucket: set[concurrent.futures.Future[Any]],
        queue_limit: int,
        func,
        *args,
    ) -> bool:
        with self.state_lock:
            if len(future_bucket) >= queue_limit:
                return False
            try:
                future = executor.submit(func, *args)
            except RuntimeError:
                return False
            future_bucket.add(future)
        future.add_done_callback(lambda done_future: self._drop_future(future_bucket, done_future))
        return True

    def _submit_db_task(self, func, *args) -> None:
        try:
            self._submit_executor_task(self.db_executor, self.pending_db_futures, DB_QUEUE_LIMIT, func, *args)
        except RuntimeError:
            return

    def _push_recent(self, state: Dict[str, Any], key: str, item: Dict[str, Any], unique_field: str) -> None:
        bucket = state[key]
        bucket[:] = [existing for existing in bucket if existing.get(unique_field) != item.get(unique_field)]
        bucket.insert(0, item)
        del bucket[10:]

    def _append_event(self, state: Dict[str, Any], event_type: str, detail: str) -> None:
        event = {
            "camera_id": self.camera_id,
            "time": self._format_clock(),
            "type": event_type,
            "detail": detail,
        }
        state["event_logs"].insert(0, event)
        del state["event_logs"][50:]
        self._submit_db_task(log_event, self.camera_id, event_type, detail)

    @staticmethod
    def _expand_box(
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        width: int,
        height: int,
        scale_x: float = 0.08,
        scale_y: float = 0.12,
    ) -> tuple[int, int, int, int]:
        box_width = x2 - x1
        box_height = y2 - y1
        pad_x = int(box_width * scale_x)
        pad_y = int(box_height * scale_y)
        return (
            max(x1 - pad_x, 0),
            max(y1 - pad_y, 0),
            min(x2 + pad_x, width),
            min(y2 + pad_y, height),
        )

    @staticmethod
    def _prepare_plate_variants(region) -> list:
        if region.size == 0:
            return []

        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY) if region.ndim == 3 else region
        if gray.size == 0:
            return []

        scale = max(PLATE_OCR_TARGET_WIDTH / max(gray.shape[1], 1), 1.0)
        interpolation = cv2.INTER_LANCZOS4 if scale > 1.5 else cv2.INTER_CUBIC
        resized = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=interpolation)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(resized)
        filtered = cv2.bilateralFilter(clahe, 9, 75, 75)
        sharpened = cv2.addWeighted(filtered, 1.45, cv2.GaussianBlur(filtered, (0, 0), 2.0), -0.45, 0)
        adaptive = cv2.adaptiveThreshold(
            sharpened,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            11,
        )
        adaptive_inv = cv2.bitwise_not(adaptive)
        _, otsu = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        otsu_inv = cv2.bitwise_not(otsu)
        return [clahe, sharpened, adaptive, adaptive_inv, otsu, otsu_inv]

    @staticmethod
    def _plate_search_windows(vehicle_crop) -> list[tuple[Any, int, int]]:
        height, width = vehicle_crop.shape[:2]
        if height == 0 or width == 0:
            return []

        windows: list[tuple[Any, int, int]] = [(vehicle_crop, 0, 0)]
        if width >= 100 and height >= 50:
            top = int(height * 0.30)
            bottom = min(height, int(height * 0.92))
            left = int(width * 0.04)
            right = min(width, int(width * 0.96))
            focused = vehicle_crop[top:bottom, left:right]
            if focused.size:
                windows.append((focused, left, top))
        return windows

    def _candidate_plate_regions(self, vehicle_crop) -> list[tuple[Any, float]]:
        regions: list[tuple[Any, float]] = []
        height, width = vehicle_crop.shape[:2]
        if height == 0 or width == 0:
            return regions

        if self.plate_detector is not None:
            for window, offset_x, offset_y in self._plate_search_windows(vehicle_crop):
                try:
                    with SharedResources.plate_lock:
                        with torch.inference_mode():
                            plate_results = self.plate_detector.predict(
                                source=window,
                                conf=PLATE_CONFIDENCE,
                                imgsz=PLATE_IMAGE_SIZE,
                                verbose=False,
                                device=self.device,
                                half=(str(self.device) != "cpu"),
                            )
                    for box in plate_results[0].boxes:
                        confidence = float(box.conf[0])
                        if confidence < PLATE_CONFIDENCE:
                            continue
                        x1, y1, x2, y2 = [int(value) for value in box.xyxy[0].tolist()]
                        left, top, right, bottom = self._expand_box(
                            x1 + offset_x,
                            y1 + offset_y,
                            x2 + offset_x,
                            y2 + offset_y,
                            width,
                            height,
                        )
                        region = vehicle_crop[top:bottom, left:right]
                        if region.size == 0:
                            continue
                        regions.append((region, confidence))
                except Exception:
                    continue

        if regions:
            return regions

        if (
            width < PLATE_HEURISTIC_MIN_VEHICLE_WIDTH
            or height < PLATE_HEURISTIC_MIN_VEHICLE_HEIGHT
        ):
            return regions

        fallback_top = int(height * 0.48)
        fallback_bottom = min(height, int(height * 0.92))
        fallback_left = int(width * 0.08)
        fallback_right = min(width, int(width * 0.92))
        heuristic_region = vehicle_crop[fallback_top:fallback_bottom, fallback_left:fallback_right]
        if heuristic_region.size:
            regions.append((heuristic_region, 0.35))
        return regions

    def _extract_plate_and_mmc(self, vehicle_crop) -> Optional[Dict[str, Any]]:
        if vehicle_crop.size == 0 or not PLATE_RECOGNIZER_API_TOKEN:
            return None

        _, img_encoded = cv2.imencode('.jpg', vehicle_crop)
        
        for attempt in range(3):
            try:
                response = requests.post(
                    'https://api.platerecognizer.com/v1/plate-reader/',
                    headers={'Authorization': f'Token {PLATE_RECOGNIZER_API_TOKEN}'},
                    files={'upload': ('image.jpg', img_encoded.tobytes(), 'image/jpeg')},
                    data={'features': 'mmc', 'regions': 'in'}
                )
                if response.status_code == 429:
                    print(f"Plate Recognizer API Error: 429 Rate Limit. Backing off for {2 ** attempt}s...")
                    time.sleep(2 ** attempt)
                    continue
                response.raise_for_status()
                break # Success
            except requests.RequestException as e:
                print(f"Plate Recognizer API Error: {e}")
                if attempt == 2:
                    return None
        else:
            return None
            
        data = response.json()
        if not data.get('results'):
            return None
            
        result = data['results'][0]
        vehicle_props = result.get('vehicle', {}).get('props', {})
        
        def pick_attribute(key, fallback_key=None):
            # Try root level first (common in some API versions)
            val = result.get(key)
            if not val and fallback_key:
                val = result.get(fallback_key)
            # Try nested vehicle.props next (common in other API versions)
            if not val:
                val = vehicle_props.get(key)
            
            if isinstance(val, list) and len(val) > 0:
                return val[0].get('name', 'Unknown')
            return 'Unknown'

        make = pick_attribute('make')
        model = pick_attribute('make_model', fallback_key='model')
        color = pick_attribute('color')
        
        # Clean up model if it includes the make
        if model != 'Unknown' and make != 'Unknown' and make.lower() in model.lower():
            model = model[len(make):].strip().capitalize()
        
        return {
            "text": result.get('plate', '').upper(),
            "confidence": result.get('score', 0.0),
            "make": make.capitalize(),
            "model": model.capitalize() if model != "Unknown" else "",
            "color": color.capitalize()
        }

    def _match_watchlist(self, embedding) -> Optional[str]:
        if not self._watchlist_has_images(self.watchlist_dir) or embedding is None:
            return None

        if hasattr(self, '_cached_watchlist_embeddings') is False:
            self._cached_watchlist_embeddings = {}

        best_match_name = None
        best_distance = float('inf')
        threshold = 1.05 # L2 distance threshold for ArcFace

        for path in self.watchlist_dir.iterdir():
            if path.is_file() and path.suffix.lower() in WATCHLIST_SUFFIXES:
                match_name = path.stem
                if match_name not in self._cached_watchlist_embeddings:
                    target_img = cv2.imread(str(path))
                    if target_img is not None:
                        target_rgb = cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB)
                        with SharedResources.face_lock:
                            if self.face_analyzer:
                                faces = self.face_analyzer.get(target_rgb)
                                if faces:
                                    self._cached_watchlist_embeddings[match_name] = faces[0].embedding
                                else:
                                    self._cached_watchlist_embeddings[match_name] = None
                    else:
                        self._cached_watchlist_embeddings[match_name] = None

                target_embedding = self._cached_watchlist_embeddings.get(match_name)
                if target_embedding is not None:
                    dist = float(np.linalg.norm(embedding - target_embedding))
                    if dist < best_distance:
                        best_distance = dist
                        best_match_name = match_name

        if best_distance < threshold:
            return best_match_name
        return None

    def _schedule_plate_task(
        self,
        frame,
        state: Dict[str, Any],
        tracker_id: int,
        class_name: str,
        box: tuple[int, int, int, int],
    ) -> None:
        task_id = f"plate:{tracker_id}"
        now = time.time()
        with self.state_lock:
            if task_id in self.pending_tasks:
                return
            refresh_interval = (
                PLATE_REFRESH_INTERVAL_SECONDS if tracker_id in self.plate_results else PLATE_SCAN_INTERVAL_SECONDS
            )
            if now - self.last_plate_attempt.get(tracker_id, 0.0) < refresh_interval:
                return
            self.pending_tasks.add(task_id)
            self.last_plate_attempt[tracker_id] = now

        x1, y1, x2, y2 = box
        # Implement 15% padding for vehicle crops to improve MMC accuracy
        h, w = frame.shape[:2]
        px1, py1, px2, py2 = self._expand_box(x1, y1, x2, y2, w, h, scale_x=0.15, scale_y=0.15)
        vehicle_crop = frame[py1:py2, px1:px2].copy()
        
        if vehicle_crop.size == 0:
            with self.state_lock:
                self.pending_tasks.discard(task_id)
            return
        if not self._submit_executor_task(
            self.plate_executor,
            self.pending_plate_futures,
            PLATE_QUEUE_LIMIT,
            self._process_plate_async,
            vehicle_crop,
            state,
            tracker_id,
            class_name,
        ):
            with self.state_lock:
                self.pending_tasks.discard(task_id)

    def _schedule_face_task(
        self,
        frame,
        state: Dict[str, Any],
        tracker_id: int,
        box: tuple[int, int, int, int],
    ) -> None:
        task_id = f"face:{tracker_id}"
        now = time.time()
        with self.state_lock:
            if task_id in self.pending_tasks:
                return
            if now - self.last_face_attempt.get(tracker_id, 0.0) < FACE_SCAN_INTERVAL_SECONDS:
                return
            self.pending_tasks.add(task_id)
            self.last_face_attempt[tracker_id] = now

        x1, y1, x2, y2 = box
        person_crop = frame[max(y1, 0):max(y2, 0), max(x1, 0):max(x2, 0)].copy()
        if person_crop.size == 0:
            with self.state_lock:
                self.pending_tasks.discard(task_id)
            return
        if not self._submit_executor_task(
            self.face_executor,
            self.pending_face_futures,
            FACE_QUEUE_LIMIT,
            self._process_face_async,
            person_crop,
            state,
            tracker_id,
        ):
            with self.state_lock:
                self.pending_tasks.discard(task_id)

    def _process_plate_async(
        self,
        vehicle_crop,
        state: Dict[str, Any],
        tracker_id: int,
        class_name: str,
    ) -> None:
        try:
            mmc_data = self._extract_plate_and_mmc(vehicle_crop)
            if not mmc_data or not mmc_data["text"]:
                return

            plate_text = mmc_data["text"]
            plate_confidence = mmc_data["confidence"]
            vehicle_type_enriched = f"{mmc_data['color']} {mmc_data['make']} {mmc_data['model']} ({class_name.capitalize()})" if mmc_data['make'] != 'Unknown' else class_name

            with self.state_lock:
                now = time.time()
                
                # 1. Vote on Plate Text
                vote_bucket = self.plate_votes.setdefault(tracker_id, {})
                vote = vote_bucket.setdefault(
                    plate_text,
                    {"hits": 0.0, "score": 0.0, "best_confidence": 0.0},
                )
                vote["hits"] += 1.0
                vote["score"] += float(plate_confidence or 0.0)
                vote["best_confidence"] = max(vote["best_confidence"], float(plate_confidence or 0.0))

                best_text, best_vote = max(
                    vote_bucket.items(),
                    key=lambda item: (item[1]["hits"], item[1]["score"], item[1]["best_confidence"]),
                )
                
                # 2. Vote on MMC (Make, Model, Color)
                # We vote on the tuple to ensure attribute consistency (e.g. no "Blue Red Toyota")
                mmc_tuple = (mmc_data["make"], mmc_data["model"], mmc_data["color"])
                mmc_bucket = self.mmc_votes.setdefault(tracker_id, {})
                m_vote = mmc_bucket.setdefault(
                    str(mmc_tuple), 
                    {"hits": 0, "make": mmc_data["make"], "model": mmc_data["model"], "color": mmc_data["color"]}
                )
                m_vote["hits"] += 1
                
                best_mmc_raw = max(mmc_bucket.values(), key=lambda v: v["hits"])
                
                # 3. Check Confirmations
                plate_confirmed = (
                    best_vote["hits"] >= PLATE_CONFIRMATION_HITS
                    or best_vote["best_confidence"] >= PLATE_DIRECT_ACCEPT_CONFIDENCE
                    or best_vote["score"] >= PLATE_MIN_AGGREGATE_SCORE
                )
                
                # MMC consensus: require at least 2 hits if we have low confidence, or just take the best after 1 if plate is confirmed.
                # However, for stability, we'll prefer the most frequent MMC.
                v_make, v_model, v_color = best_mmc_raw["make"], best_mmc_raw["model"], best_mmc_raw["color"]
                
                if not plate_confirmed:
                    return

                # Construct enriched string from VOTED MMC
                if v_make != "Unknown":
                    vehicle_type_enriched = f"{v_color} {v_make} {v_model} ({class_name.capitalize()})".replace("  ", " ").strip()
                else:
                    vehicle_type_enriched = class_name.capitalize()

                existing_text = self.plate_results.get(tracker_id, {}).get("text")
                if existing_text == best_text:
                    self.plate_results[tracker_id]["expires"] = now + 8.0
                    touch_timestamp_cache(self.detected_plate_texts, best_text, now)
                    return

                if best_text not in self.detected_plate_texts:
                    state["plates_detected"] += 1
                touch_timestamp_cache(self.detected_plate_texts, best_text, now)

                self.plate_results[tracker_id] = {
                    "text": best_text,
                    "confidence": round(best_vote["best_confidence"], 3),
                    "make": v_make,
                    "model": v_model,
                    "color": v_color,
                    "expires": now + 8.0,
                }
                self._push_recent(
                    state,
                    "recent_plates",
                    {
                        "tracker_id": tracker_id,
                        "plate_text": best_text,
                        "vehicle_type": vehicle_type_enriched,
                        "confidence": round(best_vote["best_confidence"], 3),
                        "time": self._format_clock(),
                    },
                    "plate_text",
                )
                self._push_recent(
                    state,
                    "recent_vehicles",
                    {
                        "tracker_id": tracker_id,
                        "vehicle_type": vehicle_type_enriched,
                        "plate_text": best_text,
                        "time": self._format_clock(),
                    },
                    "tracker_id",
                )
                detail = f"Plate '{best_text}' detected on {class_name} #{tracker_id}"
                self._append_event(state, "ANPR Match", detail)

            self._submit_db_task(
                upsert_plate_read,
                self.camera_id,
                tracker_id,
                best_text,
                vehicle_type_enriched,
                round(best_vote["best_confidence"], 3),
            )
            self._submit_db_task(
                upsert_vehicle_record,
                self.camera_id,
                tracker_id,
                vehicle_type_enriched,
                best_text,
            )
        finally:
            with self.state_lock:
                self.pending_tasks.discard(f"plate:{tracker_id}")

    def _process_face_async(self, person_crop, state: Dict[str, Any], tracker_id: int) -> None:
        try:
            if person_crop.size == 0 or not self.face_analyzer:
                return

            upper_crop = person_crop[: max(int(person_crop.shape[0] * 0.55), 1), :]
            face_rgb = cv2.cvtColor(upper_crop, cv2.COLOR_BGR2RGB)

            with SharedResources.face_lock:
                faces = self.face_analyzer.get(face_rgb)
                
            if not faces:
                return

            best_face = faces[0]
            bbox = best_face.bbox.astype(int)
            fx, fy, fx2, fy2 = bbox
            fw, fh = fx2 - fx, fy2 - fy

            gender = "Man" if best_face.gender == 1 else "Woman"
            match_name = self._match_watchlist(best_face.embedding)
            watchlist_hit = bool(match_name)

            with self.state_lock:
                now = time.time()
                cached_face = self.face_results.get(tracker_id)
                previous_gender = cached_face.get("gender") if cached_face else None
                self.face_results[tracker_id] = {
                    "gender": gender,
                    "match_name": match_name,
                    "watchlist_hit": watchlist_hit,
                    "face_box": (fx, fy, fw, fh),
                    "expires": now + 8.0,
                }
                if tracker_id not in self.analyzed_face_track_ids:
                    state["faces_detected"] += 1
                    if gender in state["gender_stats"]:
                        state["gender_stats"][gender] += 1
                elif previous_gender is None and gender in state["gender_stats"]:
                    state["gender_stats"][gender] += 1
                touch_timestamp_cache(self.analyzed_face_track_ids, tracker_id, now)

                face_record = {
                    "tracker_id": tracker_id,
                    "identity": match_name or "Anonymous",
                    "gender": gender,
                    "watchlist_hit": watchlist_hit,
                    "time": self._format_clock(),
                }
                self._push_recent(state, "recent_faces", face_record, "tracker_id")
                if watchlist_hit:
                    detail = f"Watchlist match '{match_name}' on person #{tracker_id}"
                    self._append_event(state, "Watchlist Alert", detail)
                else:
                    detail = f"Face analytics completed for person #{tracker_id} ({gender})"
                    self._append_event(state, "Face Analytics", detail)

            self._submit_db_task(
                upsert_face_record,
                self.camera_id,
                tracker_id,
                match_name,
                gender,
                None,
                watchlist_hit,
            )
        finally:
            with self.state_lock:
                self.pending_tasks.discard(f"face:{tracker_id}")

    def _cleanup_expired_cache(self) -> None:
        now = time.time()
        stale_track_ids = [track_id for track_id, seen_at in self.last_seen.items() if now - seen_at > TRACK_STALE_SECONDS]
        for track_id in stale_track_ids:
            self.last_seen.pop(track_id, None)
            self.track_states.pop(track_id, None)
            self.render_tracks.pop(track_id, None)
            self.face_results.pop(track_id, None)
            self.plate_results.pop(track_id, None)
            self.plate_votes.pop(track_id, None)
            self.mmc_votes.pop(track_id, None)
            self.last_face_attempt.pop(track_id, None)
            self.last_plate_attempt.pop(track_id, None)
            self.analyzed_face_track_ids.pop(track_id, None)

        self.render_tracks = {
            track_id: result
            for track_id, result in self.render_tracks.items()
            if result.get("expires", 0.0) > now
        }
        self.face_results = {
            track_id: result
            for track_id, result in self.face_results.items()
            if result.get("expires", 0.0) > now
        }
        self.plate_results = {
            track_id: result
            for track_id, result in self.plate_results.items()
            if result.get("expires", 0.0) > now
        }
        trim_timestamp_cache(self.detected_plate_texts, now, UNIQUE_CACHE_TTL_SECONDS, PLATE_CACHE_LIMIT)
        trim_timestamp_cache(self.analyzed_face_track_ids, now, UNIQUE_CACHE_TTL_SECONDS, FACE_TRACK_CACHE_LIMIT)

    @staticmethod
    def _clip_box(frame, box: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
        x1, y1, x2, y2 = box
        return (
            max(x1, 0),
            max(y1, 0),
            min(x2, frame.shape[1]),
            min(y2, frame.shape[0]),
        )

    def _annotate_face(self, frame, box: tuple[int, int, int, int], face_result: Optional[Dict[str, Any]]) -> None:
        x1, y1, x2, y2 = box
        if face_result and face_result.get("face_box"):
            fx, fy, fw, fh = face_result["face_box"]
            face_left = x1 + fx
            face_top = y1 + fy
            face_right = face_left + fw
            face_bottom = face_top + fh
        else:
            width = x2 - x1
            height = y2 - y1
            face_left = x1 + int(width * 0.2)
            face_top = y1
            face_right = x1 + int(width * 0.8)
            face_bottom = y1 + int(height * 0.3)

        face_left = max(face_left, 0)
        face_top = max(face_top, 0)
        face_right = min(face_right, frame.shape[1])
        face_bottom = min(face_bottom, frame.shape[0])

        if face_right <= face_left or face_bottom <= face_top:
            return

        if not face_result or not face_result.get("watchlist_hit"):
            face_region = frame[face_top:face_bottom, face_left:face_right]
            if face_region.size:
                frame[face_top:face_bottom, face_left:face_right] = cv2.GaussianBlur(face_region, (51, 51), 20)

        if face_result and face_result.get("watchlist_hit"):
            label = f"WATCHLIST: {face_result['match_name']}"
            color = (0, 0, 255)
        elif face_result and face_result.get("gender"):
            label = face_result["gender"]
            color = (255, 0, 255)
        else:
            label = "ANONYMIZED"
            color = (255, 0, 255)

        self._draw_label(frame, label, (x1, max(y1 - 12, 20)), color)

    @staticmethod
    def _draw_label(frame, text: str, anchor: tuple[int, int], color: tuple[int, int, int]) -> None:
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.55
        thickness = 2
        (text_width, text_height), baseline = cv2.getTextSize(text, font, scale, thickness)
        x = max(anchor[0], 0)
        y = max(anchor[1], text_height + baseline + 6)

        left = max(x - 4, 0)
        top = max(y - text_height - baseline - 8, 0)
        right = min(x + text_width + 8, frame.shape[1] - 1)
        bottom = min(y + 2, frame.shape[0] - 1)
        cv2.rectangle(frame, (left, top), (right, bottom), color, -1)
        cv2.putText(frame, text, (x, y - 4), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)

    @staticmethod
    def _draw_zone(frame) -> int:
        frame_height, frame_width = frame.shape[:2]
        zone_y = int(frame_height * 0.55)
        cv2.line(frame, (0, zone_y), (frame_width, zone_y), (0, 165, 255), 2)
        cv2.putText(
            frame,
            "Analytics Zone",
            (12, max(zone_y - 12, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 165, 255),
            2,
        )
        return zone_y

    def _draw_scene(self, frame):
        self._draw_zone(frame)
        with self.state_lock:
            render_tracks = copy.deepcopy(self.render_tracks)
            face_results = copy.deepcopy(self.face_results)
            plate_results = copy.deepcopy(self.plate_results)

        for tracker_id in sorted(render_tracks.keys()):
            render_track = render_tracks[tracker_id]
            box = self._clip_box(frame, render_track["box"])
            x1, y1, x2, y2 = box
            if x2 <= x1 or y2 <= y1:
                continue

            class_name = render_track["class_name"]
            confidence = render_track["confidence"]
            if class_name in VEHICLE_CLASSES:
                label = f"#{tracker_id} {class_name} {confidence:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (59, 130, 246), 2)
                self._draw_label(frame, label, (x1, max(y1 - 10, 20)), (59, 130, 246))

                plate_result = plate_results.get(tracker_id)
                if plate_result:
                    self._draw_label(
                        frame,
                        plate_result["text"],
                        (x1, min(y2 + 26, frame.shape[0] - 8)),
                        (0, 180, 255),
                    )
            else:
                face_result = face_results.get(tracker_id)
                self._annotate_face(frame, box, face_result)

                color = (0, 0, 255) if face_result and face_result.get("watchlist_hit") else (34, 197, 94)
                label = f"#{tracker_id} person {confidence:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                self._draw_label(frame, label, (x1, min(y2 + 26, frame.shape[0] - 8)), color)

        return frame

    def render_frame(self, frame):
        with self.state_lock:
            self._cleanup_expired_cache()
        return self._draw_scene(frame)

    def refresh_track_tasks(self, frame, state: Dict[str, Any]) -> None:
        with self.state_lock:
            self._cleanup_expired_cache()
            render_tracks = copy.deepcopy(self.render_tracks)
            track_states = copy.deepcopy(self.track_states)

        for tracker_id, render_track in render_tracks.items():
            track_state = track_states.get(tracker_id, {})
            if track_state.get("frames_seen", 0) < MIN_STABLE_FRAMES:
                continue

            box = self._clip_box(frame, render_track["box"])
            x1, y1, x2, y2 = box
            if x2 <= x1 or y2 <= y1:
                continue

            class_name = render_track["class_name"]
            if class_name in VEHICLE_CLASSES:
                if (x2 - x1) >= 90 and (y2 - y1) >= 70:
                    self._schedule_plate_task(frame, state, tracker_id, class_name, box)
            elif (x2 - x1) >= 90 and (y2 - y1) >= 120:
                self._schedule_face_task(frame, state, tracker_id, box)

    def process_frame(self, frame, state: Dict[str, Any]):
        frame_height, frame_width = frame.shape[:2]
        zone_y = int(frame_height * 0.55)
        now = time.time()
        render_track_ttl = self._render_track_ttl(state)

        with SharedResources.detector_lock:
            with torch.inference_mode():
                results = self.detector.predict(
                    source=frame,
                    conf=DETECTION_CONFIDENCE,
                    imgsz=DETECTION_IMAGE_SIZE,
                    classes=list(TARGET_CLASSES.keys()),
                    device=self.device,
                    half=(str(self.device) != "cpu"),
                    verbose=False,
                )

        detections = sv.Detections.from_ultralytics(results[0])
        if len(detections) > 0:
            detections = detections[detections.confidence >= DETECTION_CONFIDENCE]
        detections = self.tracker.update_with_detections(detections)

        current_vehicle_ids: set[int] = set()
        current_people_ids: set[int] = set()
        current_vehicle_types = {vehicle_type: 0 for vehicle_type in VEHICLE_CLASSES}
        zone_count = 0

        with self.state_lock:
            for index in range(len(detections)):
                tracker_id = None
                if detections.tracker_id is not None:
                    tracker_id = int(detections.tracker_id[index])
                if tracker_id is None:
                    continue

                x1, y1, x2, y2 = [int(value) for value in detections.xyxy[index].tolist()]
                class_id = int(detections.class_id[index])
                confidence = float(detections.confidence[index])
                class_name = TARGET_CLASSES.get(class_id)
                if class_name is None:
                    continue

                self.last_seen[tracker_id] = now
                self.render_tracks[tracker_id] = {
                    "box": (x1, y1, x2, y2),
                    "class_name": class_name,
                    "confidence": confidence,
                    "expires": now + render_track_ttl,
                }
                track_state = self.track_states.setdefault(
                    tracker_id,
                    {
                        "frames_seen": 0,
                        "person_recorded": False,
                        "vehicle_recorded": False,
                        "vehicle_db_recorded": False,
                    },
                )
                track_state["frames_seen"] += 1
                center_y = (y1 + y2) // 2
                is_stable_track = track_state["frames_seen"] >= MIN_STABLE_FRAMES
                if is_stable_track and center_y >= zone_y:
                    zone_count += 1

                box = (x1, y1, x2, y2)
                if class_name in VEHICLE_CLASSES:
                    if is_stable_track:
                        current_vehicle_ids.add(tracker_id)
                        current_vehicle_types[class_name] += 1
                        if not track_state["vehicle_db_recorded"]:
                            track_state["vehicle_db_recorded"] = True
                            self._submit_db_task(
                                upsert_vehicle_record,
                                self.camera_id,
                                tracker_id,
                                class_name,
                            )
                        if not track_state["vehicle_recorded"]:
                            track_state["vehicle_recorded"] = True
                            state["vehicle_total_count"] += 1
                            state["vehicle_types"][class_name] += 1
                            self._push_recent(
                                state,
                                "recent_vehicles",
                                {
                                    "tracker_id": tracker_id,
                                    "vehicle_type": class_name,
                                    "plate_text": None,
                                    "time": self._format_clock(),
                                },
                                "tracker_id",
                            )

                    if is_stable_track and (x2 - x1) >= 160 and (y2 - y1) >= 120:
                        self._schedule_plate_task(frame, state, tracker_id, class_name, box)
                else:
                    if is_stable_track:
                        current_people_ids.add(tracker_id)
                    if is_stable_track and not track_state["person_recorded"]:
                        track_state["person_recorded"] = True
                        state["people_total_count"] += 1

                    if is_stable_track and (x2 - x1) >= 90 and (y2 - y1) >= 120:
                        self._schedule_face_task(frame, state, tracker_id, box)

            state["vehicle_count"] = len(current_vehicle_ids)
            state["people_count"] = len(current_people_ids)
            state["vehicle_current_types"] = current_vehicle_types
            state["zone_count"] = zone_count
            state["crowd_density"] = self._estimate_density(
                state["people_count"],
                frame_width,
                frame_height,
            )
            state["last_updated"] = self._format_clock()
            state["plate_detector_ready"] = self.plate_detector is not None
            state["detector_model"] = DETECTION_MODEL_NAME
            state["plate_model"] = PLATE_MODEL_NAME
            state["device"] = "cuda" if str(self.device) != "cpu" else "cpu"

            if time.time() - self.last_metric_write >= METRIC_WRITE_INTERVAL_SECONDS:
                self.last_metric_write = time.time()
                self._submit_db_task(
                    store_metric,
                    self.camera_id,
                    state["vehicle_count"],
                    state["people_count"],
                    state["zone_count"],
                )

            self._cleanup_expired_cache()

        return self._draw_scene(frame)
