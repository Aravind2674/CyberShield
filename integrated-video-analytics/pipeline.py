from __future__ import annotations

import concurrent.futures
import copy
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import easyocr
import supervision as sv
import torch
from deepface import DeepFace
from ultralytics import YOLO

from database import (
    log_event,
    store_metric,
    upsert_face_record,
    upsert_plate_read,
    upsert_vehicle_record,
)

TARGET_CLASSES = {
    0: "person",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}
VEHICLE_CLASSES = {"car", "motorcycle", "bus", "truck"}
DETECTION_CONFIDENCE = float(os.getenv("CYBERSHIELD_DETECT_CONFIDENCE", "0.35"))
PLATE_CONFIDENCE = float(os.getenv("CYBERSHIELD_PLATE_CONFIDENCE", "0.35"))
DETECTION_IMAGE_SIZE = int(os.getenv("CYBERSHIELD_DETECT_IMGSZ", "960"))
MIN_STABLE_FRAMES = int(os.getenv("CYBERSHIELD_MIN_STABLE_FRAMES", "3"))
DETECTION_MODEL_NAME = os.getenv(
    "CYBERSHIELD_DETECT_MODEL",
    "yolov8s.pt" if torch.cuda.is_available() else "yolov8n.pt",
)
PLATE_MODEL_NAME = os.getenv(
    "CYBERSHIELD_PLATE_MODEL",
    "https://huggingface.co/yasirfaizahmed/license-plate-object-detection/resolve/main/best.pt",
)
WATCHLIST_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class SharedResources:
    _detector: Optional[YOLO] = None
    _plate_detector: Optional[YOLO] = None
    _ocr_reader: Optional[easyocr.Reader] = None
    _face_cascade: Optional[cv2.CascadeClassifier] = None

    detector_lock = threading.Lock()
    plate_lock = threading.Lock()
    ocr_lock = threading.Lock()
    deepface_lock = threading.Lock()
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
                    print(f"Plate detector unavailable, using OCR fallback only: {exc}")
                    cls._plate_detector = None
            return cls._plate_detector

    @classmethod
    def get_ocr_reader(cls) -> easyocr.Reader:
        with cls.init_lock:
            if cls._ocr_reader is None:
                print("Loading EasyOCR reader")
                cls._ocr_reader = easyocr.Reader(["en"], gpu=torch.cuda.is_available())
            return cls._ocr_reader

    @classmethod
    def get_face_cascade(cls) -> cv2.CascadeClassifier:
        with cls.init_lock:
            if cls._face_cascade is None:
                cls._face_cascade = cv2.CascadeClassifier(
                    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
                )
            return cls._face_cascade


class VideoPipeline:
    def __init__(self, camera_id: str):
        self.camera_id = camera_id
        self.device = 0 if torch.cuda.is_available() else "cpu"
        self.detector = SharedResources.get_detector()
        self.plate_detector = SharedResources.get_plate_detector()
        self.reader = SharedResources.get_ocr_reader()
        self.face_cascade = SharedResources.get_face_cascade()

        self.tracker = sv.ByteTrack()
        self.state_lock = threading.RLock()
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)
        self.watchlist_dir = Path(__file__).resolve().parent / "watchlist"

        self.track_states: Dict[int, Dict[str, Any]] = {}
        self.face_results: Dict[int, Dict[str, Any]] = {}
        self.plate_results: Dict[int, Dict[str, Any]] = {}
        self.detected_plate_texts: set[str] = set()
        self.analyzed_face_track_ids: set[int] = set()
        self.last_seen: Dict[int, float] = {}
        self.last_face_attempt: Dict[int, float] = {}
        self.last_plate_attempt: Dict[int, float] = {}
        self.pending_tasks: set[str] = set()
        self.last_metric_write = 0.0

    def shutdown(self) -> None:
        self.executor.shutdown(wait=False, cancel_futures=True)

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
    def _watchlist_has_images(watchlist_dir: Path) -> bool:
        return watchlist_dir.exists() and any(
            path.suffix.lower() in WATCHLIST_SUFFIXES for path in watchlist_dir.iterdir() if path.is_file()
        )

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
        log_event(self.camera_id, event_type, detail)

    def _extract_plate_text(self, vehicle_crop) -> tuple[Optional[str], Optional[float]]:
        candidate_regions = []
        if self.plate_detector is not None:
            try:
                with SharedResources.plate_lock:
                    plate_results = self.plate_detector.predict(
                        source=vehicle_crop,
                        conf=PLATE_CONFIDENCE,
                        verbose=False,
                        device=self.device,
                    )
                for box in plate_results[0].boxes:
                    confidence = float(box.conf[0])
                    if confidence < PLATE_CONFIDENCE:
                        continue
                    x1, y1, x2, y2 = [int(value) for value in box.xyxy[0].tolist()]
                    candidate_regions.append(vehicle_crop[max(y1, 0):max(y2, 0), max(x1, 0):max(x2, 0)])
            except Exception:
                candidate_regions = []

        if not candidate_regions:
            height, width = vehicle_crop.shape[:2]
            candidate_regions.append(vehicle_crop[int(height * 0.55):height, 0:width])

        for region in candidate_regions:
            if region.size == 0:
                continue
            gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            with SharedResources.ocr_lock:
                ocr_results = self.reader.readtext(enhanced)
            for _, text, probability in ocr_results:
                cleaned = "".join(ch for ch in text if ch.isalnum()).upper()
                if probability >= 0.45 and len(cleaned) >= 5:
                    return cleaned, float(probability)

        return None, None

    def _match_watchlist(self, face_rgb) -> Optional[str]:
        if not self._watchlist_has_images(self.watchlist_dir):
            return None

        with SharedResources.deepface_lock:
            matches = DeepFace.find(
                img_path=face_rgb,
                db_path=str(self.watchlist_dir),
                enforce_detection=False,
                silent=True,
            )

        if not matches or len(matches[0]) == 0:
            return None

        best = matches[0].iloc[0]
        distance = float(best.get("distance", 1.0))
        threshold = float(best.get("threshold", 0.68))
        if distance > threshold:
            return None

        identity_path = Path(best["identity"])
        return identity_path.stem

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
            if now - self.last_plate_attempt.get(tracker_id, 0.0) < 8.0:
                return
            self.pending_tasks.add(task_id)
            self.last_plate_attempt[tracker_id] = now

        x1, y1, x2, y2 = box
        vehicle_crop = frame[max(y1, 0):max(y2, 0), max(x1, 0):max(x2, 0)].copy()
        self.executor.submit(self._process_plate_async, vehicle_crop, state, tracker_id, class_name)

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
            if now - self.last_face_attempt.get(tracker_id, 0.0) < 5.0:
                return
            self.pending_tasks.add(task_id)
            self.last_face_attempt[tracker_id] = now

        x1, y1, x2, y2 = box
        person_crop = frame[max(y1, 0):max(y2, 0), max(x1, 0):max(x2, 0)].copy()
        self.executor.submit(self._process_face_async, person_crop, state, tracker_id)

    def _process_plate_async(
        self,
        vehicle_crop,
        state: Dict[str, Any],
        tracker_id: int,
        class_name: str,
    ) -> None:
        try:
            plate_text, plate_confidence = self._extract_plate_text(vehicle_crop)
            if not plate_text:
                return

            with self.state_lock:
                existing_text = self.plate_results.get(tracker_id, {}).get("text")
                if existing_text == plate_text:
                    self.plate_results[tracker_id]["expires"] = time.time() + 6.0
                    return

                if plate_text not in self.detected_plate_texts:
                    self.detected_plate_texts.add(plate_text)
                    state["plates_detected"] += 1

                self.plate_results[tracker_id] = {
                    "text": plate_text,
                    "expires": time.time() + 6.0,
                }
                self._push_recent(
                    state,
                    "recent_plates",
                    {
                        "tracker_id": tracker_id,
                        "plate_text": plate_text,
                        "vehicle_type": class_name,
                        "confidence": round(plate_confidence or 0.0, 3),
                        "time": self._format_clock(),
                    },
                    "plate_text",
                )
                detail = f"Plate '{plate_text}' detected on {class_name} #{tracker_id}"
                self._append_event(state, "ANPR Match", detail)

            upsert_plate_read(self.camera_id, tracker_id, plate_text, class_name, plate_confidence)
            upsert_vehicle_record(self.camera_id, tracker_id, class_name, plate_text)
        finally:
            with self.state_lock:
                self.pending_tasks.discard(f"plate:{tracker_id}")

    def _process_face_async(self, person_crop, state: Dict[str, Any], tracker_id: int) -> None:
        try:
            if person_crop.size == 0:
                return

            upper_crop = person_crop[: max(int(person_crop.shape[0] * 0.55), 1), :]
            with SharedResources.deepface_lock:
                extracted_faces = DeepFace.extract_faces(
                    img_path=cv2.cvtColor(upper_crop, cv2.COLOR_BGR2RGB),
                    detector_backend="opencv",
                    enforce_detection=False,
                    align=True,
                )
            if not extracted_faces:
                return

            facial_area = max(
                (face.get("facial_area", {}) for face in extracted_faces),
                key=lambda area: area.get("w", 0) * area.get("h", 0),
            )
            fx = int(facial_area.get("x", 0))
            fy = int(facial_area.get("y", 0))
            fw = int(facial_area.get("w", 0))
            fh = int(facial_area.get("h", 0))
            if fw <= 0 or fh <= 0:
                return

            face_bgr = upper_crop[fy:fy + fh, fx:fx + fw]
            if face_bgr.size == 0:
                return
            face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)

            with SharedResources.deepface_lock:
                analysis = DeepFace.analyze(
                    img_path=face_rgb,
                    actions=["gender"],
                    enforce_detection=False,
                    silent=True,
                )

            if isinstance(analysis, list):
                analysis = analysis[0]
            gender = self._normalize_gender(analysis.get("dominant_gender"))
            match_name = self._match_watchlist(face_rgb)
            watchlist_hit = bool(match_name)

            with self.state_lock:
                cached_face = self.face_results.get(tracker_id)
                previous_gender = cached_face.get("gender") if cached_face else None
                self.face_results[tracker_id] = {
                    "gender": gender,
                    "match_name": match_name,
                    "watchlist_hit": watchlist_hit,
                    "face_box": (fx, fy, fw, fh),
                    "expires": time.time() + 8.0,
                }
                if tracker_id not in self.analyzed_face_track_ids:
                    self.analyzed_face_track_ids.add(tracker_id)
                    state["faces_detected"] += 1
                    if gender in state["gender_stats"]:
                        state["gender_stats"][gender] += 1
                elif previous_gender is None and gender in state["gender_stats"]:
                    state["gender_stats"][gender] += 1

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

            upsert_face_record(self.camera_id, tracker_id, match_name, gender, None, watchlist_hit)
        finally:
            with self.state_lock:
                self.pending_tasks.discard(f"face:{tracker_id}")

    def _cleanup_expired_cache(self) -> None:
        now = time.time()
        stale_track_ids = [track_id for track_id, seen_at in self.last_seen.items() if now - seen_at > 15.0]
        for track_id in stale_track_ids:
            self.last_seen.pop(track_id, None)
            self.track_states.pop(track_id, None)
            self.face_results.pop(track_id, None)
            self.plate_results.pop(track_id, None)
            self.last_face_attempt.pop(track_id, None)
            self.last_plate_attempt.pop(track_id, None)

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
            cv2.putText(
                frame,
                label,
                (x1, max(y1 - 12, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
            )
        elif face_result and face_result.get("gender"):
            label = face_result["gender"]
            cv2.putText(
                frame,
                label,
                (x1, max(y1 - 12, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 0, 255),
                2,
            )
        else:
            cv2.putText(
                frame,
                "ANONYMIZED",
                (x1, max(y1 - 12, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 0, 255),
                2,
            )

    def process_frame(self, frame, state: Dict[str, Any]):
        frame_height, frame_width = frame.shape[:2]
        zone_y = int(frame_height * 0.55)
        now = time.time()

        with SharedResources.detector_lock:
            results = self.detector.predict(
                source=frame,
                conf=DETECTION_CONFIDENCE,
                imgsz=DETECTION_IMAGE_SIZE,
                classes=list(TARGET_CLASSES.keys()),
                device=self.device,
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
                track_state = self.track_states.setdefault(
                    tracker_id,
                    {
                        "frames_seen": 0,
                        "last_side": None,
                        "vehicle_counted": False,
                        "person_recorded": False,
                        "vehicle_recorded": False,
                        "vehicle_db_recorded": False,
                    },
                )
                track_state["frames_seen"] += 1
                center_y = (y1 + y2) // 2
                side = "below" if center_y >= zone_y else "above"
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
                            upsert_vehicle_record(self.camera_id, tracker_id, class_name)
                        if not track_state["vehicle_recorded"]:
                            track_state["vehicle_recorded"] = True
                            state["vehicle_total_count"] += 1
                            state["vehicle_types"][class_name] += 1
                    track_state["last_side"] = side

                    label = f"#{tracker_id} {class_name} {confidence:.2f}"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (59, 130, 246), 2)
                    cv2.putText(
                        frame,
                        label,
                        (x1, max(y1 - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (59, 130, 246),
                        2,
                    )

                    plate_result = self.plate_results.get(tracker_id)
                    if plate_result:
                        cv2.putText(
                            frame,
                            plate_result["text"],
                            (x1, min(y2 + 22, frame_height - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.65,
                            (0, 255, 255),
                            2,
                        )

                    if is_stable_track and (x2 - x1) >= 70 and (y2 - y1) >= 70:
                        self._schedule_plate_task(frame, state, tracker_id, class_name, box)
                else:
                    if is_stable_track:
                        current_people_ids.add(tracker_id)
                    if is_stable_track and not track_state["person_recorded"]:
                        track_state["person_recorded"] = True
                        state["people_total_count"] += 1

                    face_result = self.face_results.get(tracker_id)
                    self._annotate_face(frame, box, face_result)

                    color = (0, 0, 255) if face_result and face_result.get("watchlist_hit") else (34, 197, 94)
                    label = f"#{tracker_id} person {confidence:.2f}"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(
                        frame,
                        label,
                        (x1, min(y2 + 22, frame_height - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        color,
                        2,
                    )

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

            if time.time() - self.last_metric_write >= 2.0:
                self.last_metric_write = time.time()
                store_metric(
                    self.camera_id,
                    state["vehicle_count"],
                    state["people_count"],
                    state["zone_count"],
                )

            self._cleanup_expired_cache()

        return frame
