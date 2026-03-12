from __future__ import annotations

import math
import threading
import time
from typing import Any

import cv2


class CameraStream:
    """Handle both live sources and uploaded video files with predictable frame reads."""

    def __init__(self, source: Any):
        self.source = self._normalize_source(source)
        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video source {source}")

        self.is_live = self._is_live_source(self.source)
        if self.is_live:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 3000)
            self.cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 3000)

        self.frame_lock = threading.Lock()
        self.source_fps = self._read_source_fps()
        self.ret, self.frame = self.cap.read()
        self.running = bool(self.ret)
        self._pending_first_frame = bool(self.ret)
        self.thread: threading.Thread | None = None

        if self.is_live:
            self.thread = threading.Thread(target=self._update_live, daemon=True)
            self.thread.start()

    @staticmethod
    def _normalize_source(source: Any) -> Any:
        if isinstance(source, str) and source.isdigit():
            return int(source)
        return source

    @staticmethod
    def _is_live_source(source: Any) -> bool:
        if isinstance(source, int):
            return True
        source_str = str(source).lower()
        return source_str.startswith("rtsp://") or source_str.startswith("http://") or source_str.startswith("https://")

    def _read_source_fps(self) -> float:
        fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if not math.isfinite(fps) or fps < 1.0 or fps > 120.0:
            return 0.0
        return fps

    @property
    def fps(self) -> float:
        if self.source_fps > 0.0:
            return self.source_fps
        return 20.0 if self.is_live else 24.0

    def _update_live(self) -> None:
        failures = 0
        while self.running:
            try:
                ret, frame = self.cap.read()
            except cv2.error:
                ret, frame = False, None
            if ret:
                with self.frame_lock:
                    self.ret = ret
                    self.frame = frame
                failures = 0
            else:
                failures += 1
                if failures >= 30:
                    self.running = False
                else:
                    time.sleep(0.05)

    def read(self):
        if not self.running:
            return False, None

        if self.is_live:
            with self.frame_lock:
                if self.frame is None:
                    return False, None
                return self.ret, self.frame.copy()

        if self._pending_first_frame:
            self._pending_first_frame = False
            if self.frame is None:
                return False, None
            return self.ret, self.frame.copy()

        self.ret, self.frame = self.cap.read()
        if not self.ret:
            self.running = False
            return False, None
        return self.ret, self.frame.copy()

    def release(self) -> None:
        self.running = False
        self.cap.release()
        if self.thread is not None:
            self.thread.join(timeout=1.0)
