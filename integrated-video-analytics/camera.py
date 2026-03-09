from __future__ import annotations

import threading
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

    def _update_live(self) -> None:
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                self.ret = ret
                self.frame = frame
            else:
                self.running = False

    def read(self):
        if not self.running:
            return False, None

        if self.is_live:
            return self.ret, self.frame

        if self._pending_first_frame:
            self._pending_first_frame = False
            return self.ret, self.frame

        self.ret, self.frame = self.cap.read()
        if not self.ret:
            self.running = False
            return False, None
        return self.ret, self.frame

    def release(self) -> None:
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=1.0)
        self.cap.release()
