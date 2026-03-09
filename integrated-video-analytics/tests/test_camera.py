from pathlib import Path

import cv2
import numpy as np

from camera import CameraStream


def _write_test_video(video_path: Path) -> None:
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(str(video_path), fourcc, 5.0, (32, 32))
    assert writer.isOpened()

    frame_one = np.full((32, 32, 3), (0, 0, 255), dtype=np.uint8)
    frame_two = np.full((32, 32, 3), (0, 255, 0), dtype=np.uint8)
    writer.write(frame_one)
    writer.write(frame_two)
    writer.release()


def test_numeric_string_sources_are_normalized():
    assert CameraStream._normalize_source("0") == 0
    assert CameraStream._normalize_source("rtsp://example") == "rtsp://example"


def test_file_stream_returns_first_frame_without_skipping(tmp_path):
    video_path = tmp_path / "sample.avi"
    _write_test_video(video_path)

    stream = CameraStream(str(video_path))
    try:
        success, frame = stream.read()
        assert success is True
        assert frame is not None
        assert frame[0, 0, 2] > 200
        assert frame[0, 0, 1] < 50
    finally:
        stream.release()
