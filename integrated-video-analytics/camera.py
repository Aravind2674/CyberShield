import cv2
import threading

class CameraStream:
    """Threaded RTSP/camera capture to prevent frame drops."""
    def __init__(self, source):
        self.source = source
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video source {source}")
            
        self.ret, self.frame = self.cap.read()
        self.running = True
        
        # Start daemon thread strictly for reading frames
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _update(self):
        while self.running:
            # We don't process here, we just continually read to flush the buffer
            # and prevent the stream from backing up (latency issue in RTSP)
            cap_ret, cap_frame = self.cap.read()
            if cap_ret:
                self.ret = cap_ret
                self.frame = cap_frame
            else:
                self.running = False

    def read(self):
        # Always return the freshest frame
        return self.ret, self.frame

    def release(self):
        self.running = False
        self.thread.join(timeout=1.0)
        self.cap.release()
