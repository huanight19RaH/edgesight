"""Video and webcam frame capture module with telemetry."""

import time
from typing import Generator, Optional, Tuple, Union
import cv2
import numpy as np


class VideoSource:
    """Manages OpenCV VideoCapture with robust FPS and error recovery."""

    def __init__(self, source: Union[int, str] = 0, width: int = 640, height: int = 480):
        self.source = int(source) if str(source).isdigit() else str(source)
        self.width = width
        self.height = height
        self.cap: Optional[cv2.VideoCapture] = None
        self._fps_history = []
        self._last_time = time.perf_counter()

    def open(self) -> bool:
        """Opens video stream and configures resolution."""
        self.cap = cv2.VideoCapture(self.source)
        if isinstance(self.source, int):
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        return self.cap.isOpened()

    def frames(self) -> Generator[Tuple[np.ndarray, float], None, None]:
        """Yields (frame, instant_fps) until stream ends."""
        if not self.cap or not self.cap.isOpened():
            if not self.open():
                raise RuntimeError(f"Failed to open video source: {self.source}")

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            now = time.perf_counter()
            dt = now - self._last_time
            self._last_time = now
            fps = 1.0 / dt if dt > 0 else 0.0

            yield frame, fps

    def release(self):
        """Releases the camera device."""
        if self.cap and self.cap.isOpened():
            self.cap.release()
