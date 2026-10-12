"""Video, webcam, and synthetic frame capture module with telemetry."""

import sys
import time
from typing import Generator, Optional, Tuple, Union
import cv2
import numpy as np


class VideoSource:
    """Manages OpenCV VideoCapture and synthetic frame streams with error recovery."""

    def __init__(
        self,
        source: Union[int, str] = 0,
        width: int = 640,
        height: int = 480,
        synthetic_fps: float = 30.0,
    ):
        self.raw_source = source
        self.width = width
        self.height = height
        self.synthetic_fps = synthetic_fps
        self.cap: Optional[cv2.VideoCapture] = None
        self._is_synthetic = False
        self._last_time = time.perf_counter()
        self._frame_index = 0

        # Detect source type
        src_str = str(source).strip().lower()
        if src_str in ("synthetic", "mock", "dummy"):
            self._is_synthetic = True
            self.source = "synthetic"
        elif str(source).isdigit():
            self.source = int(source)
        else:
            self.source = str(source)

    def open(self) -> bool:
        """Opens video stream or initializes synthetic generator."""
        if self._is_synthetic:
            self._last_time = time.perf_counter()
            self._frame_index = 0
            return True

        if isinstance(self.source, int):
            # Prefer DirectShow backend on Windows for faster webcam initialization
            if sys.platform.startswith("win"):
                self.cap = cv2.VideoCapture(self.source, cv2.CAP_DSHOW)
                if not self.cap or not self.cap.isOpened():
                    self.cap = cv2.VideoCapture(self.source)
            else:
                self.cap = cv2.VideoCapture(self.source)

            if self.cap and self.cap.isOpened():
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        else:
            self.cap = cv2.VideoCapture(self.source)

        return bool(self.cap and self.cap.isOpened())

    def _generate_synthetic_frame(self) -> np.ndarray:
        """Generates dynamic synthetic frame for testing without hardware camera."""
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        # Background gradient
        t = self._frame_index * 0.05
        grid_x = int((np.sin(t) + 1.0) * 0.5 * (self.width - 120))
        grid_y = int((np.cos(t) + 1.0) * 0.5 * (self.height - 120))

        # Simulated test object (bounding target)
        cv2.rectangle(
            frame,
            (grid_x, grid_y),
            (grid_x + 120, grid_y + 120),
            (0, 200, 255),
            -1,
        )
        cv2.putText(
            frame,
            f"SYNTHETIC STREAM #{self._frame_index}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        # Artificial sleep to match synthetic FPS
        delay = 1.0 / self.synthetic_fps
        time.sleep(max(0.001, delay * 0.1))
        return frame

    def frames(self) -> Generator[Tuple[np.ndarray, float], None, None]:
        """Yields (frame, instant_fps) until stream ends."""
        if not self._is_synthetic:
            if not self.cap or not self.cap.isOpened():
                if not self.open():
                    raise RuntimeError(f"Failed to open video source: {self.source}")

        while True:
            if self._is_synthetic:
                frame = self._generate_synthetic_frame()
                self._frame_index += 1
            else:
                ret, frame = self.cap.read()
                if not ret or frame is None:
                    break
                self._frame_index += 1

            now = time.perf_counter()
            dt = now - self._last_time
            self._last_time = now
            fps = 1.0 / dt if dt > 0 else 0.0

            yield frame, fps

    def get_properties(self) -> dict:
        """Returns stream metadata."""
        if self._is_synthetic:
            return {
                "source": "synthetic",
                "width": self.width,
                "height": self.height,
                "fps": self.synthetic_fps,
            }

        if self.cap and self.cap.isOpened():
            actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = float(self.cap.get(cv2.CAP_PROP_FPS))
            return {
                "source": self.source,
                "width": actual_w or self.width,
                "height": actual_h or self.height,
                "fps": actual_fps or 30.0,
            }

        return {"source": self.source, "width": self.width, "height": self.height, "fps": 0.0}

    def release(self):
        """Releases video resource."""
        if self.cap and self.cap.isOpened():
            self.cap.release()
            self.cap = None

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
