"""Video, webcam, and synthetic frame capture module with telemetry and async buffering."""

import sys
import threading
import time
from typing import Generator, Optional, Tuple, Union
import cv2
import numpy as np


class VideoSource:
    """Manages OpenCV VideoCapture and synthetic frame streams with error recovery and optional async threading."""

    def __init__(
        self,
        source: Union[int, str] = 0,
        width: int = 640,
        height: int = 480,
        synthetic_fps: float = 30.0,
        threaded: bool = False,
    ):
        self.raw_source = source
        self.width = width
        self.height = height
        self.synthetic_fps = synthetic_fps
        self.threaded = threaded
        self.cap: Optional[cv2.VideoCapture] = None
        self._is_synthetic = False
        self._last_time = time.perf_counter()
        self._frame_index = 0

        # Threading state
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._frame_lock = threading.Lock()
        self._latest_frame: Optional[np.ndarray] = None
        self._frame_ready_event = threading.Event()
        self._camera_fps: float = 0.0
        self._capture_count: int = 0

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
        """Opens video stream or initializes synthetic generator, launching capture thread if enabled."""
        if self._is_synthetic:
            self._last_time = time.perf_counter()
            self._frame_index = 0
            if self.threaded:
                self._start_thread()
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

        opened = bool(self.cap and self.cap.isOpened())
        if opened and self.threaded:
            self._start_thread()

        return opened

    def _start_thread(self):
        """Starts background worker thread for non-blocking frame ingestion."""
        self._stop_event.clear()
        self._frame_ready_event.clear()
        self._thread = threading.Thread(target=self._threaded_worker, daemon=True)
        self._thread.start()
        # Wait for initial frame to become ready
        self._frame_ready_event.wait(timeout=3.0)

    def _threaded_worker(self):
        """Continuous background thread fetching latest frame to eliminate buffer lag."""
        last_t = time.perf_counter()
        while not self._stop_event.is_set():
            if self._is_synthetic:
                frame = self._generate_synthetic_frame()
                self._frame_index += 1
            else:
                if not self.cap or not self.cap.isOpened():
                    break
                ret, frame = self.cap.read()
                if not ret or frame is None:
                    break
                self._frame_index += 1

            now = time.perf_counter()
            dt = now - last_t
            last_t = now
            inst_fps = 1.0 / dt if dt > 0 else 0.0
            if self._camera_fps > 0:
                self._camera_fps = 0.9 * self._camera_fps + 0.1 * inst_fps
            else:
                self._camera_fps = inst_fps

            with self._frame_lock:
                self._latest_frame = frame
                self._capture_count += 1

            self._frame_ready_event.set()

    def read_latest(self) -> Tuple[Optional[np.ndarray], float]:
        """Atomically fetches the most recent frame without waiting on sensor read."""
        if not self.threaded:
            # Fallback to direct read if not running in threaded mode
            if self._is_synthetic:
                frame = self._generate_synthetic_frame()
                self._frame_index += 1
                return frame, self.synthetic_fps
            if self.cap and self.cap.isOpened():
                ret, frame = self.cap.read()
                return (frame, 30.0) if ret else (None, 0.0)
            return None, 0.0

        with self._frame_lock:
            if self._latest_frame is None:
                return None, 0.0
            return self._latest_frame.copy(), self._camera_fps

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

        if self.threaded:
            while not self._stop_event.is_set():
                if self._frame_ready_event.wait(timeout=1.0):
                    with self._frame_lock:
                        if self._latest_frame is None:
                            continue
                        frame = self._latest_frame.copy()

                    now = time.perf_counter()
                    dt = now - self._last_time
                    self._last_time = now
                    fps = 1.0 / dt if dt > 0 else 0.0
                    yield frame, fps
                else:
                    # Timeout reached
                    break
        else:
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
                "threaded": self.threaded,
                "camera_fps": self._camera_fps,
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
                "threaded": self.threaded,
                "camera_fps": self._camera_fps,
            }

        return {
            "source": self.source,
            "width": self.width,
            "height": self.height,
            "fps": 0.0,
            "threaded": self.threaded,
            "camera_fps": 0.0,
        }

    def release(self):
        """Releases worker thread and underlying video hardware."""
        if self.threaded:
            self._stop_event.set()
            if self._thread and self._thread.is_alive():
                self._thread.join(timeout=1.0)
                self._thread = None

        if self.cap and self.cap.isOpened():
            self.cap.release()
            self.cap = None

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()


class ThreadedVideoSource(VideoSource):
    """Convenience subclass enforcing async background thread frame ingestion."""

    def __init__(
        self,
        source: Union[int, str] = 0,
        width: int = 640,
        height: int = 480,
        synthetic_fps: float = 30.0,
    ):
        super().__init__(
            source=source,
            width=width,
            height=height,
            synthetic_fps=synthetic_fps,
            threaded=True,
        )
