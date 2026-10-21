"""Model inference wrappers for edge vision detection and telemetry."""

from abc import ABC, abstractmethod
import os
import time
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np


class BaseEdgeDetector(ABC):
    """Abstract interface for edge vision detectors."""

    @abstractmethod
    def predict(self, frame: np.ndarray) -> Tuple[Dict[str, Any], Dict[str, float]]:
        """Runs end-to-end detection and returns (results, telemetry)."""
        pass


class YOLOEdgeDetector(BaseEdgeDetector):
    """Ultralytics YOLO PyTorch detector optimized for real-time edge CPU inference."""

    def __init__(
        self,
        model_path: str = "weights/yolov8n.pt",
        conf_thresh: float = 0.35,
        device: str = "cpu",
        warmup: bool = True,
    ):
        self.model_path = model_path
        self.conf_thresh = conf_thresh
        self.device = device

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model weights not found at: {model_path}. "
                "Ensure weights are downloaded into weights/yolov8n.pt"
            )

        from ultralytics import YOLO

        self.model = YOLO(model_path)
        self.model_name = os.path.basename(model_path)

        # Pre-warm model to eliminate first-frame compilation latency
        if warmup:
            self._warmup()

    def _warmup(self):
        """Runs a 1-frame dummy pass on CPU to warm PyTorch caches."""
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        _ = self.model.predict(
            dummy_frame,
            conf=self.conf_thresh,
            device=self.device,
            verbose=False,
            imgsz=640,
        )

    def predict(self, frame: np.ndarray) -> Tuple[Dict[str, Any], Dict[str, float]]:
        """Runs inference and records per-stage telemetry in milliseconds."""
        t0 = time.perf_counter()

        # Preprocess stage (Ultralytics handles interior preprocessing, we track caller prep)
        t_prep_start = time.perf_counter()
        orig_shape = frame.shape
        t_prep_end = time.perf_counter()

        # Inference stage
        t_inf_start = time.perf_counter()
        outputs = self.model.predict(
            frame,
            conf=self.conf_thresh,
            device=self.device,
            verbose=False,
            imgsz=640,
        )
        t_inf_end = time.perf_counter()

        # Postprocess stage: extract structured detection list
        t_post_start = time.perf_counter()
        boxes: List[List[float]] = []
        scores: List[float] = []
        class_ids: List[int] = []
        class_names: List[str] = []

        if outputs and len(outputs) > 0:
            res = outputs[0]
            if res.boxes is not None and len(res.boxes) > 0:
                xyxy = res.boxes.xyxy.cpu().numpy()
                confs = res.boxes.conf.cpu().numpy()
                cls_indices = res.boxes.cls.cpu().numpy().astype(int)
                names_dict = res.names

                for box, conf, cls_idx in zip(xyxy, confs, cls_indices):
                    boxes.append([float(c) for c in box])
                    scores.append(float(conf))
                    class_ids.append(int(cls_idx))
                    class_names.append(str(names_dict.get(cls_idx, f"cls_{cls_idx}")))

        t_post_end = time.perf_counter()
        t_total = time.perf_counter()

        # Ultralytics internal speed dict if available
        internal_speed = outputs[0].speed if outputs else {}
        prep_ms = internal_speed.get("preprocess", (t_prep_end - t_prep_start) * 1000.0)
        inf_ms = internal_speed.get("inference", (t_inf_end - t_inf_start) * 1000.0)
        post_ms = internal_speed.get("postprocess", (t_post_end - t_post_start) * 1000.0)
        total_ms = (t_total - t0) * 1000.0

        results = {
            "boxes": boxes,
            "scores": scores,
            "class_ids": class_ids,
            "class_names": class_names,
            "count": len(boxes),
            "orig_shape": orig_shape,
            "model": self.model_name,
            "device": self.device,
            "backend": "pytorch",
        }

        telemetry = {
            "preprocess_ms": float(prep_ms),
            "inference_ms": float(inf_ms),
            "postprocess_ms": float(post_ms),
            "total_ms": float(total_ms),
        }

        return results, telemetry


class ONNXEdgeDetector(BaseEdgeDetector):
    """ONNX Runtime accelerated detector for high-throughput, low-latency edge inference."""

    def __init__(
        self,
        model_path: str = "weights/yolov8n.onnx",
        conf_thresh: float = 0.35,
        device: str = "cpu",
        warmup: bool = True,
    ):
        self.model_path = model_path
        self.conf_thresh = conf_thresh
        self.device = device

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"ONNX model weights not found at: {model_path}. "
                "Ensure weights exist at weights/yolov8n.onnx"
            )

        from ultralytics import YOLO

        self.model = YOLO(model_path, task="detect")
        self.model_name = os.path.basename(model_path)

        # Pre-warm ONNX session to avoid first-frame compilation jitter
        if warmup:
            self._warmup()

    def _warmup(self):
        """Runs a 1-frame dummy pass to warm ONNX Runtime session and memory buffers."""
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        _ = self.model.predict(
            dummy_frame,
            conf=self.conf_thresh,
            device=self.device,
            verbose=False,
            imgsz=640,
        )

    def predict(self, frame: np.ndarray) -> Tuple[Dict[str, Any], Dict[str, float]]:
        """Runs ONNX Runtime inference and records per-stage telemetry in milliseconds."""
        t0 = time.perf_counter()

        t_prep_start = time.perf_counter()
        orig_shape = frame.shape
        t_prep_end = time.perf_counter()

        t_inf_start = time.perf_counter()
        outputs = self.model.predict(
            frame,
            conf=self.conf_thresh,
            device=self.device,
            verbose=False,
            imgsz=640,
        )
        t_inf_end = time.perf_counter()

        t_post_start = time.perf_counter()
        boxes: List[List[float]] = []
        scores: List[float] = []
        class_ids: List[int] = []
        class_names: List[str] = []

        if outputs and len(outputs) > 0:
            res = outputs[0]
            if res.boxes is not None and len(res.boxes) > 0:
                xyxy = res.boxes.xyxy.cpu().numpy()
                confs = res.boxes.conf.cpu().numpy()
                cls_indices = res.boxes.cls.cpu().numpy().astype(int)
                names_dict = res.names

                for box, conf, cls_idx in zip(xyxy, confs, cls_indices):
                    boxes.append([float(c) for c in box])
                    scores.append(float(conf))
                    class_ids.append(int(cls_idx))
                    class_names.append(str(names_dict.get(cls_idx, f"cls_{cls_idx}")))

        t_post_end = time.perf_counter()
        t_total = time.perf_counter()

        internal_speed = outputs[0].speed if outputs else {}
        prep_ms = internal_speed.get("preprocess", (t_prep_end - t_prep_start) * 1000.0)
        inf_ms = internal_speed.get("inference", (t_inf_end - t_inf_start) * 1000.0)
        post_ms = internal_speed.get("postprocess", (t_post_end - t_post_start) * 1000.0)
        total_ms = (t_total - t0) * 1000.0

        results = {
            "boxes": boxes,
            "scores": scores,
            "class_ids": class_ids,
            "class_names": class_names,
            "count": len(boxes),
            "orig_shape": orig_shape,
            "model": self.model_name,
            "device": self.device,
            "backend": "onnx",
        }

        telemetry = {
            "preprocess_ms": float(prep_ms),
            "inference_ms": float(inf_ms),
            "postprocess_ms": float(post_ms),
            "total_ms": float(total_ms),
        }

        return results, telemetry


class BaselineDetector(BaseEdgeDetector):
    """Lightweight baseline detector (Canny edge & contour filter) for mock testing."""

    def __init__(self, confidence_threshold: float = 0.5):
        self.conf_thresh = confidence_threshold
        self.model_name = "BaselineCanny"

    def predict(self, frame: np.ndarray) -> Tuple[Dict[str, Any], Dict[str, float]]:
        t0 = time.perf_counter()

        t_prep_start = time.perf_counter()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        t_prep_end = time.perf_counter()

        t_inf_start = time.perf_counter()
        edges = cv2.Canny(blurred, 50, 150)
        t_inf_end = time.perf_counter()

        t_post_start = time.perf_counter()
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        scores = []
        class_ids = []
        class_names = []

        h, w = frame.shape[:2]
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 1000:
                bx, by, bw, bh = cv2.boundingRect(cnt)
                score = min(1.0, area / (w * h * 0.25))
                if score >= self.conf_thresh:
                    boxes.append([float(bx), float(by), float(bx + bw), float(by + bh)])
                    scores.append(float(score))
                    class_ids.append(0)
                    class_names.append("salient_object")
        t_post_end = time.perf_counter()
        t_total = time.perf_counter()

        results = {
            "boxes": boxes,
            "scores": scores,
            "class_ids": class_ids,
            "class_names": class_names,
            "count": len(boxes),
            "orig_shape": frame.shape,
            "model": self.model_name,
            "device": "cpu",
            "backend": "baseline",
        }

        telemetry = {
            "preprocess_ms": (t_prep_end - t_prep_start) * 1000.0,
            "inference_ms": (t_inf_end - t_inf_start) * 1000.0,
            "postprocess_ms": (t_post_end - t_post_start) * 1000.0,
            "total_ms": (t_total - t0) * 1000.0,
        }

        return results, telemetry


def resolve_weights_path(weights_path: str) -> str:
    """Finds weights path relative to CWD or edgesight project root."""
    if os.path.exists(weights_path):
        return weights_path

    # Try relative to edgesight project root (src/edgesight/../../weights/...)
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    candidate_1 = os.path.join(project_root, weights_path)
    if os.path.exists(candidate_1):
        return candidate_1

    candidate_2 = os.path.join(project_root, "weights", os.path.basename(weights_path))
    if os.path.exists(candidate_2):
        return candidate_2

    return weights_path


def load_detector(
    model_type: str = "auto",
    weights_path: Optional[str] = None,
    conf_thresh: float = 0.35,
    device: str = "cpu",
    backend: str = "auto",
) -> BaseEdgeDetector:
    """Factory creating appropriate detector based on availability and request.

    Args:
        model_type: 'auto', 'yolo', 'onnx', or 'baseline'
        weights_path: Optional path to weights file
        conf_thresh: Confidence threshold for predictions
        device: 'cpu' or device identifier
        backend: 'auto', 'onnx', or 'pytorch'
    """
    if model_type.lower() == "baseline":
        return BaselineDetector(confidence_threshold=conf_thresh)

    # Determine resolved weights path
    if weights_path is not None:
        resolved = resolve_weights_path(weights_path)
        if (backend == "onnx" or model_type.lower() == "onnx") and resolved.endswith(".pt"):
            onnx_equivalent = resolved[:-3] + ".onnx"
            if os.path.exists(onnx_equivalent):
                resolved = onnx_equivalent
    else:
        # Check auto discovery: prefer onnx for speed, fallback to pt
        onnx_cand = resolve_weights_path("weights/yolov8n.onnx")
        pt_cand = resolve_weights_path("weights/yolov8n.pt")
        if backend in ("auto", "onnx") and os.path.exists(onnx_cand):
            resolved = onnx_cand
        elif os.path.exists(pt_cand):
            resolved = pt_cand
        else:
            resolved = "weights/yolov8n.onnx"

    # If ONNX requested or resolved is an onnx file
    if (backend == "onnx" or resolved.endswith(".onnx") or model_type.lower() == "onnx") and os.path.exists(resolved):
        return ONNXEdgeDetector(
            model_path=resolved,
            conf_thresh=conf_thresh,
            device=device,
        )

    # If PyTorch YOLO requested or resolved is a pt file
    if (backend in ("auto", "pytorch") or model_type.lower() == "yolo") and os.path.exists(resolved):
        if resolved.endswith(".onnx"):
            return ONNXEdgeDetector(
                model_path=resolved,
                conf_thresh=conf_thresh,
                device=device,
            )
        return YOLOEdgeDetector(
            model_path=resolved,
            conf_thresh=conf_thresh,
            device=device,
        )

    return BaselineDetector(confidence_threshold=conf_thresh)
