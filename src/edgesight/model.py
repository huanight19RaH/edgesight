"""Baseline inference model interfaces."""

import time
from typing import Any, Dict, List, Tuple
import cv2
import numpy as np


class BaselineDetector:
    """Lightweight baseline detector interface with latency instrumentation."""

    def __init__(self, confidence_threshold: float = 0.5):
        self.conf_thresh = confidence_threshold

    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        """Preprocesses frame for model input."""
        # Simple color normalization / resize baseline
        resized = cv2.resize(frame, (224, 224))
        normalized = resized.astype(np.float32) / 255.0
        return normalized

    def infer(self, preprocessed_input: np.ndarray) -> Dict[str, Any]:
        """Runs mock/lightweight inference pass."""
        # Baseline mock inference returning dummy bounding boxes / gesture scores
        # Will be connected to torchvision MobileNet or YOLOv8n
        return {
            "boxes": [],
            "scores": [],
            "labels": [],
            "status": "ready"
        }

    def predict(self, frame: np.ndarray) -> Tuple[Dict[str, Any], Dict[str, float]]:
        """Runs full pipeline with microsecond-level telemetry."""
        t0 = time.perf_counter()
        inp = self.preprocess(frame)
        t1 = time.perf_counter()
        results = self.infer(inp)
        t2 = time.perf_counter()

        telemetry = {
            "preprocess_ms": (t1 - t0) * 1000.0,
            "inference_ms": (t2 - t1) * 1000.0,
            "total_ms": (t2 - t0) * 1000.0,
        }
        return results, telemetry
