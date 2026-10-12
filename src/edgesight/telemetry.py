"""Telemetry tracking and statistical profiling for on-device edge benchmarking."""

import json
import os
import platform
import time
from typing import Any, Dict, List
import numpy as np


class TelemetryRecorder:
    """Collects per-frame metrics and computes statistical percentiles (P50, P95, P99)."""

    def __init__(self, model_name: str = "Unknown", device: str = "CPU"):
        self.model_name = model_name
        self.device = device
        self.start_time = time.time()
        self.fps_records: List[float] = []
        self.prep_latencies: List[float] = []
        self.inf_latencies: List[float] = []
        self.post_latencies: List[float] = []
        self.total_latencies: List[float] = []

    def record_frame(self, instant_fps: float, stage_telemetry: Dict[str, float]):
        """Records one frame of telemetry measurements."""
        if instant_fps > 0:
            self.fps_records.append(instant_fps)
        self.prep_latencies.append(stage_telemetry.get("preprocess_ms", 0.0))
        self.inf_latencies.append(stage_telemetry.get("inference_ms", 0.0))
        self.post_latencies.append(stage_telemetry.get("postprocess_ms", 0.0))
        self.total_latencies.append(stage_telemetry.get("total_ms", 0.0))

    def _calc_stats(self, values: List[float]) -> Dict[str, float]:
        """Calculates mean, min, max, P50, P90, P95, P99 for a metric series."""
        if not values:
            return {
                "mean": 0.0,
                "min": 0.0,
                "max": 0.0,
                "p50": 0.0,
                "p90": 0.0,
                "p95": 0.0,
                "p99": 0.0,
            }
        arr = np.array(values, dtype=np.float32)
        return {
            "mean": float(np.mean(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "p50": float(np.percentile(arr, 50)),
            "p90": float(np.percentile(arr, 90)),
            "p95": float(np.percentile(arr, 95)),
            "p99": float(np.percentile(arr, 99)),
        }

    def get_summary(self) -> Dict[str, Any]:
        """Returns structured benchmark summary."""
        duration = time.time() - self.start_time
        frame_count = len(self.total_latencies)

        return {
            "environment": {
                "os": platform.system(),
                "release": platform.release(),
                "processor": platform.processor(),
                "python": platform.python_version(),
                "model": self.model_name,
                "device": self.device,
            },
            "execution": {
                "total_frames": frame_count,
                "elapsed_seconds": round(duration, 2),
                "effective_fps": round(frame_count / duration, 2) if duration > 0 else 0.0,
            },
            "fps": self._calc_stats(self.fps_records),
            "latency_ms": {
                "total": self._calc_stats(self.total_latencies),
                "inference": self._calc_stats(self.inf_latencies),
                "preprocess": self._calc_stats(self.prep_latencies),
                "postprocess": self._calc_stats(self.post_latencies),
            },
        }

    def save_json(self, filepath: str):
        """Exports summary to JSON file."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        summary = self.get_summary()
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

    def print_summary(self):
        """Prints formatted console table of benchmarking results."""
        summary = self.get_summary()
        exec_info = summary["execution"]
        lat = summary["latency_ms"]["total"]
        inf = summary["latency_ms"]["inference"]

        print("\n" + "=" * 65)
        print(f"  EdgeSight Telemetry Report — Model: {self.model_name} [{self.device}]")
        print("=" * 65)
        print(f"  Processed Frames : {exec_info['total_frames']} frames")
        print(f"  Elapsed Time     : {exec_info['elapsed_seconds']:.2f} s")
        print(f"  Effective FPS    : {exec_info['effective_fps']:.1f} FPS")
        print("-" * 65)
        print("  LATENCY METRICS (ms)")
        print(f"  Stage        Mean      P50 (Med) P90       P95       P99       Max")
        print(
            f"  Total        {lat['mean']:<9.2f}{lat['p50']:<10.2f}{lat['p90']:<10.2f}"
            f"{lat['p95']:<10.2f}{lat['p99']:<10.2f}{lat['max']:.2f}"
        )
        print(
            f"  Inference    {inf['mean']:<9.2f}{inf['p50']:<10.2f}{inf['p90']:<10.2f}"
            f"{inf['p95']:<10.2f}{inf['p99']:<10.2f}{inf['max']:.2f}"
        )
        print("=" * 65 + "\n")
