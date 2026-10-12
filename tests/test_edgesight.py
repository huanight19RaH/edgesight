"""Automated unit and integration tests for EdgeSight."""

import os
import sys
import unittest
import numpy as np

# Ensure src is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from edgesight.capture import VideoSource
from edgesight.model import BaselineDetector, YOLOEdgeDetector, load_detector
from edgesight.overlay import draw_detections, draw_telemetry_hud
from edgesight.telemetry import TelemetryRecorder


class TestEdgeSight(unittest.TestCase):
    """Test suite for EdgeSight on-device vision modules."""

    def test_synthetic_video_source(self):
        """Verifies synthetic video source generation and context manager."""
        with VideoSource(source="synthetic", width=320, height=240, synthetic_fps=60.0) as cap:
            props = cap.get_properties()
            self.assertEqual(props["source"], "synthetic")
            self.assertEqual(props["width"], 320)
            self.assertEqual(props["height"], 240)

            count = 0
            for frame, fps in cap.frames():
                self.assertEqual(frame.shape, (240, 320, 3))
                self.assertGreaterEqual(fps, 0.0)
                count += 1
                if count >= 5:
                    break
            self.assertEqual(count, 5)

    def test_baseline_detector(self):
        """Verifies BaselineDetector output schema and telemetry."""
        detector = BaselineDetector(confidence_threshold=0.1)
        dummy_frame = np.ones((480, 640, 3), dtype=np.uint8) * 128

        results, telemetry = detector.predict(dummy_frame)
        self.assertIn("boxes", results)
        self.assertIn("scores", results)
        self.assertIn("total_ms", telemetry)
        self.assertIn("inference_ms", telemetry)
        self.assertGreater(telemetry["total_ms"], 0.0)

    def test_yolo_edge_detector(self):
        """Verifies YOLOEdgeDetector with weights/yolov8n.pt."""
        weights_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "weights", "yolov8n.pt")
        )
        if not os.path.exists(weights_path):
            self.skipTest("yolov8n.pt not present, skipping YOLO test")

        detector = YOLOEdgeDetector(model_path=weights_path, device="cpu", warmup=False)
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

        results, telemetry = detector.predict(dummy_frame)
        self.assertEqual(results["device"], "cpu")
        self.assertIn("boxes", results)
        self.assertIn("total_ms", telemetry)
        self.assertIn("inference_ms", telemetry)
        self.assertIn("preprocess_ms", telemetry)
        self.assertIn("postprocess_ms", telemetry)

    def test_telemetry_recorder(self):
        """Verifies TelemetryRecorder statistics and percentiles."""
        rec = TelemetryRecorder(model_name="TestModel", device="CPU")
        for i in range(1, 21):
            rec.record_frame(
                instant_fps=30.0,
                stage_telemetry={
                    "preprocess_ms": 1.0,
                    "inference_ms": float(i),
                    "postprocess_ms": 1.0,
                    "total_ms": float(i + 2),
                },
            )

        summary = rec.get_summary()
        self.assertEqual(summary["execution"]["total_frames"], 20)
        self.assertGreater(summary["latency_ms"]["total"]["p50"], 0.0)
        self.assertGreaterEqual(
            summary["latency_ms"]["total"]["p95"],
            summary["latency_ms"]["total"]["p50"],
        )

    def test_overlays(self):
        """Verifies visual overlay rendering without dimension mutation."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_results = {
            "boxes": [[50.0, 50.0, 200.0, 200.0]],
            "scores": [0.85],
            "class_ids": [0],
            "class_names": ["person"],
            "count": 1,
        }
        annotated = draw_detections(frame.copy(), mock_results)
        self.assertEqual(annotated.shape, (480, 640, 3))

        hud_frame = draw_telemetry_hud(
            frame=annotated,
            fps=28.5,
            telemetry={"total_ms": 25.0, "inference_ms": 20.0, "preprocess_ms": 2.0, "postprocess_ms": 3.0},
            model_name="YOLOv8n",
            device="CPU",
            detection_count=1,
            smoothed=True,
            notification="Testing Notification",
        )
        self.assertEqual(hud_frame.shape, (480, 640, 3))

    def test_compute_iou(self):
        """Verifies compute_iou accuracy for overlapping and disjoint boxes."""
        from edgesight.smoother import compute_iou
        box1 = [0.0, 0.0, 100.0, 100.0]
        box2 = [0.0, 0.0, 100.0, 100.0]
        self.assertAlmostEqual(compute_iou(box1, box2), 1.0)

        box3 = [50.0, 0.0, 150.0, 100.0]  # Overlap 50x100 = 5000 / 15000 = 0.3333
        self.assertAlmostEqual(compute_iou(box1, box3), 5000.0 / 15000.0, places=4)

        box_disjoint = [200.0, 200.0, 300.0, 300.0]
        self.assertEqual(compute_iou(box1, box_disjoint), 0.0)

    def test_temporal_smoother(self):
        """Verifies EMA anti-jitter smoothing on noisy box sequences."""
        from edgesight.smoother import TemporalBoxSmoother
        smoother = TemporalBoxSmoother(alpha=0.5, iou_threshold=0.3, max_age=2)

        # Base box with artificial jitter (+- 10 pixels)
        jitter_offsets = [0.0, 10.0, -10.0, 8.0, -8.0]
        smoothed_x1s = []

        for offset in jitter_offsets:
            raw_res = {
                "boxes": [[100.0 + offset, 100.0, 200.0 + offset, 200.0]],
                "scores": [0.8],
                "class_ids": [0],
                "class_names": ["person"],
                "count": 1,
            }
            smoothed = smoother.update(raw_res)
            self.assertEqual(smoothed["count"], 1)
            self.assertIn("track_ids", smoothed)
            smoothed_x1s.append(smoothed["boxes"][0][0])

        # Variance of smoothed x1 should be lower than raw jitter variance
        raw_x1s = [100.0 + off for off in jitter_offsets]
        self.assertLess(np.var(smoothed_x1s), np.var(raw_x1s))

    def test_temporal_coasting_and_purge(self):
        """Verifies coasting on momentary occlusion and purging after max_age."""
        from edgesight.smoother import TemporalBoxSmoother
        smoother = TemporalBoxSmoother(alpha=0.6, max_age=2)

        # Frame 1: object present
        res1 = {
            "boxes": [[50.0, 50.0, 150.0, 150.0]],
            "scores": [0.9],
            "class_ids": [0],
            "class_names": ["cup"],
        }
        out1 = smoother.update(res1)
        self.assertEqual(out1["count"], 1)

        # Frame 2: object temporarily dropped / occluded (empty raw detections)
        out2 = smoother.update({"boxes": [], "scores": [], "class_ids": [], "class_names": []})
        # Should coast (keep 1 tracked object alive with slightly reduced confidence)
        self.assertEqual(out2["count"], 1)
        self.assertLess(out2["scores"][0], 0.9)

        # Frame 3: still empty (age = 2 == max_age)
        out3 = smoother.update({"boxes": [], "scores": [], "class_ids": [], "class_names": []})
        self.assertEqual(out3["count"], 1)

        # Frame 4: still empty (age = 3 > max_age) -> track should now be purged
        out4 = smoother.update({"boxes": [], "scores": [], "class_ids": [], "class_names": []})
        self.assertEqual(out4["count"], 0)


if __name__ == "__main__":
    unittest.main()

