"""CLI entrypoint and end-to-end edge vision pipeline runner."""

import argparse
import json
import os
import sys
import time
import cv2
import numpy as np

from edgesight.capture import VideoSource
from edgesight.model import load_detector
from edgesight.overlay import draw_detections, draw_telemetry_hud
from edgesight.smoother import TemporalBoxSmoother
from edgesight.telemetry import TelemetryRecorder


def main():
    parser = argparse.ArgumentParser(
        description="EdgeSight on-device inference runner with live webcam and telemetry"
    )
    parser.add_argument(
        "--source",
        default="0",
        help="Webcam index (e.g. 0), video file path, or 'synthetic'",
    )
    parser.add_argument(
        "--model",
        default="yolo",
        help="Model type ('yolo' or 'baseline') or path to .pt weights",
    )
    parser.add_argument(
        "--weights",
        default="weights/yolov8n.pt",
        help="Path to model weights file",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.35,
        help="Confidence detection threshold (0.0 - 1.0)",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Compute device ('cpu' recommended for edge profiling)",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=640,
        help="Capture frame width",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="Capture frame height",
    )
    parser.add_argument(
        "--no-smooth",
        action="store_true",
        help="Disable temporal bounding box smoothing",
    )
    parser.add_argument(
        "--smooth-alpha",
        type=float,
        default=0.65,
        help="EMA smoothing alpha weight (0.1 to 0.95)",
    )
    parser.add_argument(
        "--smooth-iou",
        type=float,
        default=0.35,
        help="IoU tracking threshold for association",
    )
    parser.add_argument(
        "--max-age",
        type=int,
        default=2,
        help="Max frames to coast an occluded track",
    )
    parser.add_argument(
        "--backend",
        choices=["auto", "onnx", "pytorch"],
        default="auto",
        help="Inference runtime backend: 'onnx' (high-performance) or 'pytorch'",
    )
    parser.add_argument(
        "--async-capture",
        action="store_true",
        help="Enable non-blocking asynchronous camera capture thread (zero stream lag)",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Headless mode without GUI window (for automated profiling)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=100,
        help="Max frames to process in benchmark mode",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path to export annotated video (.mp4)",
    )
    parser.add_argument(
        "--save-telemetry",
        default="raw/telemetry_latest.json",
        help="Path to save summary telemetry JSON report",
    )
    parser.add_argument(
        "--snapshot-dir",
        default="raw/snapshots",
        help="Directory to save inspection snapshots",
    )
    args = parser.parse_args()

    # Determine detector model & weights
    weights_path = args.weights
    if args.model.endswith(".pt") or args.model.endswith(".onnx"):
        weights_path = args.model
        model_type = "onnx" if args.model.endswith(".onnx") else "yolo"
    else:
        model_type = args.model

    print(
        f"[EdgeSight] Initializing video source: {args.source} ({args.width}x{args.height})"
        f"{' [Async Threaded]' if args.async_capture else ''}"
    )
    cap = VideoSource(
        source=args.source,
        width=args.width,
        height=args.height,
        threaded=args.async_capture,
    )
    if not cap.open():
        print(
            f"[Error] Failed to open video source: '{args.source}'. "
            "If using webcam, check if camera index is correct or try --source synthetic.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(
        f"[EdgeSight] Loading model '{model_type}' (backend={args.backend}) "
        f"on device '{args.device}' (conf={args.conf})..."
    )
    try:
        detector = load_detector(
            model_type=model_type,
            weights_path=weights_path,
            conf_thresh=args.conf,
            device=args.device,
            backend=args.backend,
        )
        model_name = getattr(detector, "model_name", "YOLOv8n")
    except Exception as e:
        print(f"[Warning] Failed to load {model_type} ({e}). Falling back to BaselineDetector.")
        detector = load_detector(model_type="baseline", conf_thresh=args.conf)
        model_name = getattr(detector, "model_name", "BaselineDetector")

    recorder = TelemetryRecorder(model_name=model_name, device=args.device)

    # Temporal smoother
    smoother = TemporalBoxSmoother(
        iou_threshold=args.smooth_iou,
        alpha=args.smooth_alpha,
        max_age=args.max_age,
    )
    enable_smooth = not args.no_smooth

    # Optional video writer
    writer = None
    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.output, fourcc, 30.0, (args.width, args.height))
        print(f"[EdgeSight] Recording output video to: {args.output}")

    # UI state
    show_hud = True
    show_boxes = True
    notification = None
    notification_until = 0.0
    frame_idx = 0

    os.makedirs(args.snapshot_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print("  EdgeSight Live Pipeline Running")
    if not args.benchmark:
        print("  Controls:")
        print("    'q' / ESC : Quit pipeline")
        print("    's'       : Capture snapshot & save detection record")
        print("    't'       : Toggle temporal smoothing (EMA)")
        print("    'c'       : Clear tracking history")
        print("    'h'       : Toggle telemetry HUD")
        print("    'b'       : Toggle bounding boxes")
    else:
        print(f"  Benchmark Mode: Processing {args.max_frames} frames headless...")
    print("=" * 60 + "\n")

    try:
        for frame, fps in cap.frames():
            frame_idx += 1

            # Inference & telemetry
            results, telemetry = detector.predict(frame)
            recorder.record_frame(fps, telemetry)

            # Apply temporal smoothing if enabled
            if enable_smooth:
                display_results = smoother.update(results)
            else:
                display_results = results

            # Annotation
            annotated = frame.copy()
            if show_boxes and display_results["count"] > 0:
                annotated = draw_detections(annotated, display_results)

            # Check expired notification
            if notification and time.time() > notification_until:
                notification = None

            if show_hud:
                annotated = draw_telemetry_hud(
                    frame=annotated,
                    fps=fps,
                    telemetry=telemetry,
                    model_name=model_name,
                    device=args.device,
                    detection_count=display_results["count"],
                    smoothed=enable_smooth,
                    notification=notification,
                )

            if writer is not None:
                writer.write(annotated)

            if not args.benchmark:
                cv2.imshow("EdgeSight Stream", annotated)
                key = cv2.waitKey(1) & 0xFF

                if key in (ord("q"), 27):  # 'q' or ESC
                    break
                elif key == ord("h"):
                    show_hud = not show_hud
                elif key == ord("b"):
                    show_boxes = not show_boxes
                elif key == ord("t"):
                    enable_smooth = not enable_smooth
                    if not enable_smooth:
                        smoother.reset()
                    notification = f"Temporal Smoothing: {'ON (EMA)' if enable_smooth else 'OFF'}"
                    notification_until = time.time() + 2.0
                elif key == ord("c"):
                    smoother.reset()
                    notification = "Tracking history cleared"
                    notification_until = time.time() + 1.5
                elif key == ord("s"):
                    # Save snapshot for failure mode / inspection analysis
                    ts = int(time.time())
                    shot_path = os.path.join(args.snapshot_dir, f"frame_{frame_idx}_{ts}.jpg")
                    cv2.imwrite(shot_path, annotated)

                    meta_path = os.path.join(args.snapshot_dir, "snapshots.jsonl")
                    with open(meta_path, "a", encoding="utf-8") as mf:
                        entry = {
                            "frame_idx": frame_idx,
                            "timestamp": ts,
                            "path": shot_path,
                            "results": display_results,
                            "telemetry": telemetry,
                        }
                        mf.write(json.dumps(entry) + "\n")

                    notification = f"Snapshot saved: {os.path.basename(shot_path)}"
                    notification_until = time.time() + 2.0
            else:
                if frame_idx >= args.max_frames:
                    break

    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if not args.benchmark:
            cv2.destroyAllWindows()

    # Telemetry report
    recorder.print_summary()
    if args.save_telemetry:
        recorder.save_json(args.save_telemetry)
        print(f"[EdgeSight] Summary telemetry saved to: {args.save_telemetry}")


if __name__ == "__main__":
    main()
