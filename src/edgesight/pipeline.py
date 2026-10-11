"""CLI entrypoint and end-to-end pipeline runner."""

import argparse
import sys
import cv2
from edgesight.capture import VideoSource
from edgesight.model import BaselineDetector


def main():
    parser = argparse.ArgumentParser(description="EdgeSight on-device inference runner")
    parser.add_argument("--source", default="0", help="Webcam index (e.g. 0) or video file path")
    parser.add_argument("--benchmark", action="store_true", help="Run in benchmark mode without GUI window")
    parser.add_argument("--max-frames", type=int, default=100, help="Max frames for benchmark mode")
    args = parser.parse_args()

    cap = VideoSource(source=args.source)
    detector = BaselineDetector()

    print(f"[EdgeSight] Initializing capture on source: {args.source}")
    if not cap.open():
        print(f"[Error] Could not open video source: {args.source}", file=sys.stderr)
        sys.exit(1)

    print("[EdgeSight] Pipeline ready. Press 'q' to quit.")
    frame_count = 0
    latencies = []

    try:
        for frame, fps in cap.frames():
            frame_count += 1
            results, telemetry = detector.predict(frame)
            latencies.append(telemetry["total_ms"])

            if not args.benchmark:
                cv2.putText(
                    frame,
                    f"FPS: {fps:.1f} | Latency: {telemetry['total_ms']:.1f}ms",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )
                cv2.imshow("EdgeSight Stream", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                if frame_count >= args.max_frames:
                    break

        if latencies:
            avg_lat = sum(latencies) / len(latencies)
            print(f"[Benchmark Summary] Processed {len(latencies)} frames | Avg latency: {avg_lat:.2f}ms")

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
