"""Visual overlay utilities: bounding boxes, labels, and telemetry HUD."""

from typing import Any, Dict, List, Optional
import cv2
import numpy as np


# Palette for distinct class boxes
PALETTE = [
    (0, 255, 128),    # Mint green
    (255, 128, 0),    # Blue
    (0, 165, 255),    # Orange
    (255, 0, 128),    # Magenta
    (0, 215, 255),    # Yellow
    (200, 200, 0),    # Cyan
    (128, 0, 255),    # Rose
    (255, 255, 0),    # Aqua
]


def get_color(class_id: int) -> tuple:
    """Returns consistent color for class ID."""
    return PALETTE[class_id % len(PALETTE)]


def draw_detections(
    frame: np.ndarray,
    results: Dict[str, Any],
    box_thickness: int = 2,
) -> np.ndarray:
    """Draws bounding boxes, tracking IDs, and confidence labels on frame."""
    boxes: List[List[float]] = results.get("boxes", [])
    scores: List[float] = results.get("scores", [])
    class_ids: List[int] = results.get("class_ids", [])
    class_names: List[str] = results.get("class_names", [])
    track_ids: List[int] = results.get("track_ids", [])

    for i, (box, score, cls_id, cls_name) in enumerate(zip(boxes, scores, class_ids, class_names)):
        x1, y1, x2, y2 = [int(round(v)) for v in box]
        color = get_color(cls_id)

        # Draw bounding rectangle
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, box_thickness)

        # Build label text (with track ID if available)
        if track_ids and i < len(track_ids):
            label_text = f"#{track_ids[i]} {cls_name} {score * 100:.0f}%"
        else:
            label_text = f"{cls_name} {score * 100:.0f}%"

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thick = 1

        (tw, th), baseline = cv2.getTextSize(label_text, font, font_scale, font_thick)
        label_y1 = max(0, y1 - th - baseline - 4)
        label_y2 = y1
        label_x2 = min(frame.shape[1], x1 + tw + 6)

        # Background rectangle for text
        cv2.rectangle(frame, (x1, label_y1), (label_x2, label_y2), (20, 20, 20), -1)
        cv2.rectangle(frame, (x1, label_y1), (label_x2, label_y2), color, 1)

        # Text
        cv2.putText(
            frame,
            label_text,
            (x1 + 3, label_y2 - baseline - 1),
            font,
            font_scale,
            (255, 255, 255),
            font_thick,
            cv2.LINE_AA,
        )

    return frame


def draw_telemetry_hud(
    frame: np.ndarray,
    fps: float,
    telemetry: Dict[str, float],
    model_name: str = "YOLOv8n",
    device: str = "CPU",
    detection_count: int = 0,
    smoothed: bool = True,
    notification: Optional[str] = None,
) -> np.ndarray:
    """Renders semi-transparent HUD banner with FPS and per-stage latency."""
    h, w = frame.shape[:2]
    hud_h = 60
    hud_w = min(w - 20, 620)

    # Semi-transparent overlay box
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (10 + hud_w, 10 + hud_h), (15, 15, 20), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    # FPS color code
    if fps >= 25.0:
        fps_color = (50, 255, 50)  # Green
    elif fps >= 15.0:
        fps_color = (0, 215, 255)  # Amber
    else:
        fps_color = (50, 50, 255)  # Red

    # Telemetry line 1: FPS and total latency
    total_ms = telemetry.get("total_ms", 0.0)
    inf_ms = telemetry.get("inference_ms", 0.0)
    prep_ms = telemetry.get("preprocess_ms", 0.0)
    post_ms = telemetry.get("postprocess_ms", 0.0)

    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (20, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        fps_color,
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        frame,
        f"| Latency: {total_ms:.1f}ms (Prep: {prep_ms:.1f} | Inf: {inf_ms:.1f} | Post: {post_ms:.1f})",
        (120, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )

    # Telemetry line 2: Model, device, smoother status, and object count
    smooth_str = "EMA ON" if smoothed else "OFF"
    line2 = f"Model: {model_name} [{device.upper()}] | Smooth: {smooth_str} | Detected: {detection_count} objs | Res: {w}x{h}"
    cv2.putText(
        frame,
        line2,
        (20, 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (180, 180, 180),
        1,
        cv2.LINE_AA,
    )

    # Optional temporary notification (e.g. screenshot saved)
    if notification:
        (nw, nh), nb = cv2.getTextSize(notification, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        notif_y = h - 25
        cv2.rectangle(frame, (10, notif_y - nh - 6), (20 + nw, notif_y + 4), (0, 100, 0), -1)
        cv2.putText(
            frame,
            notification,
            (15, notif_y - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    return frame
