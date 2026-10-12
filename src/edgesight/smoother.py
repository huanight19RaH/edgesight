"""Temporal bounding box smoothing, tracking, and anti-jitter filter."""

from typing import Any, Dict, List, Optional
import numpy as np


def compute_iou(box_a: List[float], box_b: List[float]) -> float:
    """Computes Intersection over Union (IoU) between two [x1, y1, x2, y2] boxes."""
    xA = max(box_a[0], box_b[0])
    yA = max(box_a[1], box_b[1])
    xB = min(box_a[2], box_b[2])
    yB = min(box_a[3], box_b[3])

    inter_w = max(0.0, xB - xA)
    inter_h = max(0.0, yB - yA)
    inter_area = inter_w * inter_h

    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union_area = area_a + area_b - inter_area

    if union_area <= 0.0:
        return 0.0
    return inter_area / union_area


class TrackedBox:
    """Maintains continuous trajectory and smoothed coordinates for an object."""

    def __init__(
        self,
        track_id: int,
        box: List[float],
        score: float,
        class_id: int,
        class_name: str,
    ):
        self.track_id = track_id
        self.box = [float(c) for c in box]
        self.score = float(score)
        self.class_id = class_id
        self.class_name = class_name
        self.hits = 1
        self.time_since_update = 0  # age in frames without direct detection

    def update(self, new_box: List[float], new_score: float, alpha: float):
        """Applies adaptive Exponential Moving Average (EMA) to coordinates."""
        # Calculate coordinate delta to detect rapid motion
        dx = abs(new_box[0] - self.box[0]) + abs(new_box[2] - self.box[2])
        dy = abs(new_box[1] - self.box[1]) + abs(new_box[3] - self.box[3])
        shift = dx + dy

        # If motion is fast (>30px shift), increase responsiveness to prevent lag
        effective_alpha = min(0.95, alpha + 0.25) if shift > 30.0 else alpha

        self.box = [
            effective_alpha * n + (1.0 - effective_alpha) * p
            for n, p in zip(new_box, self.box)
        ]
        self.score = 0.7 * new_score + 0.3 * self.score
        self.hits += 1
        self.time_since_update = 0

    def coast(self):
        """Coasts existing prediction when temporarily occluded or missed."""
        self.time_since_update += 1
        self.score *= 0.88  # confidence decay


class TemporalBoxSmoother:
    """Anti-jitter and temporal hysteresis filter for real-time edge vision."""

    def __init__(
        self,
        iou_threshold: float = 0.35,
        alpha: float = 0.65,
        max_age: int = 2,
        min_hits: int = 1,
    ):
        self.iou_threshold = iou_threshold
        self.alpha = alpha
        self.max_age = max_age
        self.min_hits = min_hits
        self.tracks: List[TrackedBox] = []
        self._next_id = 1

    def reset(self):
        """Resets active tracks."""
        self.tracks.clear()
        self._next_id = 1

    def update(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Processes raw detection frame and returns smoothed, stabilized boxes."""
        raw_boxes: List[List[float]] = results.get("boxes", [])
        raw_scores: List[float] = results.get("scores", [])
        raw_class_ids: List[int] = results.get("class_ids", [])
        raw_class_names: List[str] = results.get("class_names", [])

        num_dets = len(raw_boxes)
        num_tracks = len(self.tracks)

        # Match detections to existing tracks via IoU matrix
        matched_dets = set()
        matched_tracks = set()

        if num_dets > 0 and num_tracks > 0:
            # Greedy bipartite matching preferring matching classes and highest IoU
            candidates = []
            for d_idx, (d_box, d_cls) in enumerate(zip(raw_boxes, raw_class_ids)):
                for t_idx, track in enumerate(self.tracks):
                    # Prefer same class
                    cls_bonus = 0.1 if track.class_id == d_cls else 0.0
                    iou = compute_iou(d_box, track.box) + cls_bonus
                    if iou >= self.iou_threshold:
                        candidates.append((iou, d_idx, t_idx))

            # Sort descending by IoU
            candidates.sort(key=lambda x: x[0], reverse=True)

            for _, d_idx, t_idx in candidates:
                if d_idx in matched_dets or t_idx in matched_tracks:
                    continue
                # Match found: update track
                self.tracks[t_idx].update(
                    raw_boxes[d_idx],
                    raw_scores[d_idx],
                    self.alpha,
                )
                matched_dets.add(d_idx)
                matched_tracks.add(t_idx)

        # Coast unmatched tracks
        for t_idx, track in enumerate(self.tracks):
            if t_idx not in matched_tracks:
                track.coast()

        # Create new tracks for unmatched detections
        for d_idx in range(num_dets):
            if d_idx not in matched_dets:
                new_track = TrackedBox(
                    track_id=self._next_id,
                    box=raw_boxes[d_idx],
                    score=raw_scores[d_idx],
                    class_id=raw_class_ids[d_idx],
                    class_name=raw_class_names[d_idx],
                )
                self._next_id += 1
                self.tracks.append(new_track)

        # Purge dead tracks that exceed max_age
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]

        # Assemble smoothed output dictionary
        smoothed_boxes: List[List[float]] = []
        smoothed_scores: List[float] = []
        smoothed_class_ids: List[int] = []
        smoothed_class_names: List[str] = []
        track_ids: List[int] = []

        for track in self.tracks:
            # Only display tracks meeting minimum hit threshold and positive confidence
            if track.hits >= self.min_hits and track.score >= 0.25:
                smoothed_boxes.append([round(c, 1) for c in track.box])
                smoothed_scores.append(round(track.score, 3))
                smoothed_class_ids.append(track.class_id)
                smoothed_class_names.append(track.class_name)
                track_ids.append(track.track_id)

        smoothed_results = dict(results)
        smoothed_results["boxes"] = smoothed_boxes
        smoothed_results["scores"] = smoothed_scores
        smoothed_results["class_ids"] = smoothed_class_ids
        smoothed_results["class_names"] = smoothed_class_names
        smoothed_results["track_ids"] = track_ids
        smoothed_results["count"] = len(smoothed_boxes)
        smoothed_results["is_smoothed"] = True

        return smoothed_results
