"""Polygon zones and stateful dwell time tracking with hysteresis."""

import time
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
from edgesight.analytics.geometry import get_bottom_center, point_in_polygon


class ZoneEvent:
    """Represents a validated zone interaction event (enter, dwell progress, exit)."""

    def __init__(
        self,
        event_type: str,  # 'ENTER', 'EXIT', 'ALERT_OVERCROWD'
        zone_id: str,
        track_id: int,
        dwell_seconds: float,
        timestamp: float,
    ):
        self.event_type = event_type
        self.zone_id = zone_id
        self.track_id = track_id
        self.dwell_seconds = dwell_seconds
        self.timestamp = timestamp

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event": f"zone_{self.event_type.lower()}",
            "zone_id": self.zone_id,
            "track_id": self.track_id,
            "dwell_seconds": round(self.dwell_seconds, 2),
            "timestamp": self.timestamp,
        }


class PolygonZone:
    """Represents a 2D spatial area of interest (e.g., checkout counter, promo display)."""

    def __init__(
        self,
        zone_id: str,
        polygon: Sequence[Sequence[float]],
        zone_type: str = "general",
        min_enter_frames: int = 3,
        max_exit_frames: int = 5,
        max_capacity_alert: Optional[int] = None,
    ):
        self.zone_id = zone_id
        self.polygon = np.array(polygon, dtype=np.float32)
        self.zone_type = zone_type
        self.min_enter_frames = min_enter_frames
        self.max_exit_frames = max_exit_frames
        self.max_capacity_alert = max_capacity_alert

        # Internal state tracking
        # {track_id: {"state": "CANDIDATE"|"DWELLING", "enter_time": float, "inside_frames": int, "miss_frames": int}}
        self._tracks: Dict[int, Dict[str, Any]] = {}
        self._current_occupancy: int = 0
        self._total_visits: int = 0
        self._completed_dwell_times: List[float] = []

    def update(
        self,
        tracked_results: Dict[str, Any],
        current_time: Optional[float] = None,
    ) -> List[ZoneEvent]:
        """Updates zone state with current frame's detections and emits events."""
        now = current_time if current_time is not None else time.time()
        events: List[ZoneEvent] = []

        boxes = tracked_results.get("boxes", [])
        track_ids = tracked_results.get("track_ids", [])
        if not track_ids and boxes:
            track_ids = list(range(len(boxes)))

        seen_in_zone = set()

        for box, tid in zip(boxes, track_ids):
            tid = int(tid)
            contact_point = get_bottom_center(box)

            if point_in_polygon(contact_point, self.polygon):
                seen_in_zone.add(tid)

                if tid not in self._tracks:
                    # New candidate
                    self._tracks[tid] = {
                        "state": "CANDIDATE",
                        "enter_time": now,
                        "inside_frames": 1,
                        "miss_frames": 0,
                    }
                else:
                    tr = self._tracks[tid]
                    tr["miss_frames"] = 0
                    tr["inside_frames"] += 1

                    if tr["state"] == "CANDIDATE":
                        if tr["inside_frames"] >= self.min_enter_frames:
                            tr["state"] = "DWELLING"
                            self._total_visits += 1
                            events.append(
                                ZoneEvent(
                                    event_type="ENTER",
                                    zone_id=self.zone_id,
                                    track_id=tid,
                                    dwell_seconds=0.0,
                                    timestamp=now,
                                )
                            )

        # Handle tracks currently in memory that were not seen inside this frame
        active_dwelling_count = 0
        exited_tids = []

        for tid, tr in self._tracks.items():
            if tid not in seen_in_zone:
                tr["miss_frames"] += 1

            if tr["state"] == "DWELLING":
                active_dwelling_count += 1

            # Hysteresis exit check
            if tr["miss_frames"] >= self.max_exit_frames:
                if tr["state"] == "DWELLING":
                    dwell_dur = max(0.0, now - tr["enter_time"])
                    self._completed_dwell_times.append(dwell_dur)
                    events.append(
                        ZoneEvent(
                            event_type="EXIT",
                            zone_id=self.zone_id,
                            track_id=tid,
                            dwell_seconds=dwell_dur,
                            timestamp=now,
                        )
                    )
                exited_tids.append(tid)

        for tid in exited_tids:
            self._tracks.pop(tid, None)

        self._current_occupancy = sum(1 for tr in self._tracks.values() if tr["state"] == "DWELLING")

        # Capacity alert check
        if self.max_capacity_alert and self._current_occupancy >= self.max_capacity_alert:
            events.append(
                ZoneEvent(
                    event_type="ALERT_OVERCROWD",
                    zone_id=self.zone_id,
                    track_id=-1,
                    dwell_seconds=float(self._current_occupancy),
                    timestamp=now,
                )
            )

        return events

    def get_stats(self) -> Dict[str, Any]:
        """Returns statistical metrics for this zone."""
        avg_dwell = (
            float(np.mean(self._completed_dwell_times))
            if self._completed_dwell_times
            else 0.0
        )
        return {
            "zone_id": self.zone_id,
            "zone_type": self.zone_type,
            "current_occupancy": self._current_occupancy,
            "total_visits": self._total_visits,
            "avg_dwell_seconds": round(avg_dwell, 2),
            "polygon_points": self.polygon.tolist(),
        }

    def reset(self):
        """Resets tracking memory and occupancy counters."""
        self._tracks.clear()
        self._current_occupancy = 0
        self._total_visits = 0
        self._completed_dwell_times.clear()
