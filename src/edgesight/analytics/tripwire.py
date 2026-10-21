"""Virtual line tripwire module for directional foot-traffic counting."""

import time
from typing import Any, Dict, List, Optional, Tuple
from edgesight.analytics.geometry import cross_product_2d, get_bottom_center, segments_intersect


class TripwireEvent:
    """Represents a validated line crossing event."""

    def __init__(
        self,
        line_id: str,
        track_id: int,
        direction: str,
        timestamp: float,
        position: Tuple[float, float],
    ):
        self.line_id = line_id
        self.track_id = track_id
        self.direction = direction  # 'IN' or 'OUT'
        self.timestamp = timestamp
        self.position = position

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event": "tripwire_crossing",
            "line_id": self.line_id,
            "track_id": self.track_id,
            "direction": self.direction,
            "timestamp": self.timestamp,
            "position": [round(self.position[0], 2), round(self.position[1], 2)],
        }


class LineTripwire:
    """Directional virtual tripwire monitoring tracked object trajectories."""

    def __init__(
        self,
        line_id: str,
        p1: Tuple[float, float],
        p2: Tuple[float, float],
        positive_direction: str = "IN",
        negative_direction: str = "OUT",
        cooldown_sec: float = 3.0,
        max_history_age: int = 30,
    ):
        """Initializes a virtual line tripwire.

        Args:
            line_id: Unique identifier for this line (e.g. 'store_entrance').
            p1: (x, y) start point of the directed line segment.
            p2: (x, y) end point of the directed line segment.
            positive_direction: Event label when crossing from Left to Right (>0 -> <0).
            negative_direction: Event label when crossing from Right to Left (<0 -> >0).
            cooldown_sec: Cooldown period preventing immediate re-trigger for same track ID.
            max_history_age: Max frames to retain position history for inactive tracks.
        """
        self.line_id = line_id
        self.p1 = (float(p1[0]), float(p1[1]))
        self.p2 = (float(p2[0]), float(p2[1]))
        self.pos_dir = positive_direction
        self.neg_dir = negative_direction
        self.cooldown_sec = cooldown_sec
        self.max_history_age = max_history_age

        # Counters
        self.in_count = 0
        self.out_count = 0

        # State tracking: {track_id: (x, y)}
        self._prev_positions: Dict[int, Tuple[float, float]] = {}
        # Track last seen frame index: {track_id: frame_index}
        self._last_seen_frame: Dict[int, int] = {}
        # Cooldown map: {track_id: expire_timestamp}
        self._cooldown: Dict[int, float] = {}
        self._frame_idx = 0

    def update(
        self,
        tracked_results: Dict[str, Any],
        current_time: Optional[float] = None,
    ) -> List[TripwireEvent]:
        """Processes current frame's tracked objects and returns any crossing events.

        Args:
            tracked_results: Dictionary containing 'boxes' and 'track_ids' from tracker.
            current_time: Optional explicit timestamp (defaults to time.time()).

        Returns:
            List of TripwireEvent objects triggered in this frame.
        """
        self._frame_idx += 1
        now = current_time if current_time is not None else time.time()
        events: List[TripwireEvent] = []

        boxes = tracked_results.get("boxes", [])
        track_ids = tracked_results.get("track_ids", [])

        # If tracker has not assigned track_ids, fallback to enumeration
        if not track_ids and boxes:
            track_ids = list(range(len(boxes)))

        active_track_ids = set()

        for box, tid in zip(boxes, track_ids):
            tid = int(tid)
            active_track_ids.add(tid)
            curr_pos = get_bottom_center(box)

            if tid in self._prev_positions:
                prev_pos = self._prev_positions[tid]

                # Check if movement segment intersects the tripwire line segment
                if segments_intersect(self.p1, self.p2, prev_pos, curr_pos):
                    # Check cooldown
                    if now >= self._cooldown.get(tid, 0.0):
                        cp_prev = cross_product_2d(self.p1, self.p2, prev_pos)
                        cp_curr = cross_product_2d(self.p1, self.p2, curr_pos)

                        direction = None
                        if cp_prev <= 0 and cp_curr > 0:
                            direction = self.pos_dir
                            self.in_count += 1
                        elif cp_prev >= 0 and cp_curr < 0:
                            direction = self.neg_dir
                            self.out_count += 1

                        if direction is not None:
                            evt = TripwireEvent(
                                line_id=self.line_id,
                                track_id=tid,
                                direction=direction,
                                timestamp=now,
                                position=curr_pos,
                            )
                            events.append(evt)
                            self._cooldown[tid] = now + self.cooldown_sec

            self._prev_positions[tid] = curr_pos
            self._last_seen_frame[tid] = self._frame_idx

        # Purge stale tracks to prevent unbounded memory growth
        if self._frame_idx % 50 == 0:
            stale_tids = [
                tid
                for tid, last_f in self._last_seen_frame.items()
                if (self._frame_idx - last_f) > self.max_history_age
            ]
            for tid in stale_tids:
                self._prev_positions.pop(tid, None)
                self._last_seen_frame.pop(tid, None)
                self._cooldown.pop(tid, None)

        return events

    def get_stats(self) -> Dict[str, Any]:
        """Returns cumulative traffic counts and line state."""
        return {
            "line_id": self.line_id,
            "in_count": self.in_count,
            "out_count": self.out_count,
            "total_crossings": self.in_count + self.out_count,
            "p1": list(self.p1),
            "p2": list(self.p2),
        }

    def reset(self):
        """Resets tripwire counters and tracking memory."""
        self.in_count = 0
        self.out_count = 0
        self._prev_positions.clear()
        self._last_seen_frame.clear()
        self._cooldown.clear()
