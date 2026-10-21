"""Privacy-first SQLite event storage for on-device retail telemetry."""

from contextlib import contextmanager
import os
import sqlite3
import time
from typing import Any, Dict, List, Optional
from edgesight.analytics.tripwire import TripwireEvent
from edgesight.analytics.zones import ZoneEvent


class RetailEventsDB:
    """Manages local SQLite database logging privacy-safe structured analytics metadata.

    Guarantees: Zero raw video or facial biometric data is stored.
    Only anonymized numerical coordinates, timestamps, and zone durations are persisted.
    """

    def __init__(self, db_path: str = "raw/retail_events.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
        self._init_db()

    @contextmanager
    def _connection(self):
        """Context manager guaranteeing connection closure to prevent Windows file locks."""
        conn = sqlite3.connect(self.db_path)
        try:
            yield conn
        finally:
            conn.close()

    def _init_db(self):
        """Initializes tables for tripwire crossings and zone dwell events."""
        with self._connection() as conn:
            with conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS tripwire_events (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp REAL NOT NULL,
                        line_id TEXT NOT NULL,
                        track_id INTEGER NOT NULL,
                        direction TEXT NOT NULL
                    )
                    """
                )
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS zone_events (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp REAL NOT NULL,
                        zone_id TEXT NOT NULL,
                        track_id INTEGER NOT NULL,
                        event_type TEXT NOT NULL,
                        dwell_seconds REAL NOT NULL
                    )
                    """
                )
                cursor.execute(
                    "CREATE INDEX IF NOT EXISTS idx_tripwire_time ON tripwire_events(timestamp)"
                )
                cursor.execute(
                    "CREATE INDEX IF NOT EXISTS idx_zone_time ON zone_events(timestamp)"
                )

    def log_tripwire_event(self, event: TripwireEvent):
        """Persists a directional line-crossing event."""
        with self._connection() as conn:
            with conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO tripwire_events (timestamp, line_id, track_id, direction)
                    VALUES (?, ?, ?, ?)
                    """,
                    (event.timestamp, event.line_id, event.track_id, event.direction),
                )

    def log_zone_event(self, event: ZoneEvent):
        """Persists a zone entry, exit, or alert event."""
        with self._connection() as conn:
            with conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO zone_events (timestamp, zone_id, track_id, event_type, dwell_seconds)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        event.timestamp,
                        event.zone_id,
                        event.track_id,
                        event.event_type,
                        event.dwell_seconds,
                    ),
                )

    def get_tripwire_counts(self, line_id: Optional[str] = None) -> Dict[str, int]:
        """Calculates aggregated IN/OUT counts from persisted events."""
        query = "SELECT direction, COUNT(*) FROM tripwire_events"
        params = []
        if line_id:
            query += " WHERE line_id = ?"
            params.append(line_id)
        query += " GROUP BY direction"

        counts = {"IN": 0, "OUT": 0}
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            for direction, count in cursor.fetchall():
                if direction in counts:
                    counts[direction] = count
        return counts

    def get_zone_summary(self, zone_id: Optional[str] = None) -> Dict[str, Any]:
        """Calculates visit counts and average dwell time."""
        query = (
            "SELECT COUNT(*), AVG(dwell_seconds) FROM zone_events WHERE event_type = 'EXIT'"
        )
        params = []
        if zone_id:
            query += " AND zone_id = ?"
            params.append(zone_id)

        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            row = cursor.fetchone()
            total_visits = row[0] or 0
            avg_dwell = row[1] or 0.0

        return {
            "zone_id": zone_id or "all",
            "total_exits": total_visits,
            "avg_dwell_seconds": round(avg_dwell, 2),
        }
