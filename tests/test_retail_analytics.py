"""Unit tests for EdgeSight retail spatial analytics and privacy-first storage."""

import os
import sys
import tempfile
import unittest
import numpy as np

# Ensure src is on path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from edgesight.analytics.geometry import (
    cross_product_2d,
    get_bottom_center,
    point_in_polygon,
    segments_intersect,
)
from edgesight.analytics.heatmap import SpatialHeatmap
from edgesight.analytics.tripwire import LineTripwire, TripwireEvent
from edgesight.analytics.zones import PolygonZone, ZoneEvent
from edgesight.storage.events_db import RetailEventsDB


class TestRetailAnalytics(unittest.TestCase):
    """Test suite for retail spatial analytics and event tracking."""

    def test_geometry_bottom_center(self):
        """Verifies ground plane contact anchor calculation."""
        box = [100.0, 50.0, 200.0, 250.0]
        bx, by = get_bottom_center(box)
        self.assertEqual(bx, 150.0)
        self.assertEqual(by, 250.0)

    def test_geometry_cross_product(self):
        """Verifies 2D vector cross product orientation signs."""
        p1 = (100.0, 100.0)
        p2 = (100.0, 300.0)

        # Point to the right (x=150) -> cross product < 0
        self.assertLess(cross_product_2d(p1, p2, (150.0, 200.0)), 0)

        # Point to the left (x=50) -> cross product > 0
        self.assertGreater(cross_product_2d(p1, p2, (50.0, 200.0)), 0)

        # Point on the line (x=100) -> cross product == 0
        self.assertEqual(cross_product_2d(p1, p2, (100.0, 200.0)), 0)

    def test_geometry_segments_intersect(self):
        """Verifies intersection detection between line segments."""
        line1 = ((100.0, 200.0), (300.0, 200.0))
        # Crossing line
        cross_seg = ((200.0, 100.0), (200.0, 300.0))
        self.assertTrue(segments_intersect(line1[0], line1[1], cross_seg[0], cross_seg[1]))

        # Non-intersecting parallel line
        parallel_seg = ((100.0, 250.0), (300.0, 250.0))
        self.assertFalse(segments_intersect(line1[0], line1[1], parallel_seg[0], parallel_seg[1]))

    def test_geometry_point_in_polygon(self):
        """Verifies polygon containment test."""
        polygon = [[100, 100], [300, 100], [300, 300], [100, 300]]
        self.assertTrue(point_in_polygon((200.0, 200.0), polygon))
        self.assertFalse(point_in_polygon((50.0, 50.0), polygon))
        self.assertFalse(point_in_polygon((350.0, 200.0), polygon))

    def test_line_tripwire_crossing(self):
        """Verifies directional line tripwire crossing and anti-jitter cooldown."""
        # Horizontal tripwire from x=100 to x=400 at y=200 (crossing downwards: y=150 -> y=250)
        tripwire = LineTripwire(
            line_id="entrance",
            p1=(100.0, 200.0),
            p2=(400.0, 200.0),
            positive_direction="IN",
            negative_direction="OUT",
            cooldown_sec=2.0,
        )

        # Frame 1: Object above tripwire (feet at y=150)
        res1 = {
            "boxes": [[200.0, 50.0, 300.0, 150.0]],
            "track_ids": [1],
        }
        evts1 = tripwire.update(res1, current_time=100.0)
        self.assertEqual(len(evts1), 0)

        # Frame 2: Object moves across tripwire (feet at y=250)
        res2 = {
            "boxes": [[200.0, 150.0, 300.0, 250.0]],
            "track_ids": [1],
        }
        evts2 = tripwire.update(res2, current_time=100.1)
        self.assertEqual(len(evts2), 1)
        self.assertEqual(evts2[0].direction, "IN")
        self.assertEqual(tripwire.in_count, 1)

        # Frame 3: Immediate subsequent frame (still past the line, cooldown active)
        res3 = {
            "boxes": [[200.0, 160.0, 300.0, 260.0]],
            "track_ids": [1],
        }
        evts3 = tripwire.update(res3, current_time=100.2)
        self.assertEqual(len(evts3), 0)
        self.assertEqual(tripwire.in_count, 1)  # No double count

    def test_polygon_zone_dwell_lifecycle(self):
        """Verifies zone entry, dwell duration, and exit hysteresis."""
        # Zone covering (100,100) to (300,300)
        zone = PolygonZone(
            zone_id="promo_shelf",
            polygon=[[100, 100], [300, 100], [300, 300], [100, 300]],
            min_enter_frames=2,
            max_exit_frames=3,
        )

        # Frame 1: Enters zone (Candidate frame 1)
        res1 = {"boxes": [[150.0, 100.0, 250.0, 200.0]], "track_ids": [10]}
        evts1 = zone.update(res1, current_time=10.0)
        self.assertEqual(len(evts1), 0)
        self.assertEqual(zone.get_stats()["current_occupancy"], 0)

        # Frame 2: Still in zone (Candidate frame 2 >= min_enter_frames -> ENTER)
        evts2 = zone.update(res1, current_time=11.0)
        self.assertEqual(len(evts2), 1)
        self.assertEqual(evts2[0].event_type, "ENTER")
        self.assertEqual(zone.get_stats()["current_occupancy"], 1)

        # Frame 3, 4: Object disappears from zone (Miss 1, Miss 2)
        zone.update({"boxes": [], "track_ids": []}, current_time=12.0)
        zone.update({"boxes": [], "track_ids": []}, current_time=13.0)
        self.assertEqual(zone.get_stats()["current_occupancy"], 1)  # Still coasting

        # Frame 5: Miss 3 >= max_exit_frames -> EXIT event triggered
        evts5 = zone.update({"boxes": [], "track_ids": []}, current_time=14.0)
        self.assertEqual(len(evts5), 1)
        self.assertEqual(evts5[0].event_type, "EXIT")
        self.assertAlmostEqual(evts5[0].dwell_seconds, 4.0, places=1)
        self.assertEqual(zone.get_stats()["current_occupancy"], 0)

    def test_spatial_heatmap(self):
        """Verifies heatmap density accumulation and decay without numerical overflow."""
        heatmap = SpatialHeatmap(width=320, height=240, scale=0.5, decay_factor=0.9)
        self.assertEqual(heatmap.accumulator.sum(), 0.0)

        # Add detection
        res = {"boxes": [[100.0, 50.0, 140.0, 100.0]]}
        heatmap.update(res)
        first_sum = heatmap.accumulator.sum()
        self.assertGreater(first_sum, 0.0)

        # Update empty frame -> decay occurs
        heatmap.update({"boxes": []})
        second_sum = heatmap.accumulator.sum()
        self.assertLess(second_sum, first_sum)
        self.assertAlmostEqual(second_sum, first_sum * 0.9, places=2)

        # Test overlay rendering
        dummy_frame = np.zeros((240, 320, 3), dtype=np.uint8)
        rendered = heatmap.render_overlay(dummy_frame)
        self.assertEqual(rendered.shape, (240, 320, 3))

    def test_retail_events_db(self):
        """Verifies local SQLite logging and aggregation of retail events."""
        tf = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        db_path = tf.name
        tf.close()

        try:
            db = RetailEventsDB(db_path=db_path)

            # Log tripwire events
            db.log_tripwire_event(
                TripwireEvent("door", 1, "IN", 100.0, (150.0, 200.0))
            )
            db.log_tripwire_event(
                TripwireEvent("door", 2, "IN", 105.0, (160.0, 200.0))
            )
            db.log_tripwire_event(
                TripwireEvent("door", 1, "OUT", 150.0, (150.0, 200.0))
            )

            counts = db.get_tripwire_counts("door")
            self.assertEqual(counts["IN"], 2)
            self.assertEqual(counts["OUT"], 1)

            # Log zone events
            db.log_zone_event(ZoneEvent("EXIT", "shelf_a", 1, 15.5, 120.0))
            db.log_zone_event(ZoneEvent("EXIT", "shelf_a", 2, 24.5, 140.0))

            summary = db.get_zone_summary("shelf_a")
            self.assertEqual(summary["total_exits"], 2)
            self.assertEqual(summary["avg_dwell_seconds"], 20.0)

        finally:
            if os.path.exists(db_path):
                os.remove(db_path)


if __name__ == "__main__":
    unittest.main()
