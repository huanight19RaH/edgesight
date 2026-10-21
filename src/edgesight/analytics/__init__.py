"""EdgeSight retail spatial analytics module."""

from edgesight.analytics.geometry import (
    cross_product_2d,
    get_bottom_center,
    point_in_polygon,
    segments_intersect,
)
from edgesight.analytics.heatmap import SpatialHeatmap
from edgesight.analytics.tripwire import LineTripwire, TripwireEvent
from edgesight.analytics.zones import PolygonZone, ZoneEvent

__all__ = [
    "get_bottom_center",
    "cross_product_2d",
    "segments_intersect",
    "point_in_polygon",
    "LineTripwire",
    "TripwireEvent",
    "PolygonZone",
    "ZoneEvent",
    "SpatialHeatmap",
]
