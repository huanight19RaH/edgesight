"""Geometric primitives and spatial calculations for retail edge analytics."""

from typing import List, Sequence, Tuple, Union
import cv2
import numpy as np


def get_bottom_center(box: Sequence[float]) -> Tuple[float, float]:
    """Calculates ground plane contact anchor (bottom-center) from bounding box.

    In overhead/tilted surveillance cameras, the bottom-center coordinate (feet contact)
    accurately represents the subject's position on the floor, eliminating parallax distortion
    inherent to the box centroid.

    Args:
        box: [x1, y1, x2, y2] bounding box coordinates.

    Returns:
        (x_bottom, y_bottom) floor contact anchor point.
    """
    x1, y1, x2, y2 = box[:4]
    x_bottom = (float(x1) + float(x2)) * 0.5
    y_bottom = float(y2)
    return x_bottom, y_bottom


def cross_product_2d(
    p1: Tuple[float, float],
    p2: Tuple[float, float],
    p: Tuple[float, float],
) -> float:
    """Calculates 2D cross product of vector (P1 -> P2) and vector (P1 -> P).

    The sign of the cross product determines which side of the directed line (P1 -> P2)
    the point P lies on:
      > 0: Point is to the LEFT of the directed line.
      < 0: Point is to the RIGHT of the directed line.
      = 0: Point is collinear with the line.

    Formula: (P2_x - P1_x) * (P_y - P1_y) - (P2_y - P1_y) * (P_x - P1_x)
    """
    return (p2[0] - p1[0]) * (p[1] - p1[1]) - (p2[1] - p1[1]) * (p[0] - p1[0])


def segments_intersect(
    a1: Tuple[float, float],
    a2: Tuple[float, float],
    b1: Tuple[float, float],
    b2: Tuple[float, float],
) -> bool:
    """Checks whether line segment A (a1 -> a2) intersects line segment B (b1 -> b2)."""
    cp1 = cross_product_2d(a1, a2, b1)
    cp2 = cross_product_2d(a1, a2, b2)
    cp3 = cross_product_2d(b1, b2, a1)
    cp4 = cross_product_2d(b1, b2, a2)

    # Segments intersect if endpoints of each segment straddle the line of the other
    if ((cp1 > 0 and cp2 < 0) or (cp1 < 0 and cp2 > 0)) and (
        (cp3 > 0 and cp4 < 0) or (cp3 < 0 and cp4 > 0)
    ):
        return True

    return False


def point_in_polygon(
    point: Tuple[float, float],
    polygon: Union[np.ndarray, Sequence[Sequence[float]]],
) -> bool:
    """Determines whether a 2D point lies inside or on the boundary of a closed polygon.

    Uses cv2.pointPolygonTest for high-performance C++ vectorized evaluation.

    Args:
        point: (x, y) query point.
        polygon: Array of [x, y] vertices.

    Returns:
        True if point is strictly inside or on the boundary, False otherwise.
    """
    pts = np.array(polygon, dtype=np.int32 if np.issubdtype(np.array(polygon).dtype, np.integer) else np.float32)
    if len(pts.shape) == 2:
        pts = pts.reshape((-1, 1, 2))

    res = cv2.pointPolygonTest(pts, (float(point[0]), float(point[1])), measureDist=False)
    return res >= 0
