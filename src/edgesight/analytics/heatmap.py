"""2D Spatial density heatmap with temporal exponential decay."""

from typing import Any, Dict, List, Optional, Sequence, Tuple
import cv2
import numpy as np
from edgesight.analytics.geometry import get_bottom_center


class SpatialHeatmap:
    """Accumulative 2D foot-traffic density heatmap with exponential time decay."""

    def __init__(
        self,
        width: int = 640,
        height: int = 480,
        scale: float = 0.5,
        decay_factor: float = 0.995,
        kernel_radius: int = 25,
        intensity_per_point: float = 1.0,
    ):
        """Initializes a spatial heatmap accumulator.

        Args:
            width: Full frame width.
            height: Full frame height.
            scale: Downscaling factor for accumulator grid (e.g. 0.5 = 320x240 for fast CPU ops).
            decay_factor: Temporal decay multiplier per frame (0.995 = slow decay, 0.95 = fast decay).
            kernel_radius: Pixel radius of the footprint Gaussian kernel.
            intensity_per_point: Energy injected per detection anchor.
        """
        self.orig_w = width
        self.orig_h = height
        self.scale = scale
        self.decay = decay_factor
        self.radius = max(3, int(kernel_radius * scale))
        self.intensity = intensity_per_point

        self.grid_w = max(10, int(width * scale))
        self.grid_h = max(10, int(height * scale))

        # Float32 accumulation matrix
        self.accumulator = np.zeros((self.grid_h, self.grid_w), dtype=np.float32)

        # Precompute circular Gaussian footprint kernel
        self._kernel = self._create_gaussian_kernel(self.radius)

    def _create_gaussian_kernel(self, r: int) -> np.ndarray:
        """Precomputes a normalized 2D Gaussian kernel patch."""
        d = 2 * r + 1
        y, x = np.ogrid[-r : r + 1, -r : r + 1]
        dist_sq = x * x + y * y
        sigma = r / 2.0
        kernel = np.exp(-dist_sq / (2.0 * sigma * sigma)).astype(np.float32)
        kernel[dist_sq > r * r] = 0.0
        max_v = kernel.max()
        if max_v > 0:
            kernel /= max_v
        return kernel

    def update(self, tracked_results: Dict[str, Any]):
        """Injects current frame's feet anchors and applies temporal decay."""
        # Step 1: Exponential decay of historical density
        self.accumulator *= self.decay

        boxes = tracked_results.get("boxes", [])
        if not boxes:
            return

        r = self.radius
        k_h, k_w = self._kernel.shape

        for box in boxes:
            bx, by = get_bottom_center(box)
            gx = int(bx * self.scale)
            gy = int(by * self.scale)

            # Check bounds
            if 0 <= gx < self.grid_w and 0 <= gy < self.grid_h:
                # Kernel bounding coordinates on accumulator grid
                x_min = max(0, gx - r)
                x_max = min(self.grid_w, gx + r + 1)
                y_min = max(0, gy - r)
                y_max = min(self.grid_h, gy + r + 1)

                # Matching sub-kernel slice
                kx_min = x_min - (gx - r)
                kx_max = kx_min + (x_max - x_min)
                ky_min = y_min - (gy - r)
                ky_max = ky_min + (y_max - y_min)

                self.accumulator[y_min:y_max, x_min:x_max] += (
                    self._kernel[ky_min:ky_max, kx_min:kx_max] * self.intensity
                )

    def render_overlay(
        self,
        frame: np.ndarray,
        alpha: float = 0.4,
        colormap: int = cv2.COLORMAP_JET,
        threshold_ratio: float = 0.05,
    ) -> np.ndarray:
        """Blends colored heatmap onto the target frame."""
        max_val = float(self.accumulator.max())
        if max_val <= 1e-5:
            return frame

        # Normalize to 0..255
        norm = np.clip(self.accumulator / max_val * 255.0, 0, 255).astype(np.uint8)

        # Upscale to full frame resolution
        h, w = frame.shape[:2]
        full_heat = cv2.resize(norm, (w, h), interpolation=cv2.INTER_LINEAR)

        # Threshold out cold background
        thresh_val = int(255 * threshold_ratio)
        mask = full_heat > thresh_val

        # Colorize
        colored = cv2.applyColorMap(full_heat, colormap)

        # Alpha blend where active
        out_frame = frame.copy()
        out_frame[mask] = cv2.addWeighted(
            colored, alpha, out_frame, 1.0 - alpha, 0
        )[mask]

        return out_frame

    def reset(self):
        """Clears heatmap accumulator."""
        self.accumulator.fill(0.0)
