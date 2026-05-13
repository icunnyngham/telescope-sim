"""Segmented-circular aperture: N circular sub-apertures arranged on a layout.

Supported layouts:
    - "elf": ring of sub-apertures at a given radius
    - "custom": user-provided positions

This is the minimal mini-ELF / DASIE aperture builder the canonical-family
fixtures use. Hexagonal / monolithic / external_pupil variants are separate
implementations.
"""

from __future__ import annotations

from dataclasses import dataclass

import hcipy
import numpy as np
from numpy.typing import NDArray

from telescope_sim.abc import Aperture, ApertureResult
from telescope_sim.registry import register


@dataclass
class SegmentedCircularAperture(Aperture):
    """N circular sub-apertures on a configurable layout.

    Parameters
    ----------
    segment_diameter
        Diameter of each sub-aperture in pupil-grid units.
    layout
        "elf" or "custom".
    n_segments
        Number of segments (required for "elf").
    ring_radius
        Radius of the ELF ring (required for "elf").
    positions
        Explicit (N, 2) list of (x, y) coordinates (required for "custom").
    supersample
        Supersampling factor for aperture evaluation (default 16, matching
        the canonical 2024-09 implementation; older variants used 1).
    spider
        Optional dict ``{width, angle}``. If provided, two perpendicular
        spiders are multiplied into the aperture mask. ``angle`` is in
        degrees; the second spider is offset by 90°. Mirrors the canonical
        spider construction.
    """

    segment_diameter: float
    layout: str = "elf"
    n_segments: int | None = None
    ring_radius: float | None = None
    positions: NDArray[np.floating] | None = None
    supersample: int = 16
    spider: dict | None = None

    def __post_init__(self) -> None:
        if self.layout not in ("elf", "custom"):
            raise ValueError(f"unsupported layout {self.layout!r}; expected 'elf' or 'custom'")

    def _build_centers(self) -> tuple[NDArray, int]:
        """Return (n_segments, 2) array of segment centers and the count."""
        if self.layout == "elf":
            if self.n_segments is None or self.ring_radius is None:
                raise ValueError("layout='elf' requires n_segments and ring_radius")
            angles = np.linspace(0, 2 * np.pi, self.n_segments + 1)[:-1]
            xs = self.ring_radius * np.cos(angles)
            ys = self.ring_radius * np.sin(angles)
            return np.column_stack([xs, ys]), self.n_segments

        positions = np.asarray(self.positions, dtype=float)
        if positions.ndim != 2 or positions.shape[1] != 2:
            raise ValueError(f"layout='custom' positions must be (N, 2); got {positions.shape}")
        return positions, positions.shape[0]

    def build(self, pupil_grid) -> ApertureResult:
        centers, n_seg = self._build_centers()
        # HCIPy expects centers as a CartesianGrid
        mir_centers = hcipy.CartesianGrid(np.array([centers[:, 0], centers[:, 1]]))

        aper_shape = hcipy.make_circular_aperture(self.segment_diameter)
        aper_callable, segments_callables = hcipy.make_segmented_aperture(
            aper_shape, mir_centers, return_segments=True
        )

        aper_field = hcipy.evaluate_supersampled(aper_callable, pupil_grid, self.supersample)
        segments = hcipy.evaluate_supersampled(segments_callables, pupil_grid, self.supersample)

        # Optional spider — multiply into the aperture mask (does NOT affect
        # segment masks, which the canonical implementations also leave alone).
        if self.spider is not None:
            width = float(self.spider["width"])
            angle_deg = float(self.spider.get("angle", 0.0))
            angle = angle_deg * np.pi / 180.0
            # Use the pupil grid's extent as the spider half-length
            x_arr = np.asarray(pupil_grid.x)
            y_arr = np.asarray(pupil_grid.y)
            p_ext = float(max(x_arr.max() - x_arr.min(), y_arr.max() - y_arr.min()))
            s1_start = (p_ext * np.cos(angle), p_ext * np.sin(angle))
            s1_end = (p_ext * np.cos(angle + np.pi), p_ext * np.sin(angle + np.pi))
            s2_start = (
                p_ext * np.cos(angle + np.pi / 2),
                p_ext * np.sin(angle + np.pi / 2),
            )
            s2_end = (
                p_ext * np.cos(angle + np.pi + np.pi / 2),
                p_ext * np.sin(angle + np.pi + np.pi / 2),
            )
            spider1 = hcipy.aperture.generic.make_spider(s1_start, s1_end, width)
            spider2 = hcipy.aperture.generic.make_spider(s2_start, s2_end, width)
            aper_field = aper_field * spider1(pupil_grid) * spider2(pupil_grid)

        D = self.segment_diameter
        area = n_seg * np.pi * (D / 2.0) ** 2

        # Segment center coordinates as numpy array (for downstream PTT measurement)
        segment_coords = centers

        return ApertureResult(
            field=aper_field,
            area=area,
            segments=segments,
            segment_coords=segment_coords,
            metadata={
                "n_segments": n_seg,
                "segment_diameter": D,
                "layout": self.layout,
            },
        )


register("aperture", "segmented_circular")(SegmentedCircularAperture)

__all__ = ["SegmentedCircularAperture"]
