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
        the canonical implementation).
    """

    segment_diameter: float
    layout: str = "elf"
    n_segments: int | None = None
    ring_radius: float | None = None
    positions: NDArray[np.floating] | None = None
    supersample: int = 16

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
        segments = hcipy.evaluate_supersampled(
            segments_callables, pupil_grid, self.supersample
        )

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
