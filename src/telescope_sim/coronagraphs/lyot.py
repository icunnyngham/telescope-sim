"""Classical Lyot coronagraph: hard-edged focal-plane occulter + Lyot stop.

The occulter is an opaque circular spot of ``occulter_diameter`` in the
focal plane, evaluated (supersampled, so the hard edge is anti-aliased) on
a small dedicated mask grid; the region outside that grid is fully
transmissive. Propagation uses the semi-analytical scheme of Soummer et
al. 2007 — on the hcipy backend via :class:`hcipy.LyotCoronagraph`
(Babinet's principle internally), on the jax backend via matching matrix
Fourier transforms built from the same geometry arrays (see
``backends/jax/propagation.py``).

Units: the occulter diameter and mask extent are focal-plane coordinates
of the coronagraph's *internal* lens system — radians on-sky for the
default ``focal_length=1.0`` (the angular convention), meters at the
internal focal plane otherwise.
"""

from __future__ import annotations

from typing import Any

import hcipy

from telescope_sim.abc import Coronagraph
from telescope_sim.coronagraphs.standard import _resolve_lyot_field
from telescope_sim.registry import register


@register("coronagraph", "lyot")
class LyotCoronagraphImpl(Coronagraph):
    """Classical Lyot coronagraph, supported on both compute backends.

    Parameters
    ----------
    occulter_diameter
        Diameter of the opaque focal-plane spot, in the internal focal
        plane's coordinates (radians for ``focal_length=1.0``).
    mask_resolution
        Pixels per side of the small focal-plane-mask grid.
    mask_extent
        Full extent of the mask grid (same units as the diameter).
        Defaults to ``2 * occulter_diameter``. Everything outside this
        grid is treated as fully transmissive, so it only needs to cover
        the spot.
    focal_length
        Internal focal length of the Lyot system (not the science focal
        plane's). The default 1.0 puts the mask in angular units.
    lyot
        Optional Lyot-stop ``Aperture`` sub-config (``{type: ..., ...}``),
        resolved on the pupil grid exactly like the vortex kinds' stops.
    oversample
        Supersampling factor for evaluating the hard-edged spot.
    """

    name = "lyot"
    supported_backends = frozenset({"hcipy", "jax"})

    def __init__(
        self,
        occulter_diameter: float,
        *,
        mask_resolution: int = 128,
        mask_extent: float | None = None,
        focal_length: float = 1.0,
        lyot: dict | None = None,
        oversample: int = 8,
    ) -> None:
        self.occulter_diameter = float(occulter_diameter)
        if self.occulter_diameter <= 0:
            raise ValueError("occulter_diameter must be positive")
        self.mask_resolution = int(mask_resolution)
        if self.mask_resolution < 2:
            raise ValueError("mask_resolution must be at least 2")
        self.mask_extent = (
            2.0 * self.occulter_diameter if mask_extent is None else float(mask_extent)
        )
        if self.mask_extent < self.occulter_diameter:
            raise ValueError(
                f"mask_extent ({self.mask_extent}) must cover the occulter "
                f"(diameter {self.occulter_diameter})"
            )
        self.focal_length = float(focal_length)
        self.oversample = int(oversample)
        self._lyot_cfg = lyot
        # Geometry built by _bind_pupil_grid; shared verbatim by the jax
        # backend's kernel builder so both backends see identical masks.
        self.mask_grid: Any | None = None
        self.occulter: Any | None = None  # spot transmission-loss field on mask_grid
        self.lyot_field: Any | None = None  # stop transmission field on the pupil grid
        self._coro: Any | None = None

    def _bind_pupil_grid(self, pupil_grid: Any) -> None:
        self.mask_grid = hcipy.make_uniform_grid([self.mask_resolution] * 2, [self.mask_extent] * 2)
        self.occulter = hcipy.evaluate_supersampled(
            hcipy.make_circular_aperture(self.occulter_diameter), self.mask_grid, self.oversample
        )
        self.lyot_field = _resolve_lyot_field(self._lyot_cfg, pupil_grid)
        # hcipy's focal_plane_mask is the mask-region *transmission*: an
        # opaque spot is 1 - occulter. Its forward() computes the Babinet
        # subtraction of the blocked (occulter) part.
        self._coro = hcipy.LyotCoronagraph(
            pupil_grid,
            focal_plane_mask=1.0 - self.occulter,
            lyot_stop=self.lyot_field,
            focal_length=self.focal_length,
            focal_plane_mask_grid=self.mask_grid,
        )

    def apply(self, wf: Any) -> Any:
        if self._coro is None:
            raise RuntimeError("LyotCoronagraph must be bound via _bind_pupil_grid()")
        return self._coro(wf)


__all__ = ["LyotCoronagraphImpl"]
