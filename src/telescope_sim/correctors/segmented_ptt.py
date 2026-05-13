"""Segmented piston/tip/tilt corrector — wraps HCIPy's SegmentedDeformableMirror.

Actuator state has shape ``(n_segments, 3)``: piston (meters), tip-slope,
tilt-slope per segment. Caller-facing values are scaled by ``piston_scale``
and ``tip_tilt_scale``; internally the corrector multiplies through to get
the absolute surface deformation HCIPy expects.

This corrector is the primary "actuate" or "impose" element for the
canonical-family fixtures (mini-ELF and similar segmented designs). It
supports the role-based wavefront / target-strategy system documented on
:class:`telescope_sim.abc.Corrector`.
"""

from __future__ import annotations

from typing import Any

import hcipy
import numpy as np
from numpy.typing import ArrayLike, NDArray

from telescope_sim.abc import Corrector
from telescope_sim.abc.corrector import TargetStrategy, WavefrontRole
from telescope_sim.registry import register


@register("corrector", "segmented_ptt")
class SegmentedPTTCorrector(Corrector):
    """Per-segment piston/tip/tilt actuator on top of HCIPy's SegmentedDeformableMirror.

    Construct from an aperture build result that includes segments
    (``ApertureResult.segments``) and segment coordinates.
    """

    def __init__(
        self,
        segments: Any,
        segment_coords: NDArray[np.floating],
        *,
        name: str = "segments",
        piston_scale: float = 1e-6,
        tip_tilt_scale: float = 1e-6,
        wavefront_role: WavefrontRole = "actuate",
        target_strategy: TargetStrategy = "none",
        fit_source: str | None = None,
        target: bool = False,
    ) -> None:
        if segments is None:
            raise ValueError(
                "SegmentedPTTCorrector requires aperture.segments; use a "
                "segmented aperture (e.g. segmented_circular)."
            )
        self.name = name
        self._sm = hcipy.SegmentedDeformableMirror(segments)
        self._segment_coords = np.asarray(segment_coords, dtype=float)
        self._n_segments = self._segment_coords.shape[0]
        self.piston_scale = float(piston_scale)
        self.tip_tilt_scale = float(tip_tilt_scale)

        self.wavefront_role = wavefront_role
        self.target_strategy = target_strategy
        self.fit_source = fit_source
        self.target = target

    # --- Corrector interface ------------------------------------------------

    def apply(self, wf: Any) -> Any:
        """Apply the segmented mirror's current state to a wavefront."""
        return self._sm(wf)

    def set_actuators(self, values: ArrayLike) -> None:
        """Set actuator state.

        Accepts either:
        - ``(n_segments, 3)`` array — caller-facing PTT values, scaled here
        - flat array of length ``3 * n_segments`` — same content, reshaped
        """
        arr = np.asarray(values, dtype=float)
        if arr.shape == (self._n_segments, 3):
            ptt = arr
        elif arr.shape == (3 * self._n_segments,):
            ptt = arr.reshape(self._n_segments, 3)
        else:
            raise ValueError(
                f"expected shape ({self._n_segments}, 3) or "
                f"({3 * self._n_segments},), got {arr.shape}"
            )
        scaled = ptt.copy()
        scaled[:, 0] *= self.piston_scale
        scaled[:, 1:] *= self.tip_tilt_scale
        # HCIPy's SegmentedDeformableMirror takes a flat (3*n_seg,) vector
        # where each segment contributes 3 consecutive values.
        self._sm.actuators = scaled.reshape(-1)

    def flatten(self) -> None:
        self._sm.actuators = np.zeros(3 * self._n_segments)

    def fit_surface(self, phase: NDArray[np.floating]) -> NDArray[np.floating]:
        """Per-segment least-squares fit of piston/tip/tilt to a pupil-plane phase.

        This mirrors the canonical ``_measure_atmos_ptt`` pattern: for each
        segment, fit ``phase ≈ p + t_x*x + t_y*y`` over the segment's pixels.
        Returns an ``(n_segments, 3)`` array of piston/tip/tilt slopes
        scaled to caller-facing units (i.e. divided by piston_scale and
        tip_tilt_scale, and the standard /2 path-length factor applied).
        """
        if not hasattr(self, "_segment_pixel_data"):
            raise RuntimeError(
                "fit_surface requires segment_pixel_data to be set by the pipeline; "
                "call _bind_pupil_grid() first."
            )

        fits = np.zeros((self._n_segments, 3))
        phase_arr = np.asarray(phase, dtype=float)
        for i, sp in enumerate(self._segment_pixel_data):
            inds, off, xs, ys = sp["inds"], sp["off"], sp["xs"], sp["ys"]
            A = np.vstack([off, xs, ys]).T  # (n_pix, 3)
            rhs = phase_arr[inds]
            x, _, _, _ = np.linalg.lstsq(A, rhs, rcond=None)
            fits[i] = x

        # Remove global mean piston (constant offset does not affect PSF)
        fits[:, 0] -= fits[:, 0].mean()
        # Scale back to caller-facing units. The canonical implementation
        # samples atmosphere at 1 um and divides phase by 2π, then scales by
        # 1e-6 / piston_scale. We expect the *caller* to pass in a phase
        # array in meters; the surface→actuator factor of 2 (round-trip)
        # is applied here.
        fits[:, 0] /= self.piston_scale
        fits[:, 1:] /= self.tip_tilt_scale
        return fits / 2.0

    @property
    def n_actuators(self) -> int:
        return 3 * self._n_segments

    @property
    def actuators(self) -> NDArray:
        """Return the *caller-facing* (n_segments, 3) PTT values."""
        raw = np.asarray(self._sm.actuators).reshape(self._n_segments, 3)
        out = raw.copy()
        out[:, 0] /= self.piston_scale
        out[:, 1:] /= self.tip_tilt_scale
        return out

    @property
    def segment_coords(self) -> NDArray[np.floating]:
        return self._segment_coords

    # --- Pipeline-internal helper ------------------------------------------

    def _bind_pupil_grid(self, pupil_grid: Any, aperture_field: Any) -> None:
        """Precompute per-segment pixel indices for lstsq fitting.

        Called by the pipeline once at construction time so that
        :meth:`fit_surface` can run later.
        """
        x_coords = pupil_grid.x
        y_coords = pupil_grid.y
        aper_arr = np.asarray(aperture_field)
        data: list[dict[str, NDArray]] = []
        for i, (cx, cy) in enumerate(self._segment_coords):
            # Pixels belonging to segment i: any pixel where the i-th segment
            # mask is non-zero AND inside the (spider-cropped) aperture.
            seg_mask = (np.asarray(self._sm.segments[i]) != 0) & (aper_arr != 0)
            inds = np.where(seg_mask)[0]
            xs = (x_coords[inds] - cx)
            ys = (y_coords[inds] - cy)
            off = np.ones_like(xs)
            data.append({"inds": inds, "off": off, "xs": xs, "ys": ys})
        self._segment_pixel_data = data


__all__ = ["SegmentedPTTCorrector"]
