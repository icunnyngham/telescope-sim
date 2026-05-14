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
        # HCIPy's SegmentedDeformableMirror stores actuators in *block*
        # layout: ``[p_0..p_{n-1}, t_0..t_{n-1}, T_0..T_{n-1}]`` (see
        # ``hcipy.SegmentedDeformableMirror.set_segment_actuators``).
        # Match the canonical v1 pattern, which calls
        # ``sm.set_segment_actuators(np.arange(n), pist, tip, tilt)``.
        self._sm.set_segment_actuators(
            np.arange(self._n_segments),
            ptt[:, 0] * self.piston_scale,
            ptt[:, 1] * self.tip_tilt_scale,
            ptt[:, 2] * self.tip_tilt_scale,
        )

    def flatten(self) -> None:
        self._sm.actuators = np.zeros(3 * self._n_segments)

    def fit_surface(self, phase: NDArray[np.floating]) -> NDArray[np.floating]:
        """Per-segment least-squares fit of piston/tip/tilt to a pupil-plane OPD.

        Mirrors the canonical v1 ``_measure_atmos_ptt`` (see
        :class:`Corrector.fit_surface` for the convention). Input is OPD
        in meters; for each segment, fits ``OPD ≈ p + t_x*x + t_y*y``
        over the segment's pixels. Returns matching caller-facing PTT —
        an ``(n_segments, 3)`` array divided by ``piston_scale`` /
        ``tip_tilt_scale`` and by 2 (surface→OPD round-trip). Global
        mean piston removed (doesn't affect PSF).
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
        """Return the *caller-facing* (n_segments, 3) PTT values.

        HCIPy stores actuators in block layout
        ``[p_0..p_{n-1}, t_0..t_{n-1}, T_0..T_{n-1}]`` (not row-major);
        de-interleave back to the per-segment ``(p, t, T)`` rows
        callers expect.
        """
        raw = np.asarray(self._sm.actuators)
        n = self._n_segments
        out = np.empty((n, 3), dtype=float)
        out[:, 0] = raw[0:n] / self.piston_scale
        out[:, 1] = raw[n : 2 * n] / self.tip_tilt_scale
        out[:, 2] = raw[2 * n : 3 * n] / self.tip_tilt_scale
        return out

    @property
    def segment_coords(self) -> NDArray[np.floating]:
        return self._segment_coords

    # --- Pipeline-internal helper ------------------------------------------

    def _bind_pupil_grid(self, pupil_grid: Any, aperture_field: Any) -> None:
        """Precompute per-segment pixel indices for lstsq fitting.

        Called by the pipeline once at construction time so that
        :meth:`fit_surface` can run later.

        Segment masks are made disjoint via argmax across segments —
        each transmitting pixel is assigned to the single segment whose
        anti-aliased mask value is largest there. Touching segments
        (e.g. the elf_15seg ring where center-to-center spacing equals
        segment diameter) otherwise share boundary pixels through
        anti-aliasing, polluting per-segment fits.
        """
        x_coords = pupil_grid.x
        y_coords = pupil_grid.y
        aper_arr = np.asarray(aperture_field)

        # Stack all segment masks into (n_pix, n_seg); assign each
        # transmitting pixel to its argmax-segment.
        seg_stack = np.column_stack([np.asarray(s, dtype=float) for s in self._sm.segments])
        any_seg = seg_stack.max(axis=1) > 0
        labels = np.where(any_seg, seg_stack.argmax(axis=1), -1)
        aper_ok = aper_arr != 0

        data: list[dict[str, NDArray]] = []
        for i, (cx, cy) in enumerate(self._segment_coords):
            seg_mask = (labels == i) & aper_ok
            inds = np.where(seg_mask)[0]
            xs = x_coords[inds] - cx
            ys = y_coords[inds] - cy
            off = np.ones_like(xs)
            data.append({"inds": inds, "off": off, "xs": xs, "ys": ys})
        self._segment_pixel_data = data


__all__ = ["SegmentedPTTCorrector"]
