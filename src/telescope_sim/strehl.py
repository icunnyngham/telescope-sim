"""Strehl ratio estimators — exactly mirroring the legacy ``_strehl``.

Two methods, selected by ``SimConfig.strehl_method``:

- ``"peak"``: legacy ``psf.flat[ref_psf.argmax()] / ref_psf.max()``.
  Reads the *current* PSF at the **reference PSF's argmax** — so a
  tip-tilt that moves the peak off-center correctly drops Strehl below 1.
- ``"matched_filter"``: legacy
  ``sum(psf_core * ref_core) / sum(ref_core**2)``.
  A matched-filter projection over a circular core mask centered on the
  focal-grid origin. Pixels near the reference peak are upweighted by
  the reference intensity itself.

The legacy code precomputed all reference-PSF-derived quantities
(argmax, mask, weighted sums) once per filter at ``__init__``. We do the
same with frozen dataclass estimators built in
:func:`build_strehl_estimator`. The per-sample :meth:`compute` is then
O(1) for peak and O(core_pixels) for matched-filter — no recomputed
argmax or mask.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray


class StrehlEstimator(Protocol):
    """Anything with a cached state and a `.compute(psf) -> float` method."""

    def compute(self, psf: NDArray[np.floating]) -> float: ...


@dataclass(frozen=True, slots=True)
class _PeakStrehl:
    """Peak-pixel estimator with the reference argmax cached.

    Strehl = ``psf.ravel()[peak_index] / reference_peak``.
    """

    reference_peak: float
    peak_index: int

    def compute(self, psf: NDArray[np.floating]) -> float:
        if self.reference_peak <= 0:
            return 0.0
        return float(np.asarray(psf).ravel()[self.peak_index] / self.reference_peak)


@dataclass(frozen=True, slots=True)
class _MatchedFilterStrehl:
    """Matched-filter estimator with mask + reference values + norm cached.

    Strehl = ``sum(psf[core] * ref[core]) / sum(ref[core]**2)``.
    """

    core_mask: NDArray[np.bool_]
    ref_core_vals: NDArray[np.float64]
    ref_core_sq_sum: float

    def compute(self, psf: NDArray[np.floating]) -> float:
        if self.ref_core_sq_sum <= 0:
            return 0.0
        cur_core = np.asarray(psf).ravel()[self.core_mask]
        return float((cur_core * self.ref_core_vals).sum() / self.ref_core_sq_sum)


def build_strehl_estimator(
    method: str,
    reference_psf: NDArray[np.floating],
    focal_grid: Any,
    core_radius_rad: float | None,
) -> StrehlEstimator:
    """Build the cached estimator for the configured method.

    Called once per focal plane at pipeline construction, after the
    at-rest reference PSF has been captured.

    Parameters
    ----------
    method
        ``"peak"`` or ``"matched_filter"``.
    reference_psf
        The at-rest reference PSF (full 2D array).
    focal_grid
        The HCIPy focal grid (``.x`` and ``.y`` flat coordinate arrays).
        Required for ``matched_filter``; ignored for ``peak``.
    core_radius_rad
        Radius of the core mask in radians, measured from the focal-grid
        origin. Required for ``matched_filter``; ignored for ``peak``.
    """
    ref = np.asarray(reference_psf, dtype=np.float64)
    if method == "peak":
        return _PeakStrehl(
            reference_peak=float(ref.max()),
            peak_index=int(np.argmax(ref)),
        )
    if method == "matched_filter":
        if core_radius_rad is None or core_radius_rad <= 0:
            raise ValueError("strehl_method='matched_filter' requires a positive strehl_core_rad")
        x = np.asarray(focal_grid.x)
        y = np.asarray(focal_grid.y)
        mask = (np.sqrt(x**2 + y**2) < float(core_radius_rad)).astype(np.bool_)
        ref_core = ref.ravel()[mask].astype(np.float64, copy=False)
        return _MatchedFilterStrehl(
            core_mask=mask,
            ref_core_vals=ref_core,
            ref_core_sq_sum=float((ref_core**2).sum()),
        )
    raise ValueError(f"unknown strehl_method {method!r}; expected 'peak' or 'matched_filter'")


__all__ = [
    "StrehlEstimator",
    "build_strehl_estimator",
]
