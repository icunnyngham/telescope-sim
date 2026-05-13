"""Strehl ratio computation — peak-pixel and core-integral methods.

Both methods compare the current PSF against the reference PSF generated
at-rest (no actuators, no atmosphere, identity coronagraph) for the same
focal plane.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def peak_pixel_strehl(psf: NDArray[np.floating], reference_peak: float) -> float:
    """Strehl ≈ current peak / reference peak.

    The cheapest and most common Strehl estimator. Identical to the
    canonical implementation when ``strehl_core_rad`` is ``None``.
    """
    if reference_peak <= 0:
        return 0.0
    return float(np.max(psf) / reference_peak)


def core_integral_strehl(
    psf: NDArray[np.floating],
    reference_psf: NDArray[np.floating],
    focal_grid: object,
    core_radius_rad: float,
) -> float:
    """Strehl as the ratio of energy within a core radius of the reference peak.

    Mirrors the canonical implementation when ``strehl_core_rad`` is set:
    integrate ``psf`` and ``reference_psf`` inside the same circular mask
    (centered on the reference PSF peak), take their ratio. Returns 0 if
    the reference integral is non-positive.

    ``focal_grid`` is an HCIPy grid; we use its ``x`` and ``y`` flat
    coordinate arrays to build the radial mask.
    """
    ref = np.asarray(reference_psf, dtype=np.float64)
    cur = np.asarray(psf, dtype=np.float64)
    peak_idx = int(np.argmax(ref))
    x = np.asarray(focal_grid.x)
    y = np.asarray(focal_grid.y)
    x0, y0 = float(x[peak_idx]), float(y[peak_idx])
    r2 = (x - x0) ** 2 + (y - y0) ** 2
    mask = r2 <= core_radius_rad ** 2
    ref_int = float(ref.ravel()[mask].sum())
    if ref_int <= 0:
        return 0.0
    cur_int = float(cur.ravel()[mask].sum())
    return cur_int / ref_int


__all__ = ["peak_pixel_strehl", "core_integral_strehl"]
