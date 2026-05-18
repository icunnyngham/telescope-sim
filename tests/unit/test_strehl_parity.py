"""Strehl-estimator parity tests against the legacy ``_strehl`` formulas.

These are pure-Python unit tests on toy PSFs — no HCIPy sampler involved.
They pin the two formulas independently of any optical-chain coupling and
serve as the cheap-CI counterpart to the legacy-environment fixture
``16_strehl_zernike``, which exercises the same formulas end-to-end
through the canonical sampler.

Legacy reference (see TelescopeSim/telescope_sim/multi_aperture_psf.py:587):
- peak:           psf.flat[ref.argmax()] / ref.max()
- matched_filter: sum(psf[core] * ref[core]) / sum(ref[core]**2)
  where the core mask is sqrt(x**2 + y**2) < core_radius_rad,
  centered on the focal-grid origin.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from telescope_sim.strehl import build_strehl_estimator


@dataclass
class _StubFocalGrid:
    """Minimal stand-in for an hcipy focal grid: just ``.x`` and ``.y`` flat arrays."""

    x: np.ndarray
    y: np.ndarray


def _gaussian_ref(n: int = 33, sigma_px: float = 2.0) -> tuple[np.ndarray, _StubFocalGrid]:
    """A centered Gaussian PSF on an n×n grid; n odd so (0,0) is on a pixel."""
    half = (n - 1) // 2
    coords = np.arange(n) - half
    xx, yy = np.meshgrid(coords, coords, indexing="xy")
    psf = np.exp(-(xx**2 + yy**2) / (2 * sigma_px**2))
    grid = _StubFocalGrid(x=xx.ravel().astype(np.float64), y=yy.ravel().astype(np.float64))
    return psf, grid


# --- peak mode --------------------------------------------------------------


def test_peak_at_rest_returns_one():
    ref, _ = _gaussian_ref()
    est = build_strehl_estimator("peak", ref, focal_grid=None, core_radius_rad=None)
    assert est.compute(ref) == pytest.approx(1.0)


def test_peak_shifted_psf_drops_below_one():
    """Headline regression: a peak shifted off the reference argmax must read low.

    Builds a sharp delta-PSF and a copy with the delta shifted 4 pixels off-axis.
    Legacy peak Strehl = psf.flat[ref.argmax()] / ref.max() reads the *fixed*
    reference index, so the shifted PSF reports 0.0 there. The (now-fixed) v2
    bug used np.max(psf), which would have returned 1.0 since the bright pixel
    is still in the array — that's exactly the case the user flagged.
    """
    n = 33
    ref = np.zeros((n, n), dtype=np.float64)
    ref[n // 2, n // 2] = 1.0
    shifted = np.zeros((n, n), dtype=np.float64)
    shifted[n // 2 + 4, n // 2 + 4] = 1.0
    est = build_strehl_estimator("peak", ref, focal_grid=None, core_radius_rad=None)
    assert est.compute(ref) == pytest.approx(1.0)
    assert est.compute(shifted) == pytest.approx(0.0)
    # And: a half-amplitude in-place copy reads 0.5 (linear in the cell value).
    half = np.zeros((n, n), dtype=np.float64)
    half[n // 2, n // 2] = 0.5
    assert est.compute(half) == pytest.approx(0.5)


def test_peak_zero_reference_returns_zero():
    ref = np.zeros((8, 8), dtype=np.float64)
    est = build_strehl_estimator("peak", ref, focal_grid=None, core_radius_rad=None)
    assert est.compute(np.ones((8, 8))) == 0.0


# --- matched_filter mode ----------------------------------------------------


def test_matched_filter_at_rest_returns_one():
    ref, grid = _gaussian_ref()
    est = build_strehl_estimator("matched_filter", ref, focal_grid=grid, core_radius_rad=5.0)
    assert est.compute(ref) == pytest.approx(1.0)


def test_matched_filter_linear_in_psf_amplitude():
    """Matched filter is linear in psf: scaling psf by k scales Strehl by k."""
    ref, grid = _gaussian_ref()
    est = build_strehl_estimator("matched_filter", ref, focal_grid=grid, core_radius_rad=5.0)
    assert est.compute(0.5 * ref) == pytest.approx(0.5)


def test_matched_filter_formula_matches_legacy_by_hand():
    """Reproduce the legacy formula bit-for-bit on a known input.

    Legacy: sum(psf_core * ref_core) / sum(ref_core**2), with mask
    sqrt(x**2 + y**2) < r centered on the focal-grid origin.
    """
    ref, grid = _gaussian_ref(n=33, sigma_px=2.0)
    rng = np.random.default_rng(0)
    psf = ref + 0.05 * rng.standard_normal(ref.shape)  # noisy variant
    r = 5.0
    est = build_strehl_estimator("matched_filter", ref, focal_grid=grid, core_radius_rad=r)
    # Hand recompute the legacy formula directly:
    mask = np.sqrt(grid.x**2 + grid.y**2) < r
    ref_core = ref.ravel()[mask]
    psf_core = psf.ravel()[mask]
    expected = float((psf_core * ref_core).sum() / (ref_core**2).sum())
    assert est.compute(psf) == pytest.approx(expected, rel=1e-12)


def test_matched_filter_mask_is_origin_centered_not_argmax():
    """Mask center: legacy uses focal-grid origin, NOT argmax(ref_psf).

    Build a ref PSF whose argmax is deliberately *off* the origin. The
    legacy mask is at (0,0); v2 must match. We verify by checking that
    the cached ref_core_vals correspond to an origin-centered disk.
    """
    n = 33
    ref = np.zeros((n, n), dtype=np.float64)
    ref[n // 2 + 5, n // 2 + 5] = 1.0  # argmax is off-center
    _, grid = _gaussian_ref(n=n)
    est = build_strehl_estimator("matched_filter", ref, focal_grid=grid, core_radius_rad=2.5)
    # The argmax pixel sits at (5,5) from origin → outside r=2.5 mask.
    # So ref_core_vals must be all zero → the cached ref_core_sq_sum is 0
    # and the estimator short-circuits to 0.0.
    assert est.ref_core_sq_sum == 0.0
    assert est.compute(ref) == 0.0


def test_matched_filter_requires_positive_core_radius():
    ref, grid = _gaussian_ref()
    with pytest.raises(ValueError, match="strehl_core_rad"):
        build_strehl_estimator("matched_filter", ref, focal_grid=grid, core_radius_rad=None)
    with pytest.raises(ValueError, match="strehl_core_rad"):
        build_strehl_estimator("matched_filter", ref, focal_grid=grid, core_radius_rad=0.0)


def test_unknown_method_raises():
    ref, grid = _gaussian_ref()
    with pytest.raises(ValueError, match="unknown strehl_method"):
        build_strehl_estimator("bogus", ref, focal_grid=grid, core_radius_rad=1.0)


# --- schema-level backwards-compat -----------------------------------------


def test_schema_auto_promotes_core_rad_to_matched_filter():
    """If a legacy YAML sets strehl_core_rad but not strehl_method, infer matched_filter."""
    from telescope_sim.config.schema import SimConfig

    cfg = SimConfig.model_validate(
        {
            "pupil": {"resolution": 16, "extent": 1.0},
            "aperture": {"type": "noop"},
            "focal_planes": {"f": {"type": "angular"}},
            "outputs": {"o": {"tap": {"type": "intensity"}}},
            "strehl_core_rad": 1.5e-6,
        }
    )
    assert cfg.strehl_method == "matched_filter"


def test_schema_defaults_to_peak_when_no_core_rad():
    from telescope_sim.config.schema import SimConfig

    cfg = SimConfig.model_validate(
        {
            "pupil": {"resolution": 16, "extent": 1.0},
            "aperture": {"type": "noop"},
            "focal_planes": {"f": {"type": "angular"}},
            "outputs": {"o": {"tap": {"type": "intensity"}}},
        }
    )
    assert cfg.strehl_method == "peak"
    assert cfg.strehl_core_rad is None


def test_schema_rejects_matched_filter_without_core_rad():
    from pydantic import ValidationError

    from telescope_sim.config.schema import SimConfig

    with pytest.raises(ValidationError, match="strehl_core_rad"):
        SimConfig.model_validate(
            {
                "pupil": {"resolution": 16, "extent": 1.0},
                "aperture": {"type": "noop"},
                "focal_planes": {"f": {"type": "angular"}},
                "outputs": {"o": {"tap": {"type": "intensity"}}},
                "strehl_method": "matched_filter",
            }
        )
