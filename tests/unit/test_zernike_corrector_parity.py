"""Parity tests for ``ZernikeCorrector`` against the legacy DM construction.

Legacy reference (variants/coro__coro_mas_psf.py:144-148, 328):

    self.dm_basis = hcipy.make_zernike_basis(num_zern_modes, self.diameter,
                                              self.pupil_grid, starting_mode=2)
    self.dm_basis = hcipy.ModeBasis([b / np.max(np.abs(b)) for b in self.dm_basis])
    self.dm = hcipy.DeformableMirror(self.dm_basis)
    ...
    self.dm.actuators = actuate * self.actuate_scale

Commit 33914ee already fixed two latent bugs in this corrector (the missing
/2 OPD-vs-surface factor in fit_surface and the non-orthogonal diagonal
projection). This audit closes the loop by pinning the remaining axes:

  - Zernike basis construction matches legacy bit-for-bit
  - The peak-normalization `b / np.max(np.abs(b))` is applied per-mode
  - Caller-facing actuator values round-trip through `set_actuators` ↔
    `actuators` property (the multiplication by `actuate_scale` doesn't
    leak into the public getter)
  - `apply(wf)` matches a direct `hcipy.DeformableMirror(basis)(wf)` call
  - `starting_mode` override is honored (fixture 15 uses starting_mode=1)

Fit-surface behavior (the `/2` factor + lstsq + constant-offset immunity)
is comprehensively covered by tests/unit/test_residual_fit.py — not
duplicated here.
"""

from __future__ import annotations

import hcipy
import numpy as np
import pytest


@pytest.fixture(scope="module")
def pupil_grid():
    return hcipy.make_pupil_grid(64, 1.05)


@pytest.fixture(scope="module")
def aper_field(pupil_grid):
    aper_callable = hcipy.make_circular_aperture(1.0)
    return hcipy.evaluate_supersampled(aper_callable, pupil_grid, 16)


def _legacy_basis(pupil_grid, n_modes, diameter, starting_mode=2):
    """Reproduce the legacy basis construction exactly."""
    basis = hcipy.make_zernike_basis(n_modes, diameter, pupil_grid, starting_mode=starting_mode)
    return hcipy.ModeBasis([b / np.max(np.abs(b)) for b in basis])


def _v2_corrector(pupil_grid, aper_field, *, n_modes=4, starting_mode=2, actuate_scale=1.0e-7):
    from telescope_sim.correctors.zernike import ZernikeCorrector

    c = ZernikeCorrector(
        n_modes=n_modes,
        zernike_diameter=1.0,
        starting_mode=starting_mode,
        actuate_scale=actuate_scale,
    )
    c._bind_pupil_grid(pupil_grid, aper_field)
    return c


def test_zernike_basis_matches_legacy_bit_for_bit(pupil_grid, aper_field):
    """The cached basis modes match the legacy peak-normalized basis."""
    legacy = _legacy_basis(pupil_grid, n_modes=4, diameter=1.0, starting_mode=2)
    c = _v2_corrector(pupil_grid, aper_field, n_modes=4, starting_mode=2)

    assert c._basis is not None
    assert len(c._basis) == 4
    for i, (legacy_mode, v2_mode) in enumerate(zip(legacy, c._basis, strict=True)):
        np.testing.assert_allclose(
            np.asarray(v2_mode),
            np.asarray(legacy_mode),
            rtol=0,
            atol=0,
            err_msg=f"mode {i} (Noll {i + 2}) diverges from peak-normalized legacy basis",
        )


def test_zernike_peak_normalization_per_mode(pupil_grid, aper_field):
    """Each cached basis mode peaks at exactly 1.0 in absolute value."""
    c = _v2_corrector(pupil_grid, aper_field, n_modes=6, starting_mode=2)
    for i, mode in enumerate(c._basis):
        peak = float(np.max(np.abs(np.asarray(mode))))
        assert peak == pytest.approx(1.0, abs=1e-15), (
            f"mode {i} (Noll {i + 2}) has peak {peak}, expected exactly 1.0 from "
            "`b / np.max(np.abs(b))` normalization"
        )


def test_zernike_set_actuators_writes_scaled_values(pupil_grid, aper_field):
    """`set_actuators(v)` writes `v * actuate_scale` to the underlying HCIPy DM."""
    c = _v2_corrector(pupil_grid, aper_field, n_modes=3, actuate_scale=1.0e-7)
    v = np.array([0.5, -0.3, 0.1])
    c.set_actuators(v)
    np.testing.assert_allclose(np.asarray(c._dm.actuators), v * 1.0e-7, rtol=0, atol=1e-25)


def test_zernike_actuators_getter_returns_caller_values(pupil_grid, aper_field):
    """`.actuators` returns caller-facing values, NOT the internal scaled ones."""
    c = _v2_corrector(pupil_grid, aper_field, n_modes=3, actuate_scale=1.0e-7)
    v = np.array([0.5, -0.3, 0.1])
    c.set_actuators(v)
    np.testing.assert_allclose(c.actuators, v, rtol=0, atol=1e-15)


def test_zernike_flatten_zeros_the_dm(pupil_grid, aper_field):
    c = _v2_corrector(pupil_grid, aper_field, n_modes=3)
    c.set_actuators(np.array([0.5, -0.3, 0.1]))
    c.flatten()
    np.testing.assert_array_equal(np.asarray(c._dm.actuators), np.zeros(3))
    np.testing.assert_array_equal(c.actuators, np.zeros(3))


def test_zernike_apply_matches_direct_hcipy_dm(pupil_grid, aper_field):
    """v2 apply() == direct hcipy.DeformableMirror(basis)(wf) at the same actuators."""
    legacy_basis = _legacy_basis(pupil_grid, n_modes=4, diameter=1.0, starting_mode=2)
    legacy_dm = hcipy.DeformableMirror(legacy_basis)
    actuate_scale = 1.0e-7
    actuators = np.array([0.5, -0.3, 0.1, 0.2])
    legacy_dm.actuators = actuators * actuate_scale

    c = _v2_corrector(pupil_grid, aper_field, n_modes=4, actuate_scale=actuate_scale)
    c.set_actuators(actuators)

    wf = hcipy.Wavefront(aper_field, 1.0e-6)
    legacy_out = legacy_dm(wf)
    v2_out = c.apply(wf)

    np.testing.assert_allclose(
        np.asarray(v2_out.electric_field),
        np.asarray(legacy_out.electric_field),
        rtol=0,
        atol=1e-15,
    )


def test_zernike_starting_mode_honored(pupil_grid, aper_field):
    """`starting_mode=1` (used by fixture 15_fiber_mmf) actually starts at Noll 1.

    Noll 1 is piston — after peak-normalization a binary mask (1.0 inside the
    Zernike disk, 0.0 outside), so it has exactly two distinct values. Noll 2
    is tip — a linear gradient, so many distinct values. We assert both
    cross-checks against a direct HCIPy call.
    """
    c1 = _v2_corrector(pupil_grid, aper_field, n_modes=2, starting_mode=1)
    c2 = _v2_corrector(pupil_grid, aper_field, n_modes=2, starting_mode=2)

    # c1's first mode = legacy basis built with starting_mode=1, first element
    legacy_sm1 = _legacy_basis(pupil_grid, n_modes=2, diameter=1.0, starting_mode=1)
    legacy_sm2 = _legacy_basis(pupil_grid, n_modes=2, diameter=1.0, starting_mode=2)

    np.testing.assert_allclose(np.asarray(c1._basis[0]), np.asarray(legacy_sm1[0]), rtol=0, atol=0)
    np.testing.assert_allclose(np.asarray(c2._basis[0]), np.asarray(legacy_sm2[0]), rtol=0, atol=0)

    # And concretely: starting_mode=1 first mode is binary (piston),
    # starting_mode=2 first mode is a continuous gradient (tip).
    mode_piston = np.asarray(c1._basis[0])
    mode_tip = np.asarray(c2._basis[0])
    assert len(np.unique(np.round(mode_piston, 6))) == 2  # piston: only 0 and 1
    assert len(np.unique(np.round(mode_tip, 6))) > 50  # tip: many distinct values


def test_zernike_set_actuators_wrong_length_raises(pupil_grid, aper_field):
    c = _v2_corrector(pupil_grid, aper_field, n_modes=3)
    with pytest.raises(ValueError, match="3 actuators"):
        c.set_actuators(np.zeros(4))


def test_zernike_unbound_apply_raises():
    from telescope_sim.correctors.zernike import ZernikeCorrector

    c = ZernikeCorrector(n_modes=3, zernike_diameter=1.0)
    with pytest.raises(RuntimeError, match="_bind_pupil_grid"):
        c.apply(None)


def test_zernike_unbound_set_actuators_raises():
    from telescope_sim.correctors.zernike import ZernikeCorrector

    c = ZernikeCorrector(n_modes=3, zernike_diameter=1.0)
    with pytest.raises(RuntimeError, match="_bind_pupil_grid"):
        c.set_actuators(np.zeros(3))
