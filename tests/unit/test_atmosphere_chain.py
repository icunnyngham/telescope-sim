"""Tests for per-sample atmosphere support in ``sim.sample(atmos=...)``.

The atmosphere is externally owned and passed in per call — v2 holds no
atmosphere state (mirrors legacy semantics). Two pieces to verify:

1. **Reference PSF is never polluted.** The cached reference PSF (used by
   Strehl and normalization) is computed once at sim-build with no
   atmosphere. Passing an atmosphere to ``sample()`` must not change that
   cache.
2. **Fit-role correctors see atmosphere OPD.** When the atmosphere exposes
   ``.phase_for(lam)`` (HCIPy convention: phase = 2π·OPD/λ), the pipeline
   seeds ``running_opd`` with the atmospheric OPD so fit-role correctors
   with ``fit_source="cumulative_phase_pre_self"`` will naturally project
   the atmosphere into their actuator basis and cancel it.

Most tests use a minimal ``_FakeAtmos`` stand-in (any callable
``Wavefront → Wavefront`` is supported). One integration test uses the
real ``hcipy.make_zernike_basis`` to verify a representative atmospheric
Zernike mode is cancellable to machine precision.
"""

from __future__ import annotations

from pathlib import Path

import hcipy
import numpy as np
import pytest

from telescope_sim.config.loader import build_from_yaml

YAML_PATH = Path(__file__).parent / "data" / "three_zernike_residual_fit.yaml"


# --- Helpers ---------------------------------------------------------------


class _FakeAtmos:
    """Minimal atmosphere stand-in.

    Holds an OPD field in meters and behaves like ``hcipy.InfiniteAtmosphericLayer``
    enough for the pipeline's needs:

    - Callable: applies the OPD as a phase shift at the wavefront's λ.
    - ``phase_for(lam)``: returns phase = 2π·OPD/λ as a flat array, so the
      pipeline can seed ``running_opd`` for fit-role correctors.
    """

    def __init__(self, opd_field_meters):
        self._opd = np.asarray(opd_field_meters, dtype=np.float64).ravel()

    def phase_for(self, lam):
        return 2.0 * np.pi * self._opd / float(lam)

    def __call__(self, wf):
        phase = self.phase_for(wf.wavelength)
        new_field = hcipy.Field(np.asarray(wf.electric_field) * np.exp(1j * phase), wf.grid)
        return hcipy.Wavefront(new_field, wf.wavelength)


class _OpaqueAtmos:
    """An atmosphere callable WITHOUT phase_for. Tests the no-OPD-coupling path."""

    def __init__(self, opd_field_meters, lam):
        self._opd = np.asarray(opd_field_meters, dtype=np.float64).ravel()
        self._lam = float(lam)

    def __call__(self, wf):
        phase = 2.0 * np.pi * self._opd / float(wf.wavelength)
        new_field = hcipy.Field(np.asarray(wf.electric_field) * np.exp(1j * phase), wf.grid)
        return hcipy.Wavefront(new_field, wf.wavelength)


def _tilt_opd(pupil_grid, slope_radians_per_meter):
    """A simple tilt OPD across the pupil — easy to fit with low-order Zernike."""
    x = np.asarray(pupil_grid.x)
    return slope_radians_per_meter * x  # meters of OPD


def _zernike_opd(pupil_grid, noll_mode, diameter, amplitude_m):
    """A pure Zernike-mode OPD (meters), peak-normalized.

    Matches the basis the test's fit-role correctors use, so a fit-role
    Zernike corrector should cancel it to machine precision.
    """
    basis = hcipy.make_zernike_basis(noll_mode, diameter, pupil_grid, starting_mode=2)
    mode = np.asarray(basis[noll_mode - 2])
    mode_normalized = mode / np.max(np.abs(mode))
    return amplitude_m * mode_normalized


# --- Tests -----------------------------------------------------------------


def test_atmos_does_not_pollute_reference_psf():
    """The cached reference PSF is built once at sim-build with no atmosphere.

    Sampling with an atmosphere must not change ``reference_psf`` /
    ``reference_peak_intensity`` on the focal plane.
    """
    sim = build_from_yaml(YAML_PATH)
    fp = sim.focal_planes["filter1"]

    # Snapshot pre-sample
    ref_psf_before = fp.reference_psf.copy()
    ref_peak_before = fp.reference_peak_intensity
    ref_sum_before = fp.reference_psf_sum

    # Sample with a non-trivial atmosphere
    atmos = _FakeAtmos(_tilt_opd(sim._c.pupil_grid, slope_radians_per_meter=2.0e-6))
    _ = sim.sample({}, atmos=atmos)

    # Reference PSF must be unchanged (same array, exact equality)
    np.testing.assert_array_equal(np.asarray(fp.reference_psf), ref_psf_before)
    assert fp.reference_peak_intensity == ref_peak_before
    assert fp.reference_psf_sum == ref_sum_before


def test_atmos_modifies_sample_psf():
    """Passing an atmosphere measurably changes the sample PSF.

    Sanity check that atmos is being applied at all — the per-sample PSF
    with a tilted atmosphere should differ from the no-atmosphere baseline.
    """
    sim = build_from_yaml(YAML_PATH)

    out_no_atmos = sim.sample({})
    out_with_atmos = sim.sample(
        {},
        atmos=_FakeAtmos(_tilt_opd(sim._c.pupil_grid, slope_radians_per_meter=5.0e-6)),
    )

    diff = np.linalg.norm(out_no_atmos["images"]["psf"] - out_with_atmos["images"]["psf"])
    assert diff > 1e-6, "Atmosphere appears to have no effect on the sample PSF"


def test_fit_corrector_cancels_zernike_atmos():
    """A Zernike-mode atmosphere is cancelled by a matching fit-role corrector.

    Setup (from three_zernike_residual_fit.yaml): dm1, dm2 are impose-role
    Zernike correctors (set to zero here); dm3 is a fit-role Zernike
    corrector targeting ``cumulative_phase_pre_self``. With a Zernike-mode
    atmosphere on top, ``cum_opd_pre`` for dm3 contains the atmosphere OPD,
    and dm3 sets its actuators to cancel it. The corrected PSF matches the
    at-rest reference up to discrete-sampling Zernike residuals (rtol≈1e-8;
    same scale as ``test_residual_fit`` per commit 1efecd6).
    """
    sim = build_from_yaml(YAML_PATH)
    fp = sim.focal_planes["filter1"]

    # Apply a Z4 (defocus) atmosphere with amplitude 0.5 µm
    amp = 5.0e-7
    atmos_opd = _zernike_opd(sim._c.pupil_grid, noll_mode=4, diameter=1.0, amplitude_m=amp)
    atmos = _FakeAtmos(atmos_opd)

    out = sim.sample({}, atmos=atmos)
    psf_with_correction = out["images"]["psf"][..., 0]

    np.testing.assert_allclose(
        psf_with_correction,
        np.asarray(fp.reference_psf),
        rtol=1e-7,
        atol=0,
    )


def test_fit_corrector_actuators_match_atmos_opd():
    """dm3's actuator echo equals the Z-mode amplitude that defines the atmosphere.

    With Z4 (defocus) atmosphere of amplitude A, dm3's actuator value for
    Z4 (after fit + negation in pipeline step 2) should equal -A in
    actuator-amplitude units (caller-facing: divided by 2 * actuate_scale,
    per ZernikeCorrector.fit_surface). Plus the test's actuate_scale of
    1.0e-7 means a A=5e-7 m atmosphere → actuator value of -A / (2 * 1e-7)
    = -2.5 (in actuator units).
    """
    sim = build_from_yaml(YAML_PATH)

    amp = 5.0e-7
    atmos_opd = _zernike_opd(sim._c.pupil_grid, noll_mode=4, diameter=1.0, amplitude_m=amp)
    atmos = _FakeAtmos(atmos_opd)

    out = sim.sample({}, atmos=atmos)
    # target_strategy="actuators" on dm3 → echoes the (negated) fit actuator values
    actuator_echo = out["actuations"]["dm3"]

    # Z4 is the third Zernike in starting_mode=2 (Z2, Z3, Z4 → index 2).
    # Tolerance ~1e-5: discrete Zernike modes aren't exactly orthogonal on
    # the pixel grid, per commit 1efecd6's CHANGELOG note.
    expected_at_z4 = -amp / (2.0 * 1.0e-7)  # = -2.5
    assert actuator_echo[2] == pytest.approx(expected_at_z4, abs=1e-4)
    # Other modes should be near zero (atmosphere is pure Z4)
    for i, val in enumerate(actuator_echo):
        if i == 2:
            continue
        assert abs(val) < 1e-4, f"unexpected non-zero Z{i + 2} fit: {val}"


def test_atmos_without_phase_for_still_affects_wavefront():
    """An atmosphere callable lacking ``.phase_for`` still modifies the PSF,
    but fit-role correctors don't see the OPD (cum_opd_pre stays at zero).

    Documents the contract: any callable works for wavefront modification,
    but fit-role cancellation requires the HCIPy-compatible phase_for API.
    """
    sim = build_from_yaml(YAML_PATH)

    amp = 5.0e-7
    atmos_opd = _zernike_opd(sim._c.pupil_grid, noll_mode=4, diameter=1.0, amplitude_m=amp)
    # Opaque atmos: no phase_for → no fit-role coupling
    opaque = _OpaqueAtmos(atmos_opd, lam=1.0e-6)

    out_opaque = sim.sample({}, atmos=opaque)
    out_no_atmos = sim.sample({})

    # PSF differs (atmos modified the wavefront)
    diff_psf = np.linalg.norm(out_opaque["images"]["psf"] - out_no_atmos["images"]["psf"])
    assert diff_psf > 1e-6

    # But the fit corrector's actuator echo is ZERO — it didn't see the OPD
    # because phase_for wasn't available.
    np.testing.assert_allclose(
        out_opaque["actuations"]["dm3"], np.zeros_like(out_opaque["actuations"]["dm3"]), atol=1e-14
    )


def test_atmos_closed_loop_actuator_echo_drives_correction():
    """Closed-loop simulation: a previous sample's actuator echo, applied as
    the next sample's actuator values on an ``actuate`` corrector, cancels
    the atmosphere even without an in-chain fit corrector.

    This is the "Y-side residual" pattern: the fit-role corrector's echo
    tells the ML model what the actuators should have been; applying that
    same value on a regular actuate corrector closes the loop.
    """
    sim = build_from_yaml(YAML_PATH)
    fp = sim.focal_planes["filter1"]

    amp = 5.0e-7
    atmos_opd = _zernike_opd(sim._c.pupil_grid, noll_mode=4, diameter=1.0, amplitude_m=amp)
    atmos = _FakeAtmos(atmos_opd)

    # Step 1: sample with atmos and read the echo.
    out_step1 = sim.sample({}, atmos=atmos)
    fit_actuators = out_step1["actuations"]["dm3"]
    # fit_actuators is the corrector's actuators AFTER the pipeline's
    # internal negation (target_strategy="actuators" → c.actuators
    # post-set = -fit). So fit_actuators is literally "the actuator
    # values that cancel the atmosphere when applied to a DM with the
    # same basis".

    # Step 2: feed the echo back directly into dm1 (impose-role with the
    # same Zernike basis) — equivalent to the ML model commanding dm1 to
    # cancel the atmosphere. dm3 should then see cum_opd_pre ≈ 0 and fit
    # to ~zero.
    out_step2 = sim.sample({"dm1": fit_actuators}, atmos=atmos)

    # PSF should match the reference (atmos cancelled by dm1)
    np.testing.assert_allclose(
        out_step2["images"]["psf"][..., 0],
        np.asarray(fp.reference_psf),
        rtol=1e-7,
        atol=0,
    )
    # And dm3 now fits to ~zero
    np.testing.assert_allclose(
        out_step2["actuations"]["dm3"], np.zeros_like(fit_actuators), atol=1e-4
    )
