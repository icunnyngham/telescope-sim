"""Parity tests for ``PhysicalFocalPlane`` against the legacy fiber variant.

Legacy reference (variants/fiber_rms__multi_aperture_psf.py:267-269, 290):

    f_grid = hcipy.make_pupil_grid(f_res, D_focus)
    prop = hcipy.FraunhoferPropagator(self.pupil_grid, f_grid, focal_length=focal_length)
    ...
    wf.total_power = 1   # per monochromatic wavefront, hardcoded in legacy

PhysicalFocalPlane uses metres-extent focal grids (vs ``AngularFocalPlane``'s
arcsec). This audit pins the construction (grid, propagator, broadband
wavelengths) and the per-wavefront ``total_power`` normalization that the
fiber-coupling variant relies on. The chain-propagation logic itself is
covered indirectly by ``test_fiber_dual_output_tap_parity.py``.
"""

from __future__ import annotations

import hcipy
import numpy as np
import pytest


@pytest.fixture(scope="module")
def pupil_grid():
    return hcipy.make_pupil_grid(64, 3.675)


@pytest.fixture(scope="module")
def aper_field(pupil_grid):
    return hcipy.evaluate_supersampled(hcipy.make_circular_aperture(3.5), pupil_grid, 16)


def _v2_focal(pupil_grid, aper_field, *, wavefront_total_power=1.0):
    from telescope_sim.focal_planes.physical import PhysicalFocalPlane

    fp = PhysicalFocalPlane(
        central_lam=6.35e-7,
        focal_extent=5.25e-4,
        focal_res=32,
        focal_length=32.5,
        fractional_bandwidth=0.001574803,
        num_samples=3,
        wavefront_total_power=wavefront_total_power,
    )
    fp.build(pupil_grid, aper_field)
    return fp


def test_physical_focal_grid_matches_legacy(pupil_grid, aper_field):
    """focal_grid == hcipy.make_pupil_grid(focal_res, focal_extent)."""
    fp = _v2_focal(pupil_grid, aper_field)
    legacy_focal = hcipy.make_pupil_grid(32, 5.25e-4)
    np.testing.assert_allclose(
        np.asarray(fp.lam_setup.focal_grid.x), np.asarray(legacy_focal.x), rtol=0, atol=0
    )
    np.testing.assert_allclose(
        np.asarray(fp.lam_setup.focal_grid.y), np.asarray(legacy_focal.y), rtol=0, atol=0
    )


def test_physical_propagator_uses_focal_length(pupil_grid, aper_field):
    """The propagator is built with focal_length passed through to HCIPy."""
    fp = _v2_focal(pupil_grid, aper_field)
    # Cross-check: propagating a Wavefront with our propagator must match a
    # parallel direct construction with the same focal_length.
    focal_grid = hcipy.make_pupil_grid(32, 5.25e-4)
    legacy_prop = hcipy.FraunhoferPropagator(pupil_grid, focal_grid, focal_length=32.5)
    wf = hcipy.Wavefront(aper_field, 6.35e-7)
    wf.total_power = 1.0
    expected = np.asarray(legacy_prop(wf).intensity.shaped)
    actual = np.asarray(fp.lam_setup.propagator(wf).intensity.shaped)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-30)


def test_physical_broadband_lams_match_canonical_formula(pupil_grid, aper_field):
    """filter_lams = central * linspace(1 - h/2, 1 + h/2, N) — same as AngularFocalPlane."""
    fp = _v2_focal(pupil_grid, aper_field)
    h = 0.001574803
    expected_lams = 6.35e-7 * np.linspace(1.0 - h / 2.0, 1.0 + h / 2.0, 3)
    np.testing.assert_allclose(fp.lam_setup.filter_lams, expected_lams, rtol=0, atol=0)


def test_physical_wavefront_total_power_honored(pupil_grid, aper_field):
    """`wavefront_total_power=1.0` forces all per-wavelength wavefronts to total_power=1.

    The legacy fiber variant hardcodes ``wf.total_power = 1`` in the
    per-filter setup loop. v2 makes this configurable; fixture 15 sets it
    explicitly to 1.0.
    """
    fp = _v2_focal(pupil_grid, aper_field, wavefront_total_power=1.0)
    for i, wf in enumerate(fp.lam_setup.wavefronts):
        assert float(wf.total_power) == pytest.approx(1.0, abs=1e-12), (
            f"wavefront {i} (lam={fp.lam_setup.filter_lams[i]:.3e}) has "
            f"total_power={wf.total_power}, expected 1.0"
        )


def test_physical_wavefront_total_power_default_is_unset(pupil_grid, aper_field):
    """When `wavefront_total_power=None`, the HCIPy default is left in place."""
    fp = _v2_focal(pupil_grid, aper_field, wavefront_total_power=None)
    # Different from 1.0 — confirms the v2 code path didn't fire
    powers = [float(wf.total_power) for wf in fp.lam_setup.wavefronts]
    assert not all(p == pytest.approx(1.0) for p in powers), (
        "Expected the HCIPy-default total_power (NOT 1.0) when wavefront_total_power=None"
    )


def test_physical_single_wavelength_when_num_samples_one(pupil_grid, aper_field):
    """num_samples=1 produces a length-1 wavelengths array at central_lam."""
    from telescope_sim.focal_planes.physical import PhysicalFocalPlane

    fp = PhysicalFocalPlane(
        central_lam=6.35e-7,
        focal_extent=5.25e-4,
        focal_res=32,
        focal_length=32.5,
        num_samples=1,
    )
    fp.build(pupil_grid, aper_field)
    np.testing.assert_array_equal(fp.lam_setup.filter_lams, np.array([6.35e-7]))
    assert len(fp.lam_setup.wavefronts) == 1


def test_physical_reference_psf_at_rest_caches_peak_and_sum(pupil_grid, aper_field):
    """compute_reference_psf() runs the chain at rest and caches peak + sum."""
    fp = _v2_focal(pupil_grid, aper_field)
    fp.compute_reference_psf([])  # no correctors
    assert fp.reference_psf is not None
    assert fp.reference_peak_intensity is not None
    assert fp.reference_psf_sum is not None
    # peak == max of psf
    assert fp.reference_peak_intensity == pytest.approx(fp.reference_psf.max())
    # sum == sum of psf
    assert fp.reference_psf_sum == pytest.approx(fp.reference_psf.sum())


def test_physical_compute_reference_psf_before_build_raises(pupil_grid):
    from telescope_sim.focal_planes.physical import PhysicalFocalPlane

    fp = PhysicalFocalPlane(
        central_lam=6.35e-7,
        focal_extent=5.25e-4,
        focal_res=32,
        focal_length=32.5,
    )
    with pytest.raises(RuntimeError, match="build"):
        fp.compute_reference_psf([])


def test_physical_propagate_chain_matches_direct_loop(pupil_grid, aper_field):
    """`_propagate_chain` per-wavelength loop matches the legacy fiber variant.

    The legacy loop (variants/fiber_rms__multi_aperture_psf.py:372-385):

        focal_total = 0
        for wf in wfs:
            wf_sm = actuators(wf)        # no correctors here
            wf_foc = prop(wf_sm)
            focal_total += wf_foc.intensity

    v2 with an empty corrector chain should produce the same intensity.
    """
    fp = _v2_focal(pupil_grid, aper_field)
    result = fp._propagate_chain([], coronagraph=None)

    # Direct reproduction
    expected_intensity = np.zeros((32, 32), dtype=np.float64)
    for wf in fp.lam_setup.wavefronts:
        wf_foc = fp.lam_setup.propagator(wf)
        expected_intensity += np.asarray(wf_foc.intensity.shaped)

    np.testing.assert_allclose(result.intensity, expected_intensity, rtol=0, atol=1e-30)
    assert len(result.wavefronts) == 3
