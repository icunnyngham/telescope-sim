"""Parity tests for ``AngularFocalPlane`` against the canonical legacy construction.

Legacy reference (TelescopeSim/telescope_sim/multi_aperture_psf.py:249-276):

    fov = f_config['focal_extent']
    f_grid = hcipy.make_uniform_grid([f_res]*2, fov*np.pi/(180*3600))
    prop = hcipy.FraunhoferPropagator(self.pupil_grid, f_grid)
    ...
    if num_samples > 1:
        filter_lams = lam * np.linspace(1 - frac_bw/2., 1 + frac_bw/2., num_samples)
    else:
        filter_lams = [lam]
    wfs = [hcipy.Wavefront(self.aper, fil_lam) for fil_lam in filter_lams]

The audit pins each axis of this construction:

  - arcsec → radians conversion is the canonical `* np.pi / (180 * 3600)`
  - focal grid is built via ``hcipy.make_uniform_grid`` (NOT
    ``make_pupil_grid`` — that's the physical-focal-plane variant)
  - the propagator takes NO ``focal_length`` (angular-extent path)
  - broadband sampling formula matches the canonical
    ``central * linspace(1 - h/2, 1 + h/2, N)``
  - num_samples=1 takes the shortcut path (legacy used a Python list)
  - per-wavelength wavefronts are constructed via ``hcipy.Wavefront(aper, lam)``
    and DO NOT set ``total_power`` (vs PhysicalFocalPlane, which optionally does)

The sample()-time per-wavelength chain loop is shared with PhysicalFocalPlane
and is exercised at the fixture level by the existing canonical fixtures
1-3, 10, 11, 16.
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
    return hcipy.evaluate_supersampled(hcipy.make_circular_aperture(1.0), pupil_grid, 16)


def _v2_angular(pupil_grid, aper_field, *, num_samples=3, frac_bw=0.05):
    from telescope_sim.focal_planes.angular import AngularFocalPlane

    fp = AngularFocalPlane(
        central_lam=1.0e-6,
        focal_extent=0.5,  # arcsec
        focal_res=32,
        fractional_bandwidth=frac_bw,
        num_samples=num_samples,
    )
    fp.build(pupil_grid, aper_field)
    return fp


def test_angular_focal_grid_uses_arcsec_to_radians(pupil_grid, aper_field):
    """fov is converted from arcsec to radians via `* np.pi / (180 * 3600)`."""
    fp = _v2_angular(pupil_grid, aper_field, num_samples=1)
    legacy_fov_rad = 0.5 * np.pi / (180.0 * 3600.0)
    legacy_focal = hcipy.make_uniform_grid([32, 32], legacy_fov_rad)
    np.testing.assert_allclose(
        np.asarray(fp.lam_setup.focal_grid.x),
        np.asarray(legacy_focal.x),
        rtol=0,
        atol=0,
    )
    np.testing.assert_allclose(
        np.asarray(fp.lam_setup.focal_grid.y),
        np.asarray(legacy_focal.y),
        rtol=0,
        atol=0,
    )


def test_angular_focal_grid_uses_make_uniform_grid(pupil_grid, aper_field):
    """The focal grid is constructed via ``hcipy.make_uniform_grid``.

    Legacy literally calls ``hcipy.make_uniform_grid([f_res]*2, fov_rad)``.
    For even-N grids the result happens to coincide with ``make_pupil_grid``,
    but the API contract is to use the uniform builder — distinct from the
    half-pixel-shifted pupil grid in some HCIPy versions and code paths.
    """
    fp = _v2_angular(pupil_grid, aper_field, num_samples=1)
    fov_rad = 0.5 * np.pi / (180.0 * 3600.0)
    uniform = hcipy.make_uniform_grid([32, 32], fov_rad)
    np.testing.assert_allclose(
        np.asarray(fp.lam_setup.focal_grid.x), np.asarray(uniform.x), rtol=0, atol=0
    )
    np.testing.assert_allclose(
        np.asarray(fp.lam_setup.focal_grid.y), np.asarray(uniform.y), rtol=0, atol=0
    )


def test_angular_propagator_omits_focal_length(pupil_grid, aper_field):
    """The propagator is constructed WITHOUT focal_length (angular variant)."""
    fp = _v2_angular(pupil_grid, aper_field, num_samples=1)
    fov_rad = 0.5 * np.pi / (180.0 * 3600.0)
    focal_grid = hcipy.make_uniform_grid([32, 32], fov_rad)
    legacy_prop = hcipy.FraunhoferPropagator(pupil_grid, focal_grid)  # no focal_length
    wf = hcipy.Wavefront(aper_field, 1.0e-6)
    expected = np.asarray(legacy_prop(wf).intensity.shaped)
    actual = np.asarray(fp.lam_setup.propagator(wf).intensity.shaped)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-30)


def test_angular_broadband_lams_match_canonical_formula(pupil_grid, aper_field):
    """filter_lams = central * linspace(1 - h/2, 1 + h/2, N)."""
    fp = _v2_angular(pupil_grid, aper_field, num_samples=5, frac_bw=0.05)
    h = 0.05
    expected_lams = 1.0e-6 * np.linspace(1.0 - h / 2.0, 1.0 + h / 2.0, 5)
    np.testing.assert_allclose(fp.lam_setup.filter_lams, expected_lams, rtol=0, atol=0)


def test_angular_single_wavelength_when_num_samples_one(pupil_grid, aper_field):
    """num_samples=1 produces a length-1 lams array (legacy used [lam])."""
    fp = _v2_angular(pupil_grid, aper_field, num_samples=1)
    np.testing.assert_array_equal(fp.lam_setup.filter_lams, np.array([1.0e-6]))
    assert len(fp.lam_setup.wavefronts) == 1


def test_angular_wavefronts_dont_override_total_power(pupil_grid, aper_field):
    """The angular focal plane has no `wavefront_total_power` field — legacy default applies.

    Distinct from PhysicalFocalPlane, which optionally normalizes
    total_power=1.0 for the fiber-coupling lineage.
    """
    fp = _v2_angular(pupil_grid, aper_field, num_samples=1)
    wf = fp.lam_setup.wavefronts[0]
    # Direct comparison: build a Wavefront the legacy way and confirm same
    # total_power
    legacy_wf = hcipy.Wavefront(aper_field, 1.0e-6)
    assert float(wf.total_power) == pytest.approx(float(legacy_wf.total_power), rel=1e-15)


def test_angular_propagate_chain_loop_matches_legacy(pupil_grid, aper_field):
    """`_propagate_chain` per-wavelength loop matches the legacy `_psf` body.

    Legacy (TelescopeSim/.../multi_aperture_psf.py:357-368):

        focal_total = 0
        for wf in wfs:
            wf_sm = actuators(wf)             # no correctors here
            focal_total += prop(wf_sm).intensity
    """
    fp = _v2_angular(pupil_grid, aper_field, num_samples=3, frac_bw=0.05)
    result = fp._propagate_chain([], coronagraph=None)

    expected_intensity = np.zeros((32, 32), dtype=np.float64)
    for wf in fp.lam_setup.wavefronts:
        wf_foc = fp.lam_setup.propagator(wf)
        expected_intensity += np.asarray(wf_foc.intensity.shaped)

    np.testing.assert_allclose(result.intensity, expected_intensity, rtol=0, atol=1e-30)
    assert len(result.wavefronts) == 3


def test_angular_reference_psf_caches_peak_and_sum(pupil_grid, aper_field):
    fp = _v2_angular(pupil_grid, aper_field, num_samples=1)
    fp.compute_reference_psf([])
    assert fp.reference_peak_intensity == pytest.approx(fp.reference_psf.max())
    assert fp.reference_psf_sum == pytest.approx(fp.reference_psf.sum())


def test_angular_compute_reference_psf_before_build_raises(pupil_grid):
    from telescope_sim.focal_planes.angular import AngularFocalPlane

    fp = AngularFocalPlane(central_lam=1.0e-6, focal_extent=0.5, focal_res=32)
    with pytest.raises(RuntimeError, match="build"):
        fp.compute_reference_psf([])
