"""Parity tests for ``FiberDualOutputTap`` against the legacy fiber variant.

Legacy reference (variants/fiber_rms__multi_aperture_psf.py:369-386):

    focal_total = 0
    mmf_total = 0
    for wf in wfs:
        wf_sm = actuators(wf)
        wf_foc = prop(wf_sm)
        focal_total += wf_foc.intensity
        mmf_total += self.multi_mode_fiber(wf_foc).intensity
    return np.stack([focal_total.shaped, mmf_total.shaped])

v2's :class:`FiberDualOutputTap.extract` does the equivalent: takes a
``FocalPlaneResult`` (whose ``intensity`` is already summed across
wavelengths and whose ``wavefronts`` is the per-wavelength focal
wavefront list), iterates the fiber over each wavefront, and returns
``np.stack([focal, mmf], axis=0)[..., None]`` (shape (2, H, W, 1) — the
trailing channel axis matches the legacy per-filter shape after
``out_samp[..., None]`` in ``sample()``).

Key audit points:
- StepIndexFiber construction matches legacy ``hcipy.StepIndexFiber(rcore, NA, length)``
- ``_max_in_cache`` is honored when set via the ``max_in_cache`` config field
- focal_total sums match legacy's accumulated intensity across wavelengths
- mmf_total sums match legacy's accumulated fiber-coupled intensity
- Output shape is (2, H, W, 1) matching the per-filter slice in legacy
"""

from __future__ import annotations

import hcipy
import numpy as np
import pytest

import telescope_sim.outputs.fiber_dual  # noqa: F401  (registers fiber_dual)

# LP-fiber mode solves dominate runtime (~2 min). Follow the same opt-in
# pattern as fixture 15_fiber_mmf — needs --runfiber to run on CI.
pytestmark = pytest.mark.fiber


# Fiber params — matches fixture 15 (downscaled focal grid for speed).
RCORE = 2.1e-4
NA = 0.1
FIBER_LENGTH = 7.4e-3
N_LAM = 3  # smaller than fixture 15's 7, to keep mode solves fast
LAM_CENTER = 6.35e-7
FRAC_BW = 0.001
FOCAL_RES = 32  # small for fast turnaround
FOCAL_EXTENT = 5.25e-4  # 2.5 * rcore
PUPIL_RES = 64
PUPIL_EXTENT = 3.675
APER_DIAM = 3.5


@pytest.fixture(scope="module")
def pupil_grid():
    return hcipy.make_pupil_grid(PUPIL_RES, PUPIL_EXTENT)


@pytest.fixture(scope="module")
def focal_grid():
    return hcipy.make_pupil_grid(FOCAL_RES, FOCAL_EXTENT)


@pytest.fixture(scope="module")
def propagator(pupil_grid, focal_grid):
    return hcipy.FraunhoferPropagator(pupil_grid, focal_grid, focal_length=32.5)


@pytest.fixture(scope="module")
def aper_field(pupil_grid):
    aper_callable = hcipy.make_circular_aperture(APER_DIAM)
    return hcipy.evaluate_supersampled(aper_callable, pupil_grid, 16)


@pytest.fixture(scope="module")
def focal_wavefronts(pupil_grid, propagator, aper_field):
    """Per-wavelength focal-plane wavefronts with a non-trivial defocus phase."""
    half = FRAC_BW / 2.0
    lams = LAM_CENTER * np.linspace(1.0 - half, 1.0 + half, N_LAM)
    x = np.asarray(pupil_grid.x)
    y = np.asarray(pupil_grid.y)
    r2 = (x**2 + y**2) / (APER_DIAM / 2.0) ** 2
    phase = 0.7 * r2  # mild defocus
    aberrated_field = aper_field * np.exp(1j * phase)
    wavefronts = []
    for lam in lams:
        wf = hcipy.Wavefront(aberrated_field, lam)
        wf.total_power = 1.0
        wavefronts.append(propagator(wf))
    return wavefronts


@pytest.fixture(scope="module")
def fp_results(focal_wavefronts):
    """Build a FocalPlaneResult that the v2 tap consumes."""
    from telescope_sim.focal_planes.physical import FocalPlaneResult

    intensity = np.zeros((FOCAL_RES, FOCAL_RES), dtype=np.float64)
    for wf_focal in focal_wavefronts:
        intensity += np.asarray(wf_focal.intensity.shaped)
    return {"filter1": FocalPlaneResult(intensity=intensity, wavefronts=focal_wavefronts)}


def _legacy_loop(focal_wavefronts, fiber):
    """Reproduce the legacy fiber_rms loop body line-for-line."""
    focal_total = np.zeros((FOCAL_RES, FOCAL_RES), dtype=np.float64)
    mmf_total = np.zeros((FOCAL_RES, FOCAL_RES), dtype=np.float64)
    for wf_foc in focal_wavefronts:
        focal_total += np.asarray(wf_foc.intensity.shaped)
        mmf_total += np.asarray(fiber(wf_foc).intensity.shaped)
    return np.stack([focal_total, mmf_total], axis=0)


def test_fiber_dual_output_matches_legacy_loop(focal_wavefronts, fp_results):
    """v2 extract() reproduces the legacy fiber_rms _psf() output exactly."""
    from telescope_sim.outputs.fiber_dual import FiberDualOutputTap

    # Legacy-style direct construction
    legacy_fiber = hcipy.StepIndexFiber(RCORE, NA, FIBER_LENGTH)
    legacy_fiber._max_in_cache = N_LAM
    legacy_stack = _legacy_loop(focal_wavefronts, legacy_fiber)

    # v2 wrapper
    tap = FiberDualOutputTap(
        focal_plane_name="filter1",
        fiber={
            "type": "step_index",
            "core_radius": RCORE,
            "NA": NA,
            "fiber_length": FIBER_LENGTH,
            "max_in_cache": N_LAM,
        },
    )
    v2_out = tap.extract(fp_results)

    # v2 shape is (2, H, W, 1) — legacy stack shape is (2, H, W).
    assert v2_out.shape == (2, FOCAL_RES, FOCAL_RES, 1)
    v2_stack = v2_out[..., 0]  # drop trailing channel axis for comparison
    np.testing.assert_allclose(v2_stack[0], legacy_stack[0], rtol=0, atol=1e-14)
    np.testing.assert_allclose(v2_stack[1], legacy_stack[1], rtol=1e-12, atol=1e-18)


def test_fiber_dual_max_in_cache_honored(focal_wavefronts, fp_results):
    """`max_in_cache` from config sets `_max_in_cache` on the fiber instance."""
    from telescope_sim.outputs.fiber_dual import FiberDualOutputTap

    tap = FiberDualOutputTap(
        focal_plane_name="filter1",
        fiber={
            "type": "step_index",
            "core_radius": RCORE,
            "NA": NA,
            "fiber_length": FIBER_LENGTH,
            "max_in_cache": 5,
        },
    )
    _ = tap.extract(fp_results)
    assert tap._fiber._max_in_cache == 5

    # Omitting max_in_cache leaves the HCIPy default in place
    tap2 = FiberDualOutputTap(
        focal_plane_name="filter1",
        fiber={
            "type": "step_index",
            "core_radius": RCORE,
            "NA": NA,
            "fiber_length": FIBER_LENGTH,
        },
    )
    _ = tap2.extract(fp_results)
    # HCIPy default for _max_in_cache; just confirm it's not 5 (so our override took effect above)
    assert tap2._fiber._max_in_cache != 5


def test_fiber_dual_focal_intensity_is_wavelength_sum(focal_wavefronts, fp_results):
    """The focal-channel output equals the FocalPlaneResult.intensity (wavelength-summed).

    Guards against the v2 silently rebuilding the focal intensity instead of
    consuming the pre-summed one (which would double-count or skip
    wavelengths if the iterator and result.intensity got out of sync).
    """
    from telescope_sim.outputs.fiber_dual import FiberDualOutputTap

    tap = FiberDualOutputTap(
        focal_plane_name="filter1",
        fiber={
            "type": "step_index",
            "core_radius": RCORE,
            "NA": NA,
            "fiber_length": FIBER_LENGTH,
            "max_in_cache": N_LAM,
        },
    )
    out = tap.extract(fp_results)
    focal_channel = out[0, :, :, 0]
    expected = fp_results["filter1"].intensity
    np.testing.assert_allclose(focal_channel, expected, rtol=0, atol=1e-14)


def test_fiber_dual_rejects_missing_focal_plane(fp_results):
    """A wrong focal_plane_name surfaces clearly, not silently."""
    from telescope_sim.outputs.fiber_dual import FiberDualOutputTap

    tap = FiberDualOutputTap(
        focal_plane_name="missing_filter",
        fiber={
            "type": "step_index",
            "core_radius": RCORE,
            "NA": NA,
            "fiber_length": FIBER_LENGTH,
        },
    )
    with pytest.raises(KeyError, match="missing_filter"):
        tap.extract(fp_results)


def test_fiber_dual_rejects_unknown_fiber_type(fp_results):
    from telescope_sim.outputs.fiber_dual import FiberDualOutputTap

    tap = FiberDualOutputTap(
        focal_plane_name="filter1",
        fiber={"type": "graded_index", "core_radius": 1e-4},
    )
    with pytest.raises(ValueError, match="graded_index"):
        tap.extract(fp_results)
