"""Parity + behavior tests for ``NoisyDetectorPostProcessor``.

Migrated in v2.0.0a9 from the v2.0.0a7 ``NoisyIntensityOutputTap`` tests.
Same 17 assertions; the post-processor is constructed and bound directly
(no full sim build) so the tests stay tight and fast.

Layered for a stochastic component:

1. **Structural** (RNG-free): shape, kwarg honoring, per-sample override
   beats YAML default, detector built once (eager bind), single-focal-plane
   restriction, clamp_nonnegative flag.
2. **Noise-off identity**: all noise sources zeroed → exact equality with
   the manually-constructed power_field * dt = flux*area expectation.
   Decouples wiring from RNG.
3. **Statistical** (seeded, N=96): per-noise-source assertions on
   mean/std match analytical predictions within ~3σ/√N.
4. **Determinism**: same ``np.random.seed`` → bit-for-bit identical
   outputs; different seeds diverge.

The gold-standard legacy parity check lives at the fixture level
(``fixtures/runner/digests/17_noisy_psf/expected.json``).

RNG policy: HCIPy's NoisyDetector uses ``np.random.*`` globally. Tests
seed just before each call.
"""

from __future__ import annotations

import hcipy
import numpy as np
import pytest

import telescope_sim.post  # noqa: F401 — registers all post-processors
from telescope_sim.abc import PipelineContext

# --- Shared fixtures --------------------------------------------------------


N_LAM = 3
FOCAL_RES = 32
PUPIL_RES = 64
APER_DIAM = 1.0
PUPIL_EXTENT = 1.05
APER_AREA = float(np.pi * (APER_DIAM / 2) ** 2)
INT_PHOT_FLUX = 1.0e6  # photons/m^2 — flux*area ≈ 3e6 photons total


@pytest.fixture(scope="module")
def pupil_grid():
    return hcipy.make_pupil_grid(PUPIL_RES, PUPIL_EXTENT)


@pytest.fixture(scope="module")
def focal_grid():
    fov_rad = 0.5 * np.pi / (180.0 * 3600.0)
    return hcipy.make_uniform_grid([FOCAL_RES, FOCAL_RES], fov_rad)


@pytest.fixture(scope="module")
def aper_field(pupil_grid):
    return hcipy.evaluate_supersampled(hcipy.make_circular_aperture(APER_DIAM), pupil_grid, 16)


@pytest.fixture(scope="module")
def clean_psf_image(pupil_grid, focal_grid, aper_field):
    """A clean (H, W, 1) PSF image — what an IntensityOutputTap would produce."""
    prop = hcipy.FraunhoferPropagator(pupil_grid, focal_grid)
    lams = 1.0e-6 * np.linspace(0.99, 1.01, N_LAM)
    intensity = np.zeros((FOCAL_RES, FOCAL_RES), dtype=np.float64)
    for lam in lams:
        wf = hcipy.Wavefront(aper_field, lam)
        intensity += np.asarray(prop(wf).intensity.shaped)
    return intensity[..., None]


class _FakeFocalPlane:
    """Stand-in focal plane for _bind_loader_dependencies — only exposes the grid."""

    def __init__(self, focal_grid):
        self.lam_setup = type("_S", (), {"focal_grid": focal_grid})()


class _FakeApertureResult:
    def __init__(self, area):
        self.area = area


def _bind(pp, focal_grid):
    pp._bind_loader_dependencies(
        aperture_result=_FakeApertureResult(APER_AREA),
        focal_planes={"f1": _FakeFocalPlane(focal_grid)},
        focal_plane_names=["f1"],
    )


def _make_pp(*, int_phot_flux=INT_PHOT_FLUX, detector=None, clamp_nonnegative=True):
    from telescope_sim.post.noisy_detector import NoisyDetectorPostProcessor

    return NoisyDetectorPostProcessor(
        int_phot_flux=int_phot_flux,
        detector=detector or {},
        clamp_nonnegative=clamp_nonnegative,
    )


def _ctx(overrides=None):
    return PipelineContext(
        output_name="scene",
        focal_plane_name="f1",
        reference_peak_intensity=None,
        reference_psf_sum=None,
        overrides=overrides or {},
    )


# === Layer 1: structural (no RNG) ==========================================


def test_noisy_pp_output_shape_channels_last(clean_psf_image, focal_grid):
    pp = _make_pp(detector={"include_photon_noise": False})
    _bind(pp, focal_grid)
    out = pp(clean_psf_image, _ctx())
    assert out.shape == (FOCAL_RES, FOCAL_RES, 1)
    assert out.dtype == np.float64


def test_noisy_pp_rejects_multi_focal_plane(focal_grid):
    pp = _make_pp(int_phot_flux=None)
    with pytest.raises(ValueError, match="exactly one focal_plane"):
        pp._bind_loader_dependencies(
            aperture_result=_FakeApertureResult(APER_AREA),
            focal_planes={"a": _FakeFocalPlane(focal_grid), "b": _FakeFocalPlane(focal_grid)},
            focal_plane_names=["a", "b"],
        )


def test_noisy_pp_unbound_raises(clean_psf_image):
    pp = _make_pp(int_phot_flux=None)
    with pytest.raises(RuntimeError, match="_bind_loader_dependencies"):
        pp(clean_psf_image, _ctx())


def test_noisy_pp_per_sample_override_beats_yaml_default(clean_psf_image, focal_grid):
    """`overrides={"int_phot_flux": X}` overrides the constructor default."""
    np.random.seed(0)
    pp_low = _make_pp(int_phot_flux=1e4, detector={"include_photon_noise": False})
    _bind(pp_low, focal_grid)
    pp_high = _make_pp(int_phot_flux=1e8, detector={"include_photon_noise": False})
    _bind(pp_high, focal_grid)

    out_low = pp_low(clean_psf_image, _ctx())
    out_high = pp_high(clean_psf_image, _ctx())
    assert out_high.sum() > out_low.sum() * 100

    pp = _make_pp(int_phot_flux=1e4, detector={"include_photon_noise": False})
    _bind(pp, focal_grid)
    out_default = pp(clean_psf_image, _ctx())
    out_overridden = pp(clean_psf_image, _ctx(overrides={"int_phot_flux": 1e8}))
    assert out_overridden.sum() > out_default.sum() * 100


def test_noisy_pp_override_only_affects_known_keys(clean_psf_image, focal_grid):
    """Unknown override keys are tolerated (forward compat)."""
    pp = _make_pp(int_phot_flux=INT_PHOT_FLUX, detector={"include_photon_noise": False})
    _bind(pp, focal_grid)
    np.random.seed(0)
    out_baseline = pp(clean_psf_image, _ctx())
    np.random.seed(0)
    out_ignored = pp(clean_psf_image, _ctx(overrides={"unknown_future_kwarg": 42}))
    np.testing.assert_array_equal(out_baseline, out_ignored)


def test_noisy_pp_rejects_wrong_input_shape(focal_grid):
    pp = _make_pp(detector={"include_photon_noise": False})
    _bind(pp, focal_grid)
    bad = np.zeros((FOCAL_RES, FOCAL_RES, 3), dtype=np.float64)  # multi-channel
    with pytest.raises(ValueError, match="single-channel"):
        pp(bad, _ctx())


def test_noisy_pp_detector_built_once_at_bind(focal_grid, clean_psf_image):
    """Eager construction at _bind_loader_dependencies time, not lazily."""
    pp = _make_pp(detector={"include_photon_noise": False})
    assert pp._detector is None
    _bind(pp, focal_grid)
    first_detector = pp._detector
    assert first_detector is not None
    # Subsequent calls reuse the same instance
    np.random.seed(0)
    _ = pp(clean_psf_image, _ctx())
    np.random.seed(0)
    _ = pp(clean_psf_image, _ctx())
    assert pp._detector is first_detector


def test_noisy_pp_clamp_nonnegative_flag(clean_psf_image, focal_grid):
    """clamp_nonnegative=True applies np.abs(); =False leaves negatives in."""
    np.random.seed(0)
    pp_clamp = _make_pp(
        int_phot_flux=1e2,
        detector={"include_photon_noise": False, "read_noise": 100.0},
        clamp_nonnegative=True,
    )
    _bind(pp_clamp, focal_grid)
    out_clamp = pp_clamp(clean_psf_image, _ctx())
    assert (out_clamp >= 0).all()

    np.random.seed(0)
    pp_noclamp = _make_pp(
        int_phot_flux=1e2,
        detector={"include_photon_noise": False, "read_noise": 100.0},
        clamp_nonnegative=False,
    )
    _bind(pp_noclamp, focal_grid)
    out_noclamp = pp_noclamp(clean_psf_image, _ctx())
    assert (out_noclamp < 0).any()


# === Layer 2: noise-off identity ===========================================


def test_noisy_pp_noise_off_matches_manual_power_field(clean_psf_image, focal_grid):
    """All noise sources disabled → output equals the manually-built expectation."""
    pp = _make_pp(
        int_phot_flux=INT_PHOT_FLUX,
        detector={
            "include_photon_noise": False,
            "read_noise": 0.0,
            "dark_current_rate": 0.0,
            "flat_field": 0.0,
        },
    )
    _bind(pp, focal_grid)
    out = pp(clean_psf_image, _ctx())[..., 0]

    weights_arr = np.asarray(focal_grid.weights)
    power_field = clean_psf_image[..., 0].ravel() * weights_arr
    target = INT_PHOT_FLUX * APER_AREA
    natural_total = power_field.sum()
    expected = (power_field * (target / natural_total)).reshape(FOCAL_RES, FOCAL_RES)

    np.testing.assert_allclose(out, expected, rtol=1e-12, atol=1e-30)


def test_noisy_pp_noise_off_total_charge_matches_flux(clean_psf_image, focal_grid):
    """Sum of accumulated charge equals flux * aper_area (legacy contract)."""
    pp = _make_pp(
        int_phot_flux=INT_PHOT_FLUX,
        detector={
            "include_photon_noise": False,
            "read_noise": 0.0,
            "dark_current_rate": 0.0,
            "flat_field": 0.0,
        },
    )
    _bind(pp, focal_grid)
    out = pp(clean_psf_image, _ctx())
    s = out.sum()
    assert 0.5 * INT_PHOT_FLUX * APER_AREA < s < 1.1 * INT_PHOT_FLUX * APER_AREA


def test_noisy_pp_flux_none_uses_natural_power_field(clean_psf_image, focal_grid):
    """int_phot_flux=None → no rescaling; output equals input * grid.weights."""
    pp = _make_pp(
        int_phot_flux=None,
        detector={
            "include_photon_noise": False,
            "read_noise": 0.0,
            "dark_current_rate": 0.0,
            "flat_field": 0.0,
        },
    )
    _bind(pp, focal_grid)
    out = pp(clean_psf_image, _ctx())[..., 0]
    weights_arr = np.asarray(focal_grid.weights)
    expected = (clean_psf_image[..., 0].ravel() * weights_arr).reshape(FOCAL_RES, FOCAL_RES)
    np.testing.assert_allclose(out, expected, rtol=1e-12, atol=1e-30)


# === Layer 3: statistical assertions =======================================


def _samples_under_seed(make_pp_with_bind, image, n=64, seed=0):
    np.random.seed(seed)
    samples = np.stack([make_pp_with_bind()(image, _ctx())[..., 0] for _ in range(n)], axis=0)
    return samples


def test_noisy_pp_read_noise_alone_mean_unchanged_std_matches(clean_psf_image, focal_grid):
    """Pure read noise: mean ≈ clean, std ≈ read_noise per pixel."""
    READ_NOISE = 5.0
    N = 96

    def make():
        pp = _make_pp(
            int_phot_flux=INT_PHOT_FLUX,
            detector={
                "include_photon_noise": False,
                "read_noise": READ_NOISE,
                "dark_current_rate": 0.0,
                "flat_field": 0.0,
            },
            clamp_nonnegative=False,
        )
        _bind(pp, focal_grid)
        return pp

    samples = _samples_under_seed(make, clean_psf_image, n=N)
    sample_mean = samples.mean(axis=0)
    sample_std = samples.std(axis=0)

    pp_clean = _make_pp(
        int_phot_flux=INT_PHOT_FLUX,
        detector={
            "include_photon_noise": False,
            "read_noise": 0.0,
            "dark_current_rate": 0.0,
            "flat_field": 0.0,
        },
        clamp_nonnegative=False,
    )
    _bind(pp_clean, focal_grid)
    clean = pp_clean(clean_psf_image, _ctx())[..., 0]

    se = READ_NOISE / np.sqrt(N)
    assert np.mean(np.abs(sample_mean - clean)) < 4 * se
    assert np.abs(np.mean(sample_std) - READ_NOISE) < 0.5


def test_noisy_pp_dark_current_alone_mean_shifts_by_rate(clean_psf_image, focal_grid):
    """Dark current alone: per-pixel mean shifts by `rate * dt`."""
    DARK = 50.0
    N = 96

    def make():
        pp = _make_pp(
            int_phot_flux=INT_PHOT_FLUX,
            detector={
                "include_photon_noise": False,
                "read_noise": 0.0,
                "dark_current_rate": DARK,
                "flat_field": 0.0,
            },
            clamp_nonnegative=False,
        )
        _bind(pp, focal_grid)
        return pp

    samples = _samples_under_seed(make, clean_psf_image, n=N)

    pp_clean = _make_pp(
        int_phot_flux=INT_PHOT_FLUX,
        detector={
            "include_photon_noise": False,
            "read_noise": 0.0,
            "dark_current_rate": 0.0,
            "flat_field": 0.0,
        },
        clamp_nonnegative=False,
    )
    _bind(pp_clean, focal_grid)
    clean = pp_clean(clean_psf_image, _ctx())[..., 0]

    diff = samples.mean(axis=0) - clean
    assert np.abs(diff - DARK).max() < 1e-9


def test_noisy_pp_photon_noise_variance_equals_mean(clean_psf_image, focal_grid):
    """Pure photon noise: var(pixel) ≈ mean(pixel) (Poisson)."""
    N = 96

    def make():
        pp = _make_pp(
            int_phot_flux=INT_PHOT_FLUX,
            detector={
                "include_photon_noise": True,
                "read_noise": 0.0,
                "dark_current_rate": 0.0,
                "flat_field": 0.0,
            },
            clamp_nonnegative=False,
        )
        _bind(pp, focal_grid)
        return pp

    samples = _samples_under_seed(make, clean_psf_image, n=N)
    means = samples.mean(axis=0)
    variances = samples.var(axis=0)
    bright = means > 10.0
    if not bright.any():
        pytest.skip("no pixels bright enough to test Poisson var=mean")
    ratio = variances[bright] / means[bright]
    assert 0.7 < ratio.mean() < 1.3


# === Layer 4: determinism ==================================================


def test_noisy_pp_seeded_runs_are_bit_identical(clean_psf_image, focal_grid):
    """Same pp instance + same np.random.seed → bit-identical outputs.

    This is the user-facing pattern: build the sim once (which eagerly binds
    the detector and fixes the flat-field map), then loop on
    ``np.random.seed(N); sim.sample()`` to get reproducible noise draws.
    """
    pp = _make_pp(detector={"read_noise": 5.0, "dark_current_rate": 1.0, "flat_field": 0.05})
    _bind(pp, focal_grid)

    np.random.seed(42)
    out1 = pp(clean_psf_image, _ctx())

    np.random.seed(42)
    out2 = pp(clean_psf_image, _ctx())

    np.testing.assert_array_equal(out1, out2)


def test_noisy_pp_different_seeds_diverge(clean_psf_image, focal_grid):
    """Sanity: same processor, different seeds → measurably different outputs."""
    pp = _make_pp(detector={"read_noise": 5.0, "include_photon_noise": True})
    _bind(pp, focal_grid)

    np.random.seed(1)
    out1 = pp(clean_psf_image, _ctx())
    np.random.seed(2)
    out2 = pp(clean_psf_image, _ctx())

    assert not np.array_equal(out1, out2)
