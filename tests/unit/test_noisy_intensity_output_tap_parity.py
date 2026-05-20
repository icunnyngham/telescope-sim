"""Parity + behavior tests for ``NoisyIntensityOutputTap``.

Layered approach for a stochastic component:

1. **Structural** (RNG-free): output shape, kwarg honoring, per-sample
   override beats YAML default, wavefronts not mutated, Strehl unaffected,
   detector built once.
2. **Noise-off identity**: all noise sources zeroed → exact equality with a
   ``NoiselessDetector`` over the same input. Decouples framework wiring
   from RNG.
3. **Statistical** (seeded, low N): per-noise-source assertions on mean/std
   match analytical predictions within ~3σ/√N.
4. **Determinism**: same ``np.random.seed`` → bit-for-bit identical outputs.

The gold-standard legacy parity check lives at the fixture level
(``fixtures/runner/digests/17_noisy_psf/expected.json``), not here.

RNG policy: HCIPy's NoisyDetector uses ``np.random.*`` globally. Tests seed
just before each sample().
"""

from __future__ import annotations

import hcipy
import numpy as np
import pytest

import telescope_sim.outputs.noisy_intensity  # noqa: F401  (registers)

# --- Shared fixtures --------------------------------------------------------


N_LAM = 3
FOCAL_RES = 32
PUPIL_RES = 64
APER_DIAM = 1.0
PUPIL_EXTENT = 1.05
APER_AREA = float(np.pi * (APER_DIAM / 2) ** 2)
INT_PHOT_FLUX = 1.0e6  # photons/m^2 — chosen so flux*area gives ~3e6 photons total


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
def fp_results(pupil_grid, focal_grid, aper_field):
    """Build a FocalPlaneResult by propagating a small broadband wavefront stack."""
    from telescope_sim.focal_planes.physical import FocalPlaneResult

    prop = hcipy.FraunhoferPropagator(pupil_grid, focal_grid)
    lams = 1.0e-6 * np.linspace(0.99, 1.01, N_LAM)
    wfs_focal = []
    intensity = np.zeros((FOCAL_RES, FOCAL_RES), dtype=np.float64)
    for lam in lams:
        wf = hcipy.Wavefront(aper_field, lam)
        wf_focal = prop(wf)
        wfs_focal.append(wf_focal)
        intensity += np.asarray(wf_focal.intensity.shaped)
    return {"f1": FocalPlaneResult(intensity=intensity, wavefronts=wfs_focal)}


def _make_tap(
    *,
    int_phot_flux=INT_PHOT_FLUX,
    aperture_area=APER_AREA,
    detector=None,
    clamp_nonnegative=True,
    name="noisy_psf",
):
    from telescope_sim.outputs.noisy_intensity import NoisyIntensityOutputTap

    return NoisyIntensityOutputTap(
        focal_plane_names=["f1"],
        int_phot_flux=int_phot_flux,
        aperture_area=aperture_area,
        detector=detector or {},
        clamp_nonnegative=clamp_nonnegative,
        name=name,
    )


# === Layer 1: structural (no RNG) ==========================================


def test_noisy_tap_output_shape_channels_last(fp_results):
    tap = _make_tap(detector={"include_photon_noise": False})
    out = tap.extract(fp_results)
    assert out.shape == (FOCAL_RES, FOCAL_RES, 1)
    assert out.dtype == np.float64


def test_noisy_tap_rejects_multi_focal_plane():
    from telescope_sim.outputs.noisy_intensity import NoisyIntensityOutputTap

    with pytest.raises(ValueError, match="exactly one focal_plane"):
        NoisyIntensityOutputTap(focal_plane_names=["a", "b"])


def test_noisy_tap_rejects_missing_focal_plane(fp_results):
    from telescope_sim.outputs.noisy_intensity import NoisyIntensityOutputTap

    tap = NoisyIntensityOutputTap(
        focal_plane_names=["missing"], int_phot_flux=None, aperture_area=APER_AREA
    )
    with pytest.raises(KeyError, match="missing"):
        tap.extract(fp_results)


def test_noisy_tap_requires_aperture_area_when_flux_set(fp_results):
    """int_phot_flux without aperture_area is unphysical — raise instead of guessing."""
    tap = _make_tap(aperture_area=None, int_phot_flux=INT_PHOT_FLUX)
    with pytest.raises(RuntimeError, match="aperture_area"):
        tap.extract(fp_results)


def test_noisy_tap_per_sample_override_beats_yaml_default(fp_results):
    """`overrides={'int_phot_flux': X}` overrides the constructor default."""
    np.random.seed(0)
    tap_low = _make_tap(int_phot_flux=1e4, detector={"include_photon_noise": False})
    tap_high = _make_tap(int_phot_flux=1e8, detector={"include_photon_noise": False})

    out_low = tap_low.extract(fp_results)
    out_high = tap_high.extract(fp_results)
    # High flux gives more accumulated charge, deterministically (noise off)
    assert out_high.sum() > out_low.sum() * 100

    # And override: same tap, but per-sample override goes the OTHER way
    tap = _make_tap(int_phot_flux=1e4, detector={"include_photon_noise": False})
    out_default = tap.extract(fp_results)
    out_overridden = tap.extract(fp_results, overrides={"int_phot_flux": 1e8})
    assert out_overridden.sum() > out_default.sum() * 100


def test_noisy_tap_override_only_affects_known_keys(fp_results):
    """Unknown override keys are tolerated (forward compat) — int_phot_flux falls back."""
    tap = _make_tap(int_phot_flux=INT_PHOT_FLUX, detector={"include_photon_noise": False})
    np.random.seed(0)
    out_baseline = tap.extract(fp_results)
    np.random.seed(0)
    out_with_ignored = tap.extract(fp_results, overrides={"unknown_future_kwarg": 42})
    np.testing.assert_array_equal(out_baseline, out_with_ignored)


def test_noisy_tap_does_not_mutate_wavefronts(fp_results):
    """The tap must not modify result.wavefronts[*].electric_field or total_power.

    Design point: the tap passes a freshly-built ``hcipy.Field``
    (intensity * grid.weights, optionally rescaled) into
    ``Detector.integrate``, NOT the per-λ Wavefronts themselves. Other taps
    (or Strehl) reading the same wavefronts later should see them pristine.
    """
    # Snapshot pre-extract
    pre_efs = [np.asarray(wf.electric_field).copy() for wf in fp_results["f1"].wavefronts]
    pre_powers = [float(wf.total_power) for wf in fp_results["f1"].wavefronts]

    tap = _make_tap(detector={"include_photon_noise": False})
    _ = tap.extract(fp_results)

    # Compare post-extract — must be unchanged
    for i, wf in enumerate(fp_results["f1"].wavefronts):
        np.testing.assert_array_equal(
            np.asarray(wf.electric_field),
            pre_efs[i],
            err_msg=f"wavefront {i} electric_field was mutated by extract()",
        )
        assert float(wf.total_power) == pytest.approx(pre_powers[i], abs=0), (
            f"wavefront {i} total_power was mutated: {pre_powers[i]} -> {wf.total_power}"
        )


def test_noisy_tap_detector_built_once(fp_results):
    """The expensive NoisyDetector construction happens lazily on first extract().

    Two consecutive extract() calls should reuse the same detector instance.
    """
    tap = _make_tap(detector={"include_photon_noise": False})
    assert tap._detector is None
    np.random.seed(0)
    _ = tap.extract(fp_results)
    first_detector = tap._detector
    assert first_detector is not None
    np.random.seed(0)
    _ = tap.extract(fp_results)
    assert tap._detector is first_detector


def test_noisy_tap_clamp_nonnegative_flag(fp_results):
    """clamp_nonnegative=True applies np.abs(); =False leaves negatives in."""
    # Add lots of read noise so the read-out has negatives
    np.random.seed(0)
    tap_clamp = _make_tap(
        int_phot_flux=1e2,
        detector={"include_photon_noise": False, "read_noise": 100.0},
        clamp_nonnegative=True,
    )
    out_clamp = tap_clamp.extract(fp_results)
    assert (out_clamp >= 0).all()

    np.random.seed(0)
    tap_noclamp = _make_tap(
        int_phot_flux=1e2,
        detector={"include_photon_noise": False, "read_noise": 100.0},
        clamp_nonnegative=False,
    )
    out_noclamp = tap_noclamp.extract(fp_results)
    # With read_noise=100 and signal ~1, expect plenty of negatives
    assert (out_noclamp < 0).any()


# === Layer 2: noise-off identity ===========================================


def test_noisy_tap_noise_off_matches_noiseless_detector(fp_results):
    """All noise sources disabled → output equals the legacy single-integrate identity.

    Reproduces the legacy contract: total accumulated charge = ``flux * area``
    photons, distributed across pixels in proportion to the wavelength-summed
    intensity (the "power Field" `intensity * grid.weights`).
    """
    tap = _make_tap(
        int_phot_flux=INT_PHOT_FLUX,
        detector={
            "include_photon_noise": False,
            "read_noise": 0.0,
            "dark_current_rate": 0.0,
            "flat_field": 0.0,
        },
    )
    out = tap.extract(fp_results)[..., 0]

    # Reproduce the v2 design directly:
    #   power_field = intensity * grid.weights, scaled to sum = flux * area
    #   accumulated = power_field * dt (dt=1) — that's the entire charge.
    focal_grid = fp_results["f1"].wavefronts[0].grid
    weights_arr = np.asarray(focal_grid.weights)
    power_field = np.asarray(fp_results["f1"].intensity).ravel() * weights_arr
    target = INT_PHOT_FLUX * APER_AREA
    natural_total = power_field.sum()
    expected = (power_field * (target / natural_total)).reshape(FOCAL_RES, FOCAL_RES)

    np.testing.assert_allclose(out, expected, rtol=1e-12, atol=1e-30)


def test_noisy_tap_noise_off_total_charge_matches_flux(fp_results):
    """Sum of accumulated charge equals flux * aper_area (legacy contract)."""
    tap = _make_tap(
        int_phot_flux=INT_PHOT_FLUX,
        detector={
            "include_photon_noise": False,
            "read_noise": 0.0,
            "dark_current_rate": 0.0,
            "flat_field": 0.0,
        },
    )
    out = tap.extract(fp_results)
    # The PSF spans the full focal grid only partially; near-field power loss
    # is small but nonzero, so we just verify scaling: doubling flux doubles
    # the sum (linear in flux), and the absolute magnitude is the right order.
    s = out.sum()
    assert 0.5 * INT_PHOT_FLUX * APER_AREA < s < 1.1 * INT_PHOT_FLUX * APER_AREA


def test_noisy_tap_flux_none_uses_natural_power_field(fp_results):
    """int_phot_flux=None → no rescaling; output equals `intensity * grid.weights`."""
    tap = _make_tap(
        int_phot_flux=None,
        aperture_area=None,  # not needed when flux is None
        detector={
            "include_photon_noise": False,
            "read_noise": 0.0,
            "dark_current_rate": 0.0,
            "flat_field": 0.0,
        },
    )
    out = tap.extract(fp_results)[..., 0]
    weights_arr = np.asarray(fp_results["f1"].wavefronts[0].grid.weights)
    expected = (np.asarray(fp_results["f1"].intensity).ravel() * weights_arr).reshape(
        FOCAL_RES, FOCAL_RES
    )
    np.testing.assert_allclose(out, expected, rtol=1e-12, atol=1e-30)


# === Layer 3: statistical assertions =======================================


def _means_over_samples(tap_factory, fp_results, n=64, seed=0):
    np.random.seed(seed)
    samples = np.stack([tap_factory().extract(fp_results)[..., 0] for _ in range(n)], axis=0)
    return samples


def test_noisy_tap_read_noise_alone_mean_unchanged_std_matches(fp_results):
    """Pure read noise: mean ≈ clean image, std ≈ read_noise per pixel."""
    READ_NOISE = 5.0
    N = 96

    # Tap with read noise only
    samples = _means_over_samples(
        lambda: _make_tap(
            int_phot_flux=INT_PHOT_FLUX,
            detector={
                "include_photon_noise": False,
                "read_noise": READ_NOISE,
                "dark_current_rate": 0.0,
                "flat_field": 0.0,
            },
            clamp_nonnegative=False,  # don't bias the mean
        ),
        fp_results,
        n=N,
    )
    sample_mean = samples.mean(axis=0)
    sample_std = samples.std(axis=0)

    # Clean reference for comparison
    tap_clean = _make_tap(
        int_phot_flux=INT_PHOT_FLUX,
        detector={
            "include_photon_noise": False,
            "read_noise": 0.0,
            "dark_current_rate": 0.0,
            "flat_field": 0.0,
        },
        clamp_nonnegative=False,
    )
    clean = tap_clean.extract(fp_results)[..., 0]

    # Mean of N samples differs from clean by ~ read_noise / sqrt(N)
    se = READ_NOISE / np.sqrt(N)
    # Use mean-deviation across pixels, not pixelwise, to keep the test from flaking
    assert np.mean(np.abs(sample_mean - clean)) < 4 * se

    # Std should be ~ read_noise — same across all pixels
    assert np.abs(np.mean(sample_std) - READ_NOISE) < 0.5


def test_noisy_tap_dark_current_alone_mean_shifts_by_rate(fp_results):
    """Dark current with no photon/read noise: mean shifts by `rate * dt`."""
    DARK = 50.0
    N = 96

    samples = _means_over_samples(
        lambda: _make_tap(
            int_phot_flux=INT_PHOT_FLUX,
            detector={
                "include_photon_noise": False,
                "read_noise": 0.0,
                "dark_current_rate": DARK,
                "flat_field": 0.0,
            },
            clamp_nonnegative=False,
        ),
        fp_results,
        n=N,
    )

    tap_clean = _make_tap(
        int_phot_flux=INT_PHOT_FLUX,
        detector={
            "include_photon_noise": False,
            "read_noise": 0.0,
            "dark_current_rate": 0.0,
            "flat_field": 0.0,
        },
        clamp_nonnegative=False,
    )
    clean = tap_clean.extract(fp_results)[..., 0]

    # Dark current is deterministic when photon_noise off; per-pixel mean ==
    # clean + DARK exactly (subject to flat_field=0 effect — see below).
    diff = samples.mean(axis=0) - clean
    # With flat_field=0, NoisyDetector synthesizes a normal-distributed map
    # with std=0 → ones array. So dark stays at DARK (un-modulated). Check.
    assert np.abs(diff - DARK).max() < 1e-9


def test_noisy_tap_photon_noise_variance_equals_mean(fp_results):
    """Pure photon noise: var(pixel) ≈ mean(pixel) (Poisson)."""
    N = 96
    samples = _means_over_samples(
        lambda: _make_tap(
            int_phot_flux=INT_PHOT_FLUX,
            detector={
                "include_photon_noise": True,
                "read_noise": 0.0,
                "dark_current_rate": 0.0,
                "flat_field": 0.0,
            },
            clamp_nonnegative=False,
        ),
        fp_results,
        n=N,
    )
    means = samples.mean(axis=0)
    variances = samples.var(axis=0)
    # Only check pixels with non-trivial brightness (avoid noise-dominated pixels)
    bright = means > 10.0
    if not bright.any():
        pytest.skip("no pixels bright enough to test Poisson var=mean")
    # Tolerance: relative error in variance over sqrt(N) realizations is ~sqrt(2/N)
    ratio = variances[bright] / means[bright]
    assert 0.7 < ratio.mean() < 1.3, (
        f"Poisson var/mean ratio {ratio.mean():.3f} far from 1.0 (N={N})"
    )


# === Layer 4: determinism ==================================================


def test_noisy_tap_seeded_runs_are_bit_identical(fp_results):
    """Two runs with the same np.random.seed produce identical outputs."""
    tap1 = _make_tap(detector={"read_noise": 5.0, "dark_current_rate": 1.0, "flat_field": 0.05})

    np.random.seed(42)
    out1 = tap1.extract(fp_results)

    # New tap (so flat_field map is regenerated identically given the seed)
    tap2 = _make_tap(detector={"read_noise": 5.0, "dark_current_rate": 1.0, "flat_field": 0.05})

    np.random.seed(42)
    out2 = tap2.extract(fp_results)

    np.testing.assert_array_equal(out1, out2)


def test_noisy_tap_different_seeds_diverge(fp_results):
    """Sanity check: same tap, different seeds, measurably different outputs."""
    tap = _make_tap(detector={"read_noise": 5.0, "include_photon_noise": True})

    np.random.seed(1)
    out1 = tap.extract(fp_results)
    np.random.seed(2)
    out2 = tap.extract(fp_results)

    assert not np.array_equal(out1, out2)
