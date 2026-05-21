"""Tests for ``ConvolveImagePostProcessor``.

The convolve post-processor sits between the ``intensity`` tap (clean PSF)
and any downstream noise / normalization processors. Legacy reference
(multi_aperture_psf.py:489-491):

    out_samp = fftconvolve(convolve_im, psf / lam_setup['ref_psf_sum'],
                           mode='same')

Coverage:

1. **Structural**: output shape matches the input scene (not the PSF),
   kwarg honoring, per-sample image override beats YAML default,
   single-focal-plane restriction.
2. **Math**: bit-identical to a hand-computed ``fftconvolve(scene, psf /
   reference_psf_sum, mode='same')``.
3. **Composition**: ``convolve_image → noisy_detector`` produces a noisy
   image of the convolved-scene shape (not the PSF shape), with total
   photons matching ``flux * area`` when configured.
4. **Validation**: rejects multi-focal-plane bindings, wrong input
   shapes, malformed YAML defaults.
"""

from __future__ import annotations

import hcipy
import numpy as np
import pytest
from scipy.signal import fftconvolve

import telescope_sim.post  # noqa: F401  (registers post-processors)
from telescope_sim.abc import PipelineContext

FOCAL_RES = 32
PUPIL_RES = 64
APER_DIAM = 1.0
PUPIL_EXTENT = 1.05
APER_AREA = float(np.pi * (APER_DIAM / 2) ** 2)
SCENE_H, SCENE_W = 24, 28  # deliberately different from focal_res to verify shape contract


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
    """A clean (H, W, 1) PSF image — what IntensityOutputTap produces."""
    prop = hcipy.FraunhoferPropagator(pupil_grid, focal_grid)
    wf = hcipy.Wavefront(aper_field, 1.0e-6)
    return np.asarray(prop(wf).intensity.shaped)[..., None]


@pytest.fixture(scope="module")
def reference_psf_sum(clean_psf_image):
    """Stand-in for fp.reference_psf_sum (cached at sim-build)."""
    return float(clean_psf_image[..., 0].sum())


@pytest.fixture(scope="module")
def synthetic_scene():
    """A non-trivial scene: a Gaussian blob with a high-contrast point source."""
    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[0:SCENE_H, 0:SCENE_W]
    gauss = np.exp(-((yy - 12) ** 2 + (xx - 14) ** 2) / 40.0)
    scene = 5.0 * gauss + 0.1 * rng.uniform(size=(SCENE_H, SCENE_W))
    scene[8, 20] = 50.0  # point source
    return scene.astype(np.float64)


# Loader stand-ins -----------------------------------------------------------


class _FakeFocalPlane:
    def __init__(self, reference_psf_sum, focal_grid):
        self.reference_psf_sum = reference_psf_sum
        self.lam_setup = type("_S", (), {"focal_grid": focal_grid})()


class _FakeApertureResult:
    def __init__(self, area):
        self.area = area


def _bind(pp, reference_psf_sum, focal_grid):
    pp._bind_loader_dependencies(
        aperture_result=_FakeApertureResult(APER_AREA),
        focal_planes={"f1": _FakeFocalPlane(reference_psf_sum, focal_grid)},
        focal_plane_names=["f1"],
    )


def _ctx(overrides=None):
    return PipelineContext(
        output_name="scene",
        focal_plane_name="f1",
        reference_peak_intensity=None,
        reference_psf_sum=None,
        overrides=overrides or {},
    )


def _make_pp(*, image=None):
    from telescope_sim.post.convolve import ConvolveImagePostProcessor

    return ConvolveImagePostProcessor(image=image)


# === Structural ============================================================


def test_convolve_output_shape_matches_scene_not_psf(
    clean_psf_image, reference_psf_sum, focal_grid, synthetic_scene
):
    """``fftconvolve(scene, kernel, mode='same')`` returns shape(scene), not shape(psf)."""
    pp = _make_pp(image=synthetic_scene)
    _bind(pp, reference_psf_sum, focal_grid)
    out = pp(clean_psf_image, _ctx())
    assert out.shape == (SCENE_H, SCENE_W, 1)
    assert out.dtype == np.float64


def test_convolve_per_sample_override_beats_yaml_default(
    clean_psf_image, reference_psf_sum, focal_grid, synthetic_scene
):
    """``overrides={"convolve_image": np.array(...)}`` overrides the constructor default."""
    default_scene = synthetic_scene
    override_scene = synthetic_scene[:16, :16] * 2.0  # different shape and content

    pp = _make_pp(image=default_scene)
    _bind(pp, reference_psf_sum, focal_grid)

    out_default = pp(clean_psf_image, _ctx())
    out_override = pp(clean_psf_image, _ctx(overrides={"convolve_image": override_scene}))
    assert out_default.shape == (SCENE_H, SCENE_W, 1)
    assert out_override.shape == (16, 16, 1)
    assert not np.array_equal(out_default[:16, :16, 0], out_override[..., 0])


def test_convolve_no_default_no_override_is_passthrough(
    clean_psf_image, reference_psf_sum, focal_grid
):
    """If no image is configured anywhere, the post-processor is a passthrough.

    Useful for declaring the processor in the YAML and toggling convolve on
    only when the caller supplies an image at sample time.
    """
    pp = _make_pp(image=None)
    _bind(pp, reference_psf_sum, focal_grid)
    out = pp(clean_psf_image, _ctx())
    np.testing.assert_array_equal(out, clean_psf_image)


def test_convolve_rejects_multi_focal_plane(reference_psf_sum, focal_grid):
    pp = _make_pp(image=np.zeros((4, 4)))
    with pytest.raises(ValueError, match="exactly one focal_plane"):
        pp._bind_loader_dependencies(
            aperture_result=_FakeApertureResult(APER_AREA),
            focal_planes={
                "a": _FakeFocalPlane(reference_psf_sum, focal_grid),
                "b": _FakeFocalPlane(reference_psf_sum, focal_grid),
            },
            focal_plane_names=["a", "b"],
        )


def test_convolve_rejects_missing_reference_sum(focal_grid):
    """If the focal plane has reference_psf_sum=None (unbuilt reference PSF),
    the bind hook raises clearly."""
    pp = _make_pp(image=np.zeros((4, 4)))
    fp = _FakeFocalPlane(reference_psf_sum=None, focal_grid=focal_grid)
    with pytest.raises(RuntimeError, match="reference_psf_sum"):
        pp._bind_loader_dependencies(
            aperture_result=_FakeApertureResult(APER_AREA),
            focal_planes={"f1": fp},
            focal_plane_names=["f1"],
        )


def test_convolve_unbound_raises(clean_psf_image):
    pp = _make_pp(image=np.zeros((4, 4)))
    with pytest.raises(RuntimeError, match="_bind_loader_dependencies"):
        pp(clean_psf_image, _ctx())


def test_convolve_rejects_wrong_input_shape(reference_psf_sum, focal_grid, synthetic_scene):
    pp = _make_pp(image=synthetic_scene)
    _bind(pp, reference_psf_sum, focal_grid)
    bad = np.zeros((FOCAL_RES, FOCAL_RES, 3), dtype=np.float64)  # multi-channel
    with pytest.raises(ValueError, match="single-channel"):
        pp(bad, _ctx())


def test_convolve_rejects_non_2d_scene(clean_psf_image, reference_psf_sum, focal_grid):
    pp = _make_pp(image=np.zeros((4, 4, 4)))  # 3D scene
    _bind(pp, reference_psf_sum, focal_grid)
    with pytest.raises(ValueError, match="2D scene"):
        pp(clean_psf_image, _ctx())


# === Math: bit-identical to legacy formula =================================


def test_convolve_matches_hand_computed_legacy_formula(
    clean_psf_image, reference_psf_sum, focal_grid, synthetic_scene
):
    """The output equals ``fftconvolve(scene, psf / reference_psf_sum, mode='same')``."""
    pp = _make_pp(image=synthetic_scene)
    _bind(pp, reference_psf_sum, focal_grid)
    out = pp(clean_psf_image, _ctx())[..., 0]

    expected_kernel = clean_psf_image[..., 0] / reference_psf_sum
    expected = fftconvolve(synthetic_scene, expected_kernel, mode="same")

    np.testing.assert_allclose(out, expected, rtol=0, atol=0)


def test_convolve_kernel_uses_reference_sum_not_current_sum(
    clean_psf_image, reference_psf_sum, focal_grid, synthetic_scene
):
    """The kernel normalization divides by the AT-REST reference PSF sum
    (cached at sim-build), not by the current sample's PSF sum.

    For an aberrated PSF, ``psf.sum() ≈ reference_psf.sum()`` (Parseval),
    but pixel-for-pixel they differ. Verify by feeding a deliberately
    rescaled PSF and confirming the kernel uses the stored reference, not
    the current.
    """
    pp = _make_pp(image=synthetic_scene)
    _bind(pp, reference_psf_sum, focal_grid)

    # Scale the current PSF by 0.5 — kernel should still divide by the
    # original reference_psf_sum, so the convolved output should also
    # halve.
    out_normal = pp(clean_psf_image, _ctx())
    out_halved = pp(0.5 * clean_psf_image, _ctx())
    np.testing.assert_allclose(out_halved, 0.5 * out_normal, rtol=1e-12, atol=1e-30)


# === Composition with noisy_detector =======================================


def test_convolve_then_noisy_detector_output_shape(
    clean_psf_image, reference_psf_sum, focal_grid, synthetic_scene
):
    """The full clean→convolve→noise chain produces a (scene_H, scene_W, 1) image."""
    from telescope_sim.post.noisy_detector import NoisyDetectorPostProcessor

    # NoisyDetector needs to bind to the convolved-output focal grid. The
    # legacy contract assumes scene & focal grid match; here we mirror that
    # by building a focal_grid sized to match the scene.
    scene_grid = hcipy.make_uniform_grid([SCENE_W, SCENE_H], 1e-5 * np.pi / (180.0 * 3600.0))

    convolve_pp = _make_pp(image=synthetic_scene)
    _bind(convolve_pp, reference_psf_sum, focal_grid)

    noisy_pp = NoisyDetectorPostProcessor(
        int_phot_flux=1.0e6,
        detector={"include_photon_noise": False, "read_noise": 0.0, "flat_field": 0.0},
        clamp_nonnegative=False,
    )
    noisy_pp._bind_loader_dependencies(
        aperture_result=_FakeApertureResult(APER_AREA),
        focal_planes={"f1": _FakeFocalPlane(reference_psf_sum, scene_grid)},
        focal_plane_names=["f1"],
    )

    convolved = convolve_pp(clean_psf_image, _ctx())
    noisy = noisy_pp(convolved, _ctx())

    assert noisy.shape == (SCENE_H, SCENE_W, 1)
    # With noise off + clamp off + flux configured, total counts ≈ flux * area
    total = noisy.sum()
    assert 0.5 * 1.0e6 * APER_AREA < total < 1.1 * 1.0e6 * APER_AREA


def test_convolve_then_noisy_detector_flux_none_preserves_scene_brightness(
    clean_psf_image, reference_psf_sum, focal_grid, synthetic_scene
):
    """With ``int_phot_flux=None``, the noisy stage doesn't rescale — the scene's
    natural brightness (× grid.weights) is integrated as-is. Useful for
    extended-source simulations where the user wants the input image's
    absolute scale to carry through.
    """
    from telescope_sim.post.noisy_detector import NoisyDetectorPostProcessor

    scene_grid = hcipy.make_uniform_grid([SCENE_W, SCENE_H], 1e-5 * np.pi / (180.0 * 3600.0))

    convolve_pp = _make_pp(image=synthetic_scene)
    _bind(convolve_pp, reference_psf_sum, focal_grid)

    noisy_pp = NoisyDetectorPostProcessor(
        int_phot_flux=None,  # ← no rescale
        detector={
            "include_photon_noise": False,
            "read_noise": 0.0,
            "dark_current_rate": 0.0,
            "flat_field": 0.0,
        },
        clamp_nonnegative=False,
    )
    noisy_pp._bind_loader_dependencies(
        aperture_result=_FakeApertureResult(APER_AREA),
        focal_planes={"f1": _FakeFocalPlane(reference_psf_sum, scene_grid)},
        focal_plane_names=["f1"],
    )

    convolved = convolve_pp(clean_psf_image, _ctx())
    noisy = noisy_pp(convolved, _ctx())

    # With all noise sources zeroed, the noisy output equals the convolved
    # image × grid.weights (the power-field conversion).
    weights_arr = np.asarray(scene_grid.weights)
    expected = (np.asarray(convolved[..., 0]).ravel() * weights_arr).reshape(SCENE_H, SCENE_W)
    np.testing.assert_allclose(noisy[..., 0], expected, rtol=1e-12, atol=1e-30)
