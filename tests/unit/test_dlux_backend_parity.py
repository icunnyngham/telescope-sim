"""Cross-backend parity: the dlux (JAX) backend vs the default hcipy backend.

The dlux focal planes replace hcipy's per-wavelength ``apply()`` chain +
``FraunhoferPropagator`` with one summed pupil-plane OPD pushed through a
jitted, wavelength-vmapped matrix Fourier transform. That is a legitimate
rewrite only if it is numerically indistinguishable, so every test here
runs *the same config* through both backends and demands agreement at the
1e-12 level on max-normalized intensities (observed: ~1e-15, i.e. float64
round-off — the tolerance is slack, not a fitted bound).

Coverage map:

- ``elf_15seg`` preset: segmented PTT, two filters, 5 wavelengths each
  (broadband, ``num_samples > 1``) — images, reference PSFs, actuation echo.
- ``actuator_grid`` YAML: monochromatic (``num_samples == 1``) DM with a
  rotation misalignment — images + caller-facing echo.
- ``actuator_grid`` fit-role YAML + ``three_zernike_residual_fit`` YAML:
  the fit paths. ``fit_surface`` is shared numpy code on both backends, so
  actuator state and echoes must match *bit for bit*, not just closely.
- Physical focal plane, including the ``wavefront_total_power``
  renormalization (hcipy rescales each monochromatic wavefront; dlux folds
  the same factor into the λ-independent pupil amplitude once).
- Atmosphere: an OPD screen exposing both ``__call__(wf)`` (hcipy path) and
  ``.phase_for(lam)`` (dlux path) must land on the same image; a screen
  with only ``__call__`` must be *refused* on dlux rather than silently
  ignored.

Strehl note: cross-backend Strehl comparisons use ``matched_filter``.
``peak`` reads the reference PSF's ``argmax``, and on an even ``focal_res``
with a symmetric on-axis PSF the four central pixels tie to ~1e-15, so the
two backends can legitimately select different peak pixels — an intrinsic
degeneracy of the estimator, not a propagation difference.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import hcipy
import numpy as np
import pytest
import yaml

pytest.importorskip("dLux", reason="dlux backend requires the optional [dlux] extra")

from telescope_sim import TelescopeSim  # noqa: E402
from telescope_sim.backends.dlux.focal_planes import DLuxAngularFocalPlane  # noqa: E402
from telescope_sim.config.loader import build  # noqa: E402
from telescope_sim.config.schema import SimConfig  # noqa: E402

DATA = Path(__file__).parent / "data"

# Slack by ~3 orders of magnitude over the observed float64 round-off.
ATOL = 1e-12


# --- Helpers -----------------------------------------------------------------


def assert_image_parity(dlux_img, hcipy_img, *, atol=ATOL, min_dynamic_range=10.0):
    """Compare two image stacks after scaling by the hcipy image's peak.

    ``min_dynamic_range`` guards the test itself: a nearly-flat image would
    pass any parity assertion trivially, so require the reference image to
    actually look like a PSF (peak / floor above the given ratio).
    """
    hcipy_img = np.asarray(hcipy_img, dtype=np.float64)
    dlux_img = np.asarray(dlux_img, dtype=np.float64)
    assert dlux_img.shape == hcipy_img.shape
    peak = np.max(np.abs(hcipy_img))
    assert peak > 0
    assert peak / max(np.min(np.abs(hcipy_img)), np.finfo(float).tiny) > min_dynamic_range
    np.testing.assert_allclose(dlux_img / peak, hcipy_img / peak, rtol=0, atol=atol)


def _pair_from_yaml(path, mutate=None) -> tuple[TelescopeSim, TelescopeSim]:
    """Build (hcipy, dlux) sims from one YAML, optionally mutating the raw dict."""
    with open(path) as f:
        raw = yaml.safe_load(f)
    if mutate is not None:
        mutate(raw)
    config = SimConfig.model_validate(raw)
    return build(config, backend="hcipy"), build(config, backend="dlux")


def _widen_focal_planes(raw: dict, focal_extent: float = 2.0) -> None:
    """Give an angular focal plane a physically sensible field of view.

    The stock test YAMLs use a 1e-5 arcsec extent (~5e-5 λ/D across the
    whole grid), which renders an essentially flat patch — fine for the
    tests those files were written for, useless as a parity target. At 2
    arcsec the 1 µm / 1 m configs span ~10 λ/D with ~6 pixels per λ/D.
    """
    for fp in raw["focal_planes"].values():
        if fp["type"] == "angular":
            fp["focal_extent"] = focal_extent


class OPDScreen:
    """Fixed pupil-plane OPD screen usable by both backends.

    hcipy consumes ``__call__(wf) -> Wavefront`` (a thin phase screen);
    dlux consumes ``phase_for(lam)`` and folds the OPD into its summed
    pupil-plane OPD. Both express the same physics, which is the point.
    """

    def __init__(self, opd_m):
        self._opd = np.asarray(opd_m, dtype=np.float64).ravel()

    def phase_for(self, lam):
        return 2.0 * np.pi * self._opd / float(lam)

    def __call__(self, wf):
        field = hcipy.Field(
            np.asarray(wf.electric_field) * np.exp(1j * self.phase_for(wf.wavelength)),
            wf.grid,
        )
        return hcipy.Wavefront(field, wf.wavelength)


def _low_order_screen(pupil_grid, rms_m=5e-8, seed=3):
    """Smooth low-order OPD screen (Z2-Z6) on an oversized disk."""
    basis = hcipy.make_zernike_basis(5, 1.2, pupil_grid, starting_mode=2)
    rng = np.random.default_rng(seed)
    opd = sum(rng.normal() * np.asarray(m, dtype=np.float64) for m in basis)
    return opd * (rms_m / np.std(opd))


# --- elf_15seg preset (broadband, segmented PTT) ------------------------------


@pytest.fixture(scope="module")
def elf_pair() -> tuple[TelescopeSim, TelescopeSim]:
    """The packaged preset on both backends (256-pixel pupil — built once)."""
    return (
        TelescopeSim.from_preset("elf_15seg"),
        TelescopeSim.from_preset("elf_15seg", backend="dlux"),
    )


@pytest.fixture(scope="module")
def ptt_actuations() -> np.ndarray:
    """Seeded piston/tip/tilt commands for all 15 segments."""
    return np.random.default_rng(0).normal(size=(15, 3)) * 0.2


def test_from_preset_backend_override_selects_dlux_focal_planes(elf_pair):
    hcipy_sim, dlux_sim = elf_pair
    assert all(type(fp) is not DLuxAngularFocalPlane for fp in hcipy_sim.focal_planes.values())
    assert all(type(fp) is DLuxAngularFocalPlane for fp in dlux_sim.focal_planes.values())


def test_elf_reference_psfs_match(elf_pair):
    """Reference PSFs are built at load time (at-rest chain) — the normalization
    denominator for every downstream image, so they get their own assertion."""
    hcipy_sim, dlux_sim = elf_pair
    for name in hcipy_sim.focal_planes:
        h_fp, d_fp = hcipy_sim.focal_planes[name], dlux_sim.focal_planes[name]
        assert_image_parity(d_fp.reference_psf, h_fp.reference_psf)
        np.testing.assert_allclose(
            d_fp.reference_peak_intensity, h_fp.reference_peak_intensity, rtol=1e-12
        )
        np.testing.assert_allclose(d_fp.reference_psf_sum, h_fp.reference_psf_sum, rtol=1e-12)


def test_elf_at_rest_images_match(elf_pair):
    hcipy_sim, dlux_sim = elf_pair
    h = hcipy_sim.sample()["images"]["psf"]
    d = dlux_sim.sample()["images"]["psf"]
    assert h.shape == (128, 128, 2)  # two filters stacked on the last axis
    assert_image_parity(d, h)


def test_elf_broadband_images_match_under_random_ptt(elf_pair, ptt_actuations):
    """The headline case: 15 segments × piston/tip/tilt, 2 filters, 5 λ each."""
    hcipy_sim, dlux_sim = elf_pair
    h_out = hcipy_sim.sample({"segments": ptt_actuations})
    d_out = dlux_sim.sample({"segments": ptt_actuations})
    h_img = h_out["images"]["psf"]
    # max_intensity_norm has already divided by the reference peak, so the
    # aberrated PSF peak sits below 1 — a real actuation, not a no-op.
    assert 0.05 < h_img.max() < 1.0
    assert_image_parity(d_out["images"]["psf"], h_img)


def test_elf_actuation_echo_is_identical(elf_pair, ptt_actuations):
    """Actuator bookkeeping is backend-independent numpy — expect exact equality."""
    hcipy_sim, dlux_sim = elf_pair
    h_out = hcipy_sim.sample({"segments": ptt_actuations})
    d_out = dlux_sim.sample({"segments": ptt_actuations})
    assert set(h_out["actuations"]) == {"segments"}
    np.testing.assert_array_equal(d_out["actuations"]["segments"], h_out["actuations"]["segments"])


def test_elf_repeated_sampling_is_stable(elf_pair, ptt_actuations):
    """The jitted MFT is called once per sample; re-running must not drift
    (and the at-rest -> actuated -> at-rest round trip must return)."""
    _, dlux_sim = elf_pair
    first = dlux_sim.sample({"segments": ptt_actuations})["images"]["psf"]
    rest = dlux_sim.sample()["images"]["psf"]
    again = dlux_sim.sample({"segments": ptt_actuations})["images"]["psf"]
    np.testing.assert_array_equal(first, again)
    assert np.max(np.abs(first - rest)) > 1e-3


def test_dlux_focal_plane_result_carries_no_wavefronts(elf_pair):
    """Documented consequence of the summed-OPD path (why fiber_dual is gated)."""
    hcipy_sim, dlux_sim = elf_pair
    h_fp = next(iter(hcipy_sim.focal_planes.values()))
    d_fp = next(iter(dlux_sim.focal_planes.values()))
    assert len(h_fp._propagate_chain(hcipy_sim._c.correctors).wavefronts) == 5
    assert d_fp._propagate_chain(dlux_sim._c.correctors).wavefronts == []


# --- actuator_grid DM, monochromatic ------------------------------------------


def test_actuator_grid_monochromatic_parity():
    """num_samples == 1 path, with a rotated (misaligned) 8x8 DM."""
    hcipy_sim, dlux_sim = _pair_from_yaml(
        DATA / "actuator_grid_dm.yaml", mutate=_widen_focal_planes
    )
    cmd = np.random.default_rng(5).normal(size=(8, 8)) * 0.15
    h_out = hcipy_sim.sample({"dm": cmd})
    d_out = dlux_sim.sample({"dm": cmd})
    assert h_out["images"]["psf"].shape == (64, 64, 1)
    assert_image_parity(d_out["images"]["psf"], h_out["images"]["psf"])
    np.testing.assert_array_equal(d_out["actuations"]["dm"], h_out["actuations"]["dm"])


def test_actuator_grid_matched_filter_strehl_parity():
    """Strehl via matched_filter (see the peak-argmax degeneracy note above).

    2e-6 rad ~ 2 λ/D for this config's 1 µm / 1 m aperture, so the core
    mask covers the PSF core and a slice of the first ring.
    """

    def mutate(raw):
        _widen_focal_planes(raw)
        raw["strehl_method"] = "matched_filter"
        raw["strehl_core_rad"] = 2.0e-6

    hcipy_sim, dlux_sim = _pair_from_yaml(DATA / "actuator_grid_dm.yaml", mutate=mutate)
    cmd = np.random.default_rng(9).normal(size=(8, 8)) * 0.15

    h_rest = hcipy_sim.sample(meas_strehl=True)["strehls"]["filter1"]
    d_rest = dlux_sim.sample(meas_strehl=True)["strehls"]["filter1"]
    # At rest the sample PSF *is* the reference PSF, so Strehl is exactly 1.
    np.testing.assert_allclose(h_rest, 1.0, rtol=1e-12)
    np.testing.assert_allclose(d_rest, h_rest, rtol=1e-12)

    h_poked = hcipy_sim.sample({"dm": cmd}, meas_strehl=True)["strehls"]["filter1"]
    d_poked = dlux_sim.sample({"dm": cmd}, meas_strehl=True)["strehls"]["filter1"]
    assert 0.0 < h_poked < 1.0  # a real aberration, not a no-op
    np.testing.assert_allclose(d_poked, h_poked, rtol=1e-12)


# --- Fit-role / residual-fit paths --------------------------------------------


def test_actuator_grid_fit_role_with_atmosphere_parity():
    """Atmosphere + fit-role DM: identical fitted state and identical image.

    On hcipy the screen multiplies each monochromatic wavefront; on dlux its
    OPD is summed with the DM surface before a single propagation. The
    fit itself runs through the shared pipeline bookkeeping either way.
    """
    hcipy_sim, dlux_sim = _pair_from_yaml(
        DATA / "actuator_grid_fit_dm.yaml", mutate=_widen_focal_planes
    )
    screen = OPDScreen(_low_order_screen(hcipy_sim._c.pupil_grid))

    h_out = hcipy_sim.sample(atmos=screen)
    d_out = dlux_sim.sample(atmos=screen)

    h_corr, d_corr = hcipy_sim._c.correctors[0], dlux_sim._c.correctors[0]
    assert np.max(np.abs(np.asarray(h_corr.actuators))) > 0  # the fit actually fired
    np.testing.assert_array_equal(np.asarray(d_corr.actuators), np.asarray(h_corr.actuators))
    assert_image_parity(d_out["images"]["psf"], h_out["images"]["psf"])


def test_zernike_residual_fit_echo_parity():
    """Three Zernike DMs: two imposed, one fit-role, with residual-fit echoes.

    ``dm2`` reports ``residual_fit_only`` (the pre-self cumulative OPD in its
    own basis) and ``dm3`` ``actuators_plus_residual_fit`` — the two echo
    formulas that call ``fit_surface`` outside the apply path.
    """

    def mutate(raw):
        _widen_focal_planes(raw)
        raw["correctors"]["dm2"]["target_strategy"] = "residual_fit_only"
        raw["correctors"]["dm2"]["target"] = True
        raw["correctors"]["dm3"]["target_strategy"] = "actuators_plus_residual_fit"

    hcipy_sim, dlux_sim = _pair_from_yaml(DATA / "three_zernike_residual_fit.yaml", mutate=mutate)
    rng = np.random.default_rng(11)
    acts = {"dm1": rng.normal(size=8), "dm2": rng.normal(size=8)}

    h_out = hcipy_sim.sample(acts)
    d_out = dlux_sim.sample(acts)

    assert set(h_out["actuations"]) == {"dm2", "dm3"}
    for key in h_out["actuations"]:
        np.testing.assert_array_equal(d_out["actuations"][key], h_out["actuations"][key])
    # dm3 fits (and the pipeline negates) dm1+dm2, so the PSF returns to the
    # reference — a strong shared-code check on top of the parity assertion.
    assert_image_parity(d_out["images"]["psf"], h_out["images"]["psf"])
    ref = hcipy_sim.focal_planes["filter1"].reference_psf
    np.testing.assert_allclose(
        h_out["images"]["psf"][..., 0] / ref.max(), ref / ref.max(), rtol=0, atol=1e-6
    )


# --- Physical focal plane -----------------------------------------------------


def _physical_config(wavefront_total_power: float | None) -> dict[str, Any]:
    """Broadband (3 λ) physical focal plane at ~2.4 pixels per λf/D."""
    fp: dict[str, Any] = {
        "type": "physical",
        "central_lam": 1.5e-6,
        "focal_extent": 2.0e-4,
        "focal_res": 32,
        "focal_length": 10.0,
        "fractional_bandwidth": 0.05,
        "num_samples": 3,
    }
    if wavefront_total_power is not None:
        fp["wavefront_total_power"] = wavefront_total_power
    return {
        "pupil": {"resolution": 64, "extent": 1.05},
        "aperture": {
            "type": "external_pupil",
            "module": "hcipy",
            "function": "make_circular_aperture",
            "mode": "callable",
            "kwargs": {"diameter": 1.0},
            "area": float(np.pi * 0.25),
        },
        "correctors": {
            "dm": {
                "type": "zernike",
                "n_modes": 6,
                "zernike_diameter": 1.0,
                "starting_mode": 2,
                "actuate_scale": 5.0e-8,
                "target_strategy": "actuators",
                "target": True,
            }
        },
        "corrector_chain": ["dm"],
        "focal_planes": {"f1": fp},
        "outputs": {
            "psf": {"tap": {"type": "intensity", "focal_planes": ["f1"]}, "post_processing": []}
        },
        "strehl_method": "matched_filter",
        "strehl_core_rad": 3.0e-5,  # ~2 λf/D in the physical grid's metres
    }


@pytest.mark.parametrize("total_power", [None, 1.0])
def test_physical_focal_plane_parity(total_power):
    """Both with and without the ``wavefront_total_power`` renormalization.

    hcipy applies it per monochromatic wavefront (``wf.total_power = P``);
    dlux scales the shared real pupil amplitude by ``sqrt(P / current)``
    once. Equivalent only because the aperture field is λ-independent.
    """
    config = SimConfig.model_validate(_physical_config(total_power))
    hcipy_sim = build(config, backend="hcipy")
    dlux_sim = build(config, backend="dlux")

    act = np.random.default_rng(7).normal(size=6)
    h_out = hcipy_sim.sample({"dm": act}, meas_strehl=True)
    d_out = dlux_sim.sample({"dm": act}, meas_strehl=True)

    assert_image_parity(d_out["images"]["psf"], h_out["images"]["psf"])
    assert_image_parity(
        dlux_sim.focal_planes["f1"].reference_psf, hcipy_sim.focal_planes["f1"].reference_psf
    )
    np.testing.assert_allclose(
        dlux_sim.focal_planes["f1"].reference_psf_sum,
        hcipy_sim.focal_planes["f1"].reference_psf_sum,
        rtol=1e-12,
    )
    np.testing.assert_allclose(d_out["strehls"]["f1"], h_out["strehls"]["f1"], rtol=1e-12)
    np.testing.assert_array_equal(d_out["actuations"]["dm"], h_out["actuations"]["dm"])


def test_wavefront_total_power_actually_rescales_on_both_backends():
    """Guards the parametrization above: the two ``total_power`` settings must
    reach genuinely different absolute levels (otherwise both parametrized
    cases would exercise the same code path), by the same factor on both
    backends.

    The unnormalized pupil already carries total power
    ``P0 = Σ|aperture|²·w``; requesting ``total_power = 1`` scales the field
    by ``sqrt(1/P0)``, so focal intensity scales by ``1/P0`` exactly. (``P0``
    is a little below the geometric area π/4 because supersampled rim pixels
    are squared here but not in the area integral.)
    """
    sims = {}
    for backend in ("hcipy", "dlux"):
        for power in (None, 1.0):
            sims[backend, power] = build(
                SimConfig.model_validate(_physical_config(power)), backend=backend
            )
    sums = {k: s.focal_planes["f1"].reference_psf_sum for k, s in sims.items()}

    grid = sims["hcipy", None]._c.pupil_grid
    aper = np.asarray(sims["hcipy", None].aperture.field, dtype=np.float64)
    p0 = float((aper**2 * np.asarray(grid.weights)).sum())
    assert abs(p0 - 1.0) > 0.1  # the rescale is not a disguised no-op
    for backend in ("hcipy", "dlux"):
        np.testing.assert_allclose(sums[backend, None] / sums[backend, 1.0], p0, rtol=1e-12)
    np.testing.assert_allclose(sums["dlux", None], sums["hcipy", None], rtol=1e-12)
    np.testing.assert_allclose(sums["dlux", 1.0], sums["hcipy", 1.0], rtol=1e-12)


def test_physical_focal_plane_atmosphere_parity():
    """Atmosphere through the broadband physical plane (3 wavelengths)."""
    config = SimConfig.model_validate(_physical_config(1.0))
    hcipy_sim = build(config, backend="hcipy")
    dlux_sim = build(config, backend="dlux")
    screen = OPDScreen(_low_order_screen(hcipy_sim._c.pupil_grid, rms_m=8e-8, seed=17))

    act = np.random.default_rng(21).normal(size=6)
    h_out = hcipy_sim.sample({"dm": act}, atmos=screen)
    d_out = dlux_sim.sample({"dm": act}, atmos=screen)
    # The screen genuinely perturbs the image (vs. the no-atmosphere case).
    h_clear = hcipy_sim.sample({"dm": act})["images"]["psf"]
    assert np.max(np.abs(h_out["images"]["psf"] - h_clear)) > 1e-3 * h_clear.max()
    assert_image_parity(d_out["images"]["psf"], h_out["images"]["psf"])


# --- Atmosphere contract ------------------------------------------------------


def test_atmos_without_phase_for_is_refused_on_dlux():
    """A bare wf->wf callable cannot be folded into a summed-OPD propagation;
    the backend must say so rather than silently dropping the atmosphere."""
    config = SimConfig.model_validate(_physical_config(1.0))
    hcipy_sim = build(config, backend="hcipy")
    dlux_sim = build(config, backend="dlux")
    screen = OPDScreen(_low_order_screen(hcipy_sim._c.pupil_grid))

    def callable_only(wf):
        """Same physics as ``screen``, but deliberately no ``phase_for``."""
        return screen(wf)

    assert not hasattr(callable_only, "phase_for")
    hcipy_sim.sample(atmos=callable_only)  # hcipy accepts any wf->wf callable
    with pytest.raises(NotImplementedError, match=r"requires an atmosphere exposing"):
        dlux_sim.sample(atmos=callable_only)
