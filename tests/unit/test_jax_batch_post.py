"""On-device post-processing for ``sample_batch(key=...)``.

The key-mode contract has three load-bearing properties, each pinned
here:

1. **Determinism fork, bounded**: noisy outputs are validated
   *statistically* (photon mean/variance against the charge model, read
   noise variance, flux scaling) and for seeded reproducibility within
   the jax backend — never against host-path numpy draws bit-for-bit.
   The flat-field fixed pattern, however, IS shared with the bound hcipy
   detector, so noise-*free* detector configurations (photon noise off,
   zero read noise) are deterministic and must match the host path at
   the usual parity bar. So must every chain with no random stage at
   all — that anchor keeps the in-graph tap/norm/convolve math honest.
2. **Batched overrides**: ``int_phot_flux`` / ``convolve_image`` accept
   one value or a leading-batch-dim array, each sample matching what the
   host path produces for that override value.
3. **Loud refusals**: chains with no in-graph equivalent, host-only
   override keys, and the hcipy backend all raise clearly instead of
   silently falling back.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

pytest.importorskip("jax", reason="jax backend requires the optional [jax] extra")

from telescope_sim.abc import PipelineContext, PostProcessor  # noqa: E402
from telescope_sim.config.loader import build  # noqa: E402
from telescope_sim.config.schema import SimConfig  # noqa: E402

DATA = Path(__file__).parent / "data"
ATOL = 1e-12


def _config(outputs: dict, *, focal_extent: float = 2.0) -> SimConfig:
    """The actuator_grid unit YAML with a widened FOV and custom outputs."""
    with open(DATA / "actuator_grid_dm.yaml") as f:
        raw = yaml.safe_load(f)
    raw["focal_planes"]["filter1"]["focal_extent"] = focal_extent
    raw["outputs"] = outputs
    return SimConfig.model_validate(raw)


def _intensity_output(post: list) -> dict:
    return {"tap": {"type": "intensity", "focal_planes": ["filter1"]}, "post_processing": post}


def _noisy_output(*, detector: dict | None = None, **kwargs) -> dict:
    pp: dict = {"type": "noisy_detector", "detector": detector or {}, **kwargs}
    return _intensity_output([pp])


NOISY_FULL = {
    "int_phot_flux": 5.0e7,
    "detector": {"read_noise": 20.0, "dark_current_rate": 100.0, "flat_field": 0.05},
}


@pytest.fixture(scope="module")
def noisy_sim():
    """One norm-only output + one fully-noisy output, jax backend."""
    outputs = {
        "psf": _intensity_output(["max_intensity_norm"]),
        "noisy": _noisy_output(**NOISY_FULL),
    }
    return build(_config(outputs), backend="jax")


@pytest.fixture(scope="module")
def dm_batch() -> np.ndarray:
    return np.random.default_rng(0).normal(size=(4, 8, 8)) * 0.15


# --- Reproducibility + the noise-free anchor ---------------------------------


def test_keymode_reproducible_per_key_and_varying_across_keys_and_samples(noisy_sim, dm_batch):
    one = noisy_sim.sample_batch({"dm": dm_batch}, key=0)
    two = noisy_sim.sample_batch({"dm": dm_batch}, key=0)
    other = noisy_sim.sample_batch({"dm": dm_batch}, key=1)
    for name in one["images"]:
        np.testing.assert_array_equal(one["images"][name], two["images"][name])
    assert np.max(np.abs(one["images"]["noisy"] - other["images"]["noisy"])) > 0
    # Per-sample key split: identical actuations must still get fresh noise.
    same_cmd = np.broadcast_to(dm_batch[0], dm_batch.shape).copy()
    rep = noisy_sim.sample_batch({"dm": same_cmd}, key=2)
    assert np.max(np.abs(rep["images"]["noisy"][0] - rep["images"]["noisy"][1])) > 0


def test_keymode_noise_free_chain_matches_host_mode(noisy_sim, dm_batch):
    """The anchor: a chain with no random stage must reproduce host post.

    Echoes are computed on-device in key-mode (the forward model's
    composed maps), so they match the host readback at fp level, not bit
    level."""
    device = noisy_sim.sample_batch({"dm": dm_batch}, key=0)
    host = noisy_sim.sample_batch({"dm": dm_batch})
    np.testing.assert_allclose(device["images"]["psf"], host["images"]["psf"], rtol=0, atol=ATOL)
    for name in host["actuations"]:
        scale = max(float(np.max(np.abs(host["actuations"][name]))), np.finfo(float).tiny)
        np.testing.assert_allclose(
            device["actuations"][name] / scale,
            host["actuations"][name] / scale,
            rtol=0,
            atol=ATOL,
        )


def test_keymode_deterministic_detector_shares_flat_field_with_host():
    """Photon noise off + zero read noise leaves only the deterministic
    charge model × the flat field — which is the *realized* array drawn by
    the bound hcipy detector, shared between paths by construction."""
    outputs = {
        "noisy": _noisy_output(
            int_phot_flux=1.0e6,
            detector={"include_photon_noise": False, "flat_field": 0.05, "read_noise": 0.0},
        )
    }
    sim = build(_config(outputs), backend="jax")
    cmds = np.random.default_rng(3).normal(size=(3, 8, 8)) * 0.15
    device = sim.sample_batch({"dm": cmds}, key=0)["images"]["noisy"]
    host = sim.sample_batch({"dm": cmds})["images"]["noisy"]
    peak = np.max(np.abs(host))
    np.testing.assert_allclose(device / peak, host / peak, rtol=0, atol=ATOL)
    # The flat field is a real fixed pattern, not the identity.
    flat = np.asarray(sim._c.outputs[0].post_processors[0]._detector.flat_field)
    assert np.std(flat) > 0.01


# --- Statistical validation of the random stages ------------------------------


def test_photon_noise_mean_and_variance_match_the_charge_model():
    outputs = {
        "noisy": _noisy_output(
            int_phot_flux=1.0e6, detector={"flat_field": 0.0}, clamp_nonnegative=False
        )
    }
    sim = build(_config(outputs), backend="jax")
    n = 512
    images = sim.sample_batch({}, batch_size=n, key=42)["images"]["noisy"][..., 0]

    clean = np.asarray(sim.focal_planes["filter1"].reference_psf, dtype=np.float64)
    grid = sim.focal_planes["filter1"].lam_setup.focal_grid
    w = float(np.atleast_1d(np.asarray(grid.weights))[0])
    lam = clean * w
    lam = lam * (1.0e6 * sim.aperture.area / lam.sum())

    # Poisson: mean = var = λ. Mean tested everywhere at 5σ of the
    # estimator; variance on bright pixels at 20%.
    z = np.abs(images.mean(axis=0) - lam) / np.sqrt(np.maximum(lam, 1e-9) / n)
    assert float(z.max()) < 6.0
    bright = lam > 10
    ratio = images.var(axis=0)[bright] / lam[bright]
    assert 0.85 < float(np.median(ratio)) < 1.15


def test_read_noise_variance_and_dark_offset():
    outputs = {
        "noisy": _noisy_output(
            int_phot_flux=None,
            detector={
                "include_photon_noise": False,
                "read_noise": 7.0,
                "dark_current_rate": 50.0,
                "flat_field": 0.0,
            },
            clamp_nonnegative=False,
        )
    }
    sim = build(_config(outputs), backend="jax")
    n = 512
    images = sim.sample_batch({}, batch_size=n, key=7)["images"]["noisy"][..., 0]

    clean = np.asarray(sim.focal_planes["filter1"].reference_psf, dtype=np.float64)
    grid = sim.focal_planes["filter1"].lam_setup.focal_grid
    w = float(np.atleast_1d(np.asarray(grid.weights))[0])
    expected = clean * w + 50.0

    z = np.abs(images.mean(axis=0) - expected) / (7.0 / np.sqrt(n))
    assert float(z.max()) < 6.0
    ratio = images.var(axis=0) / 7.0**2
    assert 0.85 < float(np.median(ratio)) < 1.15


# --- Batched overrides --------------------------------------------------------


def test_batched_int_phot_flux_matches_per_sample_host_overrides(dm_batch):
    """Deterministic detector → each batched-flux sample must equal the
    host path run with that scalar override."""
    outputs = {
        "noisy": _noisy_output(
            int_phot_flux=1.0e6,
            detector={"include_photon_noise": False, "read_noise": 0.0, "flat_field": 0.0},
        )
    }
    sim = build(_config(outputs), backend="jax")
    fluxes = np.array([1.0e5, 1.0e6, 1.0e7, 1.0e8])
    device = sim.sample_batch(
        {"dm": dm_batch}, key=0, output_overrides={"noisy": {"int_phot_flux": fluxes}}
    )["images"]["noisy"]
    for b, flux in enumerate(fluxes):
        host = sim.sample(
            {"dm": dm_batch[b]}, output_overrides={"noisy": {"int_phot_flux": float(flux)}}
        )["images"]["noisy"]
        peak = np.max(np.abs(host))
        np.testing.assert_allclose(device[b] / peak, host / peak, rtol=0, atol=ATOL)


def test_convolve_image_in_graph_with_batched_scenes(dm_batch):
    rng = np.random.default_rng(11)
    scenes = rng.uniform(size=(4, 64, 64))
    outputs = {"ext": _intensity_output([{"type": "convolve_image"}])}
    sim = build(_config(outputs), backend="jax")
    device = sim.sample_batch(
        {"dm": dm_batch}, key=0, output_overrides={"ext": {"convolve_image": scenes}}
    )["images"]["ext"]
    for b in range(4):
        host = sim.sample(
            {"dm": dm_batch[b]}, output_overrides={"ext": {"convolve_image": scenes[b]}}
        )["images"]["ext"]
        peak = np.max(np.abs(host))
        # scipy vs jnp FFT: same math, different libraries — slack tolerance.
        np.testing.assert_allclose(device[b] / peak, host / peak, rtol=0, atol=1e-10)


def test_full_norm_chain_in_graph_matches_host(dm_batch):
    outputs = {"y": _intensity_output(["per_sample_norm", "channels_first"])}
    sim = build(_config(outputs), backend="jax")
    device = sim.sample_batch({"dm": dm_batch}, key=0)["images"]["y"]
    host = sim.sample_batch({"dm": dm_batch})["images"]["y"]
    assert device.shape == (4, 1, 64, 64)  # channels_first applied
    np.testing.assert_allclose(device, host, rtol=0, atol=ATOL)


@pytest.mark.parametrize("method", ["peak", "matched_filter"])
def test_keymode_strehl_on_device_matches_host_mode(method, dm_batch):
    """Both stock estimators translate in-graph; key-mode Strehl must
    match the host-mode batch (computed on the same raw intensities)."""

    outputs = {
        "psf": _intensity_output(["max_intensity_norm"]),
        "noisy": _noisy_output(**NOISY_FULL),
    }
    config = _config(outputs).model_copy(
        update=(
            {"strehl_method": "peak"}
            if method == "peak"
            else {"strehl_method": "matched_filter", "strehl_core_rad": 2.0e-6}
        )
    )
    sim = build(config, backend="jax")
    assert sim.forward_fn().strehl_names == ("filter1",)

    device = sim.sample_batch({"dm": dm_batch}, key=0, meas_strehl=True)["strehls"]["filter1"]
    host = sim.sample_batch({"dm": dm_batch}, meas_strehl=True)["strehls"]["filter1"]
    assert device.shape == (4,)
    assert 0.0 < float(device.min()) <= float(device.max()) <= 1.0 + 1e-9
    np.testing.assert_allclose(device, host, rtol=1e-12)


# --- Refusals -----------------------------------------------------------------


def test_keymode_requires_jax_backend(dm_batch):
    sim = build(_config({"psf": _intensity_output([])}), backend="hcipy")
    with pytest.raises(NotImplementedError, match="backend='jax'"):
        sim.sample_batch({"dm": dm_batch}, key=0)


def test_ineligible_post_processor_is_refused_by_name(dm_batch):
    sim = build(_config({"psf": _intensity_output([])}), backend="jax")

    class Opaque(PostProcessor):
        name = "opaque_custom"

        def __call__(self, image: np.ndarray, context: PipelineContext) -> np.ndarray:
            return image

    sim._c.outputs[0].post_processors.append(Opaque())
    sim.sample_batch({"dm": dm_batch})  # host-side post still works
    with pytest.raises(NotImplementedError, match="opaque_custom"):
        sim.sample_batch({"dm": dm_batch}, key=0)


def test_host_only_and_unknown_overrides_are_refused_in_keymode(noisy_sim, dm_batch):
    with pytest.raises(ValueError, match="cannot be applied on-device"):
        noisy_sim.sample_batch(
            {"dm": dm_batch}, key=0, output_overrides={"psf": {"int_phot_flux": 1.0}}
        )
    with pytest.raises(ValueError, match="unknown output"):
        noisy_sim.sample_batch(
            {"dm": dm_batch}, key=0, output_overrides={"nope": {"int_phot_flux": 1.0}}
        )
    with pytest.raises(ValueError, match="leading batch dimension"):
        noisy_sim.sample_batch(
            {"dm": dm_batch},
            key=0,
            output_overrides={"noisy": {"int_phot_flux": np.ones(3)}},  # batch is 4
        )
