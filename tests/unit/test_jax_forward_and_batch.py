"""forward_fn + sample_batch: the pure forward model and the batch sampler.

``forward_fn()`` collapses the actuate/impose corrector chain into
precomputed linear maps and reuses the focal planes' jitted MFT kernels,
so it must reproduce the hcipy backend's raw propagation at the same
1e-12 parity bar the jax ``sample()`` path is held to — plus behave as a
*pure function*: stage decomposition must be consistent, vmap/grad must
compose, and every unsupported configuration must be refused loudly.

``sample_batch()`` is the reference composition: one vmapped device
dispatch for propagation, then the exact per-sample host code
``sample()`` uses for taps/post/echo/Strehl. The load-bearing assertion
is therefore *batch ≡ loop-of-sample()* on the same sim, element for
element — semantics must not fork between the two entry points. The
hcipy fallback is literally a loop over ``sample()``, checked against
the jax path at the cross-backend parity bar.

Also covered here: the ``precision: float32`` config knob, the skipped
(unused) hcipy propagator construction on jax focal planes, and the
complex-aperture guard.
"""

from __future__ import annotations

from pathlib import Path

import hcipy
import numpy as np
import pytest
import yaml

pytest.importorskip("jax", reason="jax backend requires the optional [jax] extra")

import jax  # noqa: E402

from telescope_sim import TelescopeSim  # noqa: E402
from telescope_sim.backends.jax.focal_planes import JaxAngularFocalPlane  # noqa: E402
from telescope_sim.config.loader import build  # noqa: E402
from telescope_sim.config.schema import SimConfig  # noqa: E402

DATA = Path(__file__).parent / "data"

ATOL = 1e-12


def assert_image_parity(img, ref, *, atol=ATOL):
    """Compare images after scaling by the reference peak (see the jax parity suite)."""
    ref = np.asarray(ref, dtype=np.float64)
    img = np.asarray(img, dtype=np.float64)
    assert img.shape == ref.shape
    peak = np.max(np.abs(ref))
    assert peak > 0
    np.testing.assert_allclose(img / peak, ref / peak, rtol=0, atol=atol)


def _config_from_yaml(path, mutate=None) -> SimConfig:
    with open(path) as f:
        raw = yaml.safe_load(f)
    if mutate is not None:
        mutate(raw)
    return SimConfig.model_validate(raw)


def _widen_focal_planes(raw: dict, focal_extent: float = 2.0) -> None:
    """The stock unit YAMLs have a degenerate 1e-5-arcsec FOV; widen it so
    image assertions actually see PSF structure (see the jax parity suite)."""
    for fp in raw["focal_planes"].values():
        if fp["type"] == "angular":
            fp["focal_extent"] = focal_extent


@pytest.fixture(scope="module")
def elf_pair() -> tuple[TelescopeSim, TelescopeSim]:
    return (
        TelescopeSim.from_preset("elf_15seg"),
        TelescopeSim.from_preset("elf_15seg", backend="jax"),
    )


@pytest.fixture(scope="module")
def ptt_batch() -> np.ndarray:
    """Seeded batch of 3 piston/tip/tilt command sets for the 15 segments."""
    return np.random.default_rng(2).normal(size=(3, 15, 3)) * 0.2


# --- forward_fn: parity with raw propagation ---------------------------------


def test_forward_matches_hcipy_raw_propagation(elf_pair):
    """The headline contract: fwd(actuations) == hcipy's per-λ apply() chain."""
    hcipy_sim, jax_sim = elf_pair
    fwd = jax_sim.forward_fn()
    assert fwd.corrector_names == ("segments",)
    assert fwd.n_actuators == {"segments": 45}
    assert set(fwd.focal_plane_names) == set(hcipy_sim.focal_planes)

    acts = np.random.default_rng(0).normal(size=(15, 3)) * 0.2
    out = fwd({"segments": acts})
    for c in hcipy_sim._c.correctors:
        c.set_actuators(acts)
    for name, fp in hcipy_sim.focal_planes.items():
        assert_image_parity(out[name], fp._propagate_chain(hcipy_sim._c.correctors).intensity)


def test_forward_missing_keys_mean_flat(elf_pair):
    """No actuations → the at-rest chain, i.e. the reference PSF."""
    _, jax_sim = elf_pair
    out = jax_sim.forward_fn()({})
    for name, fp in jax_sim.focal_planes.items():
        assert_image_parity(out[name], fp.reference_psf)


def test_forward_stage_decomposition(elf_pair):
    """__call__ must be exactly intensity_from_opd ∘ opd_from_actuations,
    and the OPD stage must agree with the mirror-surface bookkeeping
    sample() uses (2 × surface per corrector)."""
    _, jax_sim = elf_pair
    fwd = jax_sim.forward_fn()
    acts = np.random.default_rng(1).normal(size=(15, 3)) * 0.2

    opd = np.asarray(fwd.opd_from_actuations({"segments": acts}))
    assert opd.shape == (jax_sim._c.pupil_grid.size,)
    corrector = jax_sim._c.correctors[0]
    corrector.set_actuators(acts)
    expected = 2.0 * np.asarray(corrector._sm.surface)
    scale = np.max(np.abs(expected))
    assert scale > 0
    np.testing.assert_allclose(opd / scale, expected / scale, rtol=0, atol=ATOL)

    via_stages = fwd.intensity_from_opd(opd)
    direct = fwd({"segments": acts})
    for name in direct:
        np.testing.assert_array_equal(np.asarray(via_stages[name]), np.asarray(direct[name]))


def test_forward_intensity_from_opd_is_the_external_opd_hook(elf_pair):
    """Adding screen OPD to actuation OPD must equal sample(atmos=screen)'s
    propagation — the seam batched atmosphere providers will plug into."""
    hcipy_sim, jax_sim = elf_pair
    fwd = jax_sim.forward_fn()
    grid = hcipy_sim._c.pupil_grid
    basis = hcipy.make_zernike_basis(5, 1.2, grid, starting_mode=2)
    rng = np.random.default_rng(3)
    screen_opd = sum(rng.normal() * np.asarray(m, dtype=np.float64) for m in basis)
    screen_opd *= 5e-8 / np.std(screen_opd)

    class Screen:
        def phase_for(self, lam):
            return 2.0 * np.pi * screen_opd / float(lam)

        def __call__(self, wf):
            field = hcipy.Field(
                np.asarray(wf.electric_field) * np.exp(1j * self.phase_for(wf.wavelength)), wf.grid
            )
            return hcipy.Wavefront(field, wf.wavelength)

    acts = np.random.default_rng(4).normal(size=(15, 3)) * 0.2
    out = fwd.intensity_from_opd(fwd.opd_from_actuations({"segments": acts}) + screen_opd)
    for c in hcipy_sim._c.correctors:
        c.set_actuators(acts)
    for name, fp in hcipy_sim.focal_planes.items():
        raw = fp._propagate_chain(hcipy_sim._c.correctors, atmos=Screen()).intensity
        assert_image_parity(out[name], raw)


def test_forward_actuator_grid_with_flips_and_rotation():
    """Command-indexing flips + rotated influence geometry survive the
    numeric caller→dm probe (they are permutations, i.e. linear)."""

    def mutate(raw):
        _widen_focal_planes(raw)
        raw["correctors"]["dm"]["flip_x"] = True
        raw["correctors"]["dm"]["flip_y"] = True

    config = _config_from_yaml(DATA / "actuator_grid_dm.yaml", mutate)
    hcipy_sim = build(config, backend="hcipy")
    jax_sim = build(config, backend="jax")
    cmd = np.random.default_rng(5).normal(size=(8, 8)) * 0.15

    out = jax_sim.forward_fn()({"dm": cmd})
    for c in hcipy_sim._c.correctors:
        c.set_actuators(cmd)
    raw_img = hcipy_sim.focal_planes["filter1"]._propagate_chain(hcipy_sim._c.correctors).intensity
    assert_image_parity(out["filter1"], raw_img)


def test_forward_vmap_and_grad_compose(elf_pair, ptt_batch):
    """The ML story: vmap over a batch equals the per-sample loop, and
    grad through the full forward is finite."""
    _, jax_sim = elf_pair
    fwd = jax_sim.forward_fn()

    batched = jax.vmap(fwd)({"segments": ptt_batch})
    for b in range(ptt_batch.shape[0]):
        single = fwd({"segments": ptt_batch[b]})
        for name in single:
            assert_image_parity(np.asarray(batched[name][b]), np.asarray(single[name]))

    def loss(acts):
        return fwd({"segments": acts})["filter1"].sum()

    grads = np.asarray(jax.grad(loss)(np.zeros((15, 3))))
    assert grads.shape == (15, 3)
    assert np.all(np.isfinite(grads))


# --- forward_fn: refusals -----------------------------------------------------


def test_forward_fn_requires_jax_backend(elf_pair):
    hcipy_sim, _ = elf_pair
    with pytest.raises(NotImplementedError, match="backend='jax'"):
        hcipy_sim.forward_fn()


def test_forward_fn_composes_fit_role_chains():
    """dm3 (fit-role, cumulative source) must cancel dm1+dm2 *inside the
    graph*: the composed-fit maps reproduce sample()'s host least squares,
    so the image returns to the reference PSF."""
    config = _config_from_yaml(DATA / "three_zernike_residual_fit.yaml", _widen_focal_planes)
    hcipy_sim = build(config, backend="hcipy")
    jax_sim = build(config, backend="jax")
    fwd = jax_sim.forward_fn()
    assert fwd.corrector_names == ("dm1", "dm2")  # the fit corrector is not an input
    assert fwd.echo_names == ("dm3",)

    rng = np.random.default_rng(11)
    acts = {"dm1": rng.normal(size=8), "dm2": rng.normal(size=8)}
    img = np.asarray(fwd(acts)["filter1"])
    assert_image_parity(img, hcipy_sim.sample(acts)["images"]["psf"][..., 0])
    ref = jax_sim.focal_planes["filter1"].reference_psf
    np.testing.assert_allclose(img / ref.max(), ref / ref.max(), rtol=0, atol=1e-6)

    # Fit-role correctors take no caller actuations — same as sample().
    with pytest.raises(ValueError, match="unknown corrector"):
        fwd({"dm3": np.zeros(8)})


def test_forward_actuation_echo_matches_sample_including_named_fit_source():
    """All three echo strategies + a named fit_source, against sample()."""

    def mutate(raw):
        _widen_focal_planes(raw)
        raw["correctors"]["dm2"]["target_strategy"] = "residual_fit_only"
        raw["correctors"]["dm2"]["target"] = True
        raw["correctors"]["dm3"]["target_strategy"] = "actuators_plus_residual_fit"
        raw["correctors"]["dm3"]["fit_source"] = "dm1"

    config = _config_from_yaml(DATA / "three_zernike_residual_fit.yaml", mutate)
    jax_sim = build(config, backend="jax")
    fwd = jax_sim.forward_fn()
    rng = np.random.default_rng(13)
    acts = {"dm1": rng.normal(size=8), "dm2": rng.normal(size=8)}

    device = {k: np.asarray(v) for k, v in fwd.actuation_echo(acts).items()}
    host = jax_sim.sample(acts)["actuations"]
    assert set(device) == set(host)
    for name in host:
        scale = np.max(np.abs(host[name]))
        assert scale > 0
        np.testing.assert_allclose(device[name] / scale, host[name] / scale, rtol=0, atol=ATOL)
    assert_image_parity(
        np.asarray(fwd(acts)["filter1"]), jax_sim.sample(acts)["images"]["psf"][..., 0]
    )


def test_forward_fn_rejects_nonlinear_set_actuators():
    """The build-time probe must catch a corrector whose set_actuators is
    not linear in the caller values instead of silently mismatching."""
    config = _config_from_yaml(DATA / "actuator_grid_dm.yaml", _widen_focal_planes)
    jax_sim = build(config, backend="jax")
    corrector = jax_sim._c.correctors[0]
    original = corrector.set_actuators
    corrector.set_actuators = lambda values: original(np.square(np.asarray(values, dtype=float)))
    with pytest.raises(ValueError, match="not linear"):
        jax_sim.forward_fn()


def test_forward_actuation_key_and_shape_validation(elf_pair):
    _, jax_sim = elf_pair
    fwd = jax_sim.forward_fn()
    with pytest.raises(ValueError, match="unknown corrector"):
        fwd({"nope": np.zeros(45)})
    with pytest.raises(ValueError, match="expected 45"):
        fwd({"segments": np.zeros(7)})


# --- sample_batch -------------------------------------------------------------


def test_sample_batch_equals_sample_loop_on_jax(elf_pair, ptt_batch):
    """The semantics anchor: batch ≡ loop of sample() on the same sim."""
    _, jax_sim = elf_pair
    batch = jax_sim.sample_batch({"segments": ptt_batch}, meas_strehl=True)
    singles = [jax_sim.sample({"segments": a}, meas_strehl=True) for a in ptt_batch]

    assert batch["images"]["psf"].shape == (3, 128, 128, 2)
    for b, single in enumerate(singles):
        assert_image_parity(batch["images"]["psf"][b], single["images"]["psf"])
        np.testing.assert_array_equal(
            batch["actuations"]["segments"][b], single["actuations"]["segments"]
        )
        for name, value in single["strehls"].items():
            np.testing.assert_allclose(batch["strehls"][name][b], value, rtol=1e-12)


def test_sample_batch_hcipy_fallback_matches_jax(elf_pair, ptt_batch):
    """Images + echoes across backends. Strehl is deliberately not compared
    here: the preset uses the ``peak`` method, whose reference-argmax is
    4-way degenerate on this even grid, so the backends may legitimately
    read different pixels (see the jax parity suite's Strehl note);
    within-backend batch ≡ loop Strehl is asserted above."""
    hcipy_sim, jax_sim = elf_pair
    h_batch = hcipy_sim.sample_batch({"segments": ptt_batch})
    d_batch = jax_sim.sample_batch({"segments": ptt_batch})
    assert h_batch["images"]["psf"].shape == d_batch["images"]["psf"].shape
    assert_image_parity(d_batch["images"]["psf"], h_batch["images"]["psf"])
    np.testing.assert_array_equal(
        d_batch["actuations"]["segments"], h_batch["actuations"]["segments"]
    )


def test_sample_batch_residual_fit_echo():
    """Residual-fit echo strategies exercise the host-side cumulative-OPD
    recomputation; both backends' batches must agree with their own
    per-sample loops (echoes are shared numpy → exact)."""

    def mutate(raw):
        _widen_focal_planes(raw)
        raw["correctors"]["dm"]["target_strategy"] = "actuators_plus_residual_fit"

    config = _config_from_yaml(DATA / "actuator_grid_dm.yaml", mutate)
    commands = np.random.default_rng(6).normal(size=(3, 8, 8)) * 0.15
    for backend in ("hcipy", "jax"):
        sim = build(config, backend=backend)
        batch = sim.sample_batch({"dm": commands})
        singles = [sim.sample({"dm": c}) for c in commands]
        for b, single in enumerate(singles):
            np.testing.assert_array_equal(batch["actuations"]["dm"][b], single["actuations"]["dm"])
        assert np.max(np.abs(batch["actuations"]["dm"])) > 0


def test_sample_batch_empty_actuations_needs_batch_size(elf_pair):
    _, jax_sim = elf_pair
    with pytest.raises(ValueError, match="batch_size"):
        jax_sim.sample_batch({})
    batch = jax_sim.sample_batch({}, batch_size=2)
    rest = jax_sim.sample()["images"]["psf"]
    assert batch["images"]["psf"].shape == (2, *rest.shape)
    for b in range(2):
        assert_image_parity(batch["images"]["psf"][b], rest)


def test_sample_batch_input_validation(elf_pair, ptt_batch):
    _, jax_sim = elf_pair
    with pytest.raises(ValueError, match="unknown corrector"):
        jax_sim.sample_batch({"nope": ptt_batch})
    with pytest.raises(ValueError, match="leading batch dimension"):
        jax_sim.sample_batch({"segments": ptt_batch[0]})  # a single sample, no batch axis
    with pytest.raises(ValueError, match="disagrees"):
        jax_sim.sample_batch({"segments": ptt_batch}, batch_size=5)


def test_sample_batch_fit_role_chain_matches_loop_on_both_backends():
    """Fit-role chains through the batch paths: batch ≡ loop of sample(),
    with the fit resolved in-graph (propagation) and host-side (echo)."""
    config = _config_from_yaml(DATA / "three_zernike_residual_fit.yaml", _widen_focal_planes)
    acts = {"dm1": np.random.default_rng(7).normal(size=(2, 8))}
    for backend in ("hcipy", "jax"):
        sim = build(config, backend=backend)
        batch = sim.sample_batch(acts)
        singles = [sim.sample({"dm1": acts["dm1"][b]}) for b in range(2)]
        for b, single in enumerate(singles):
            assert_image_parity(batch["images"]["psf"][b], single["images"]["psf"])
            np.testing.assert_array_equal(
                batch["actuations"]["dm3"][b], single["actuations"]["dm3"]
            )
        if backend == "jax":
            # Key-mode: propagation, post, AND echoes on device.
            keyed = sim.sample_batch(acts, key=0)
            for b, single in enumerate(singles):
                assert_image_parity(keyed["images"]["psf"][b], single["images"]["psf"])
                scale = np.max(np.abs(single["actuations"]["dm3"]))
                np.testing.assert_allclose(
                    keyed["actuations"]["dm3"][b] / scale,
                    single["actuations"]["dm3"] / scale,
                    rtol=0,
                    atol=ATOL,
                )


# --- precision knob -----------------------------------------------------------


def test_precision_float32():
    """float32 kernels: same physics at single-precision accuracy, float32 out."""
    mutate64 = _widen_focal_planes

    def mutate32(raw):
        _widen_focal_planes(raw)
        raw["precision"] = "float32"

    sim64 = build(_config_from_yaml(DATA / "actuator_grid_dm.yaml", mutate64), backend="jax")
    sim32 = build(_config_from_yaml(DATA / "actuator_grid_dm.yaml", mutate32), backend="jax")
    assert sim32.focal_planes["filter1"].reference_psf.dtype == np.float32

    cmd = np.random.default_rng(8).normal(size=(8, 8)) * 0.15
    img64 = sim64.sample({"dm": cmd})["images"]["psf"]
    img32 = sim32.sample({"dm": cmd})["images"]["psf"]
    assert img32.dtype == np.float32
    assert_image_parity(img32, img64, atol=1e-4)

    fwd32 = sim32.forward_fn()({"dm": cmd})
    assert np.asarray(fwd32["filter1"]).dtype == np.float32


def test_precision_float32_requires_jax_backend():
    def mutate(raw):
        raw["precision"] = "float32"

    config = _config_from_yaml(DATA / "actuator_grid_dm.yaml", mutate)
    with pytest.raises(ValueError, match="requires backend='jax'"):
        build(config, backend="hcipy")


# --- build-path cleanups ------------------------------------------------------


def test_jax_focal_planes_skip_hcipy_propagator(elf_pair):
    """The jax planes propagate through their own kernels; the (unused)
    hcipy propagator + wavefront construction is skipped at build time."""
    hcipy_sim, jax_sim = elf_pair
    for fp in jax_sim.focal_planes.values():
        assert fp.lam_setup.propagator is None
        assert fp.lam_setup.wavefronts == []
        with pytest.raises(NotImplementedError, match="hcipy-backend only"):
            fp.propagate(None)
    for fp in hcipy_sim.focal_planes.values():
        assert fp.lam_setup.propagator is not None
        assert len(fp.lam_setup.wavefronts) == fp.num_samples


def test_complex_aperture_is_rejected():
    grid = hcipy.make_pupil_grid(32, 1.0)
    field = hcipy.Field(np.ones(grid.size, dtype=np.complex128), grid)
    fp = JaxAngularFocalPlane(central_lam=1.0e-6, focal_extent=1.0, focal_res=16)
    with pytest.raises(ValueError, match="complex aperture"):
        fp.build(grid, field)
