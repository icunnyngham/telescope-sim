"""Cross-backend parity for the classical ``lyot`` coronagraph.

The jax backend folds the Lyot train (pupil → small-mask-grid MFT →
Babinet subtraction → Lyot stop) into the per-wavelength propagation
kernels; hcipy propagates through ``hcipy.LyotCoronagraph``. Both
backends share the exact same geometry arrays (mask grid, supersampled
occulter, resolved stop field), so any disagreement is propagation
numerics — held to the standard 1e-12 parity bar on max-normalized
images (observed ~1e-15).

Also covered: the reference PSF stays coronagraph-free on both backends,
``forward_fn`` / ``sample_batch`` include the Lyot train in-graph, and
the config-time gates (vortex still refused on jax; complex stops
refused by the kernel builder).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

pytest.importorskip("jax", reason="jax backend requires the optional [jax] extra")

# Sibling-module import that works under the bare ``pytest`` entry point
# (no repo root on sys.path) — same pattern as tests/fixtures.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_jax_backend_parity import (  # noqa: E402
    ATOL,
    _widen_focal_planes,
    assert_image_parity,
)

from telescope_sim.backends.jax.propagation import FraunhoferMFT  # noqa: E402
from telescope_sim.config.loader import build  # noqa: E402
from telescope_sim.config.schema import SimConfig  # noqa: E402

DATA = Path(__file__).parent / "data"


def _pair(mutate=None):
    with open(DATA / "lyot_zernike.yaml") as f:
        raw = yaml.safe_load(f)
    _widen_focal_planes(raw)
    if mutate is not None:
        mutate(raw)
    config = SimConfig.model_validate(raw)
    return build(config, backend="hcipy"), build(config, backend="jax")


@pytest.fixture(scope="module")
def lyot_pair():
    return _pair()


@pytest.fixture(scope="module")
def zdm_actuations():
    return {"zdm": np.random.default_rng(42).normal(scale=0.05, size=6)}


def test_reference_psfs_match_and_are_coro_free(lyot_pair):
    h, j = lyot_pair
    h_fp = h.focal_planes["filter1"]
    j_fp = j.focal_planes["filter1"]
    np.testing.assert_allclose(
        j_fp.reference_peak_intensity, h_fp.reference_peak_intensity, rtol=1e-12
    )
    assert_image_parity(j_fp.reference_psf, h_fp.reference_psf)
    # Reference bypasses the coronagraph: the at-rest science image must be
    # far dimmer than the reference peak on BOTH backends.
    for sim, fp in ((h, h_fp), (j, j_fp)):
        rest = sim.sample()["images"]["psf"][..., 0]
        assert rest.max() / fp.reference_peak_intensity < 1e-2


def test_images_match_at_rest(lyot_pair):
    h, j = lyot_pair
    assert_image_parity(j.sample()["images"]["psf"], h.sample()["images"]["psf"])


def test_broadband_images_match_under_actuation(lyot_pair, zdm_actuations):
    h, j = lyot_pair
    h_img = h.sample(actuations=zdm_actuations)["images"]["psf"]
    j_img = j.sample(actuations=zdm_actuations)["images"]["psf"]
    assert_image_parity(j_img, h_img)


def test_forward_fn_includes_lyot_train(lyot_pair, zdm_actuations):
    h, j = lyot_pair
    fwd = j.forward_fn()
    f_img = np.asarray(fwd(zdm_actuations)["filter1"])
    h_img = h.sample(actuations=zdm_actuations)["images"]["psf"][..., 0]
    j_img = j.sample(actuations=zdm_actuations)["images"]["psf"][..., 0]
    assert_image_parity(f_img, h_img)
    # forward_fn and eager jax sampling share the same jitted kernels.
    peak = j_img.max()
    np.testing.assert_allclose(f_img / peak, j_img / peak, rtol=0, atol=1e-14)


def test_sample_batch_includes_lyot_train(lyot_pair):
    h, j = lyot_pair
    rng = np.random.default_rng(3)
    batch = {"zdm": rng.normal(scale=0.05, size=(2, 6))}
    j_out = j.sample_batch(batch)
    for b in range(2):
        h_img = h.sample({"zdm": batch["zdm"][b]})["images"]["psf"]
        assert_image_parity(j_out["images"]["psf"][b], h_img)


def test_matched_filter_strehl_parity():
    def _strehl(raw):
        raw["strehl_core_rad"] = 2.4e-6 / 1.0  # ~2.4 λ/D core at λ=1 µm, D=1 m

    h, j = _pair(mutate=_strehl)
    acts = {"zdm": np.random.default_rng(5).normal(scale=0.05, size=6)}
    h_s = h.sample(actuations=acts, meas_strehl=True)["strehls"]["filter1"]
    j_s = j.sample(actuations=acts, meas_strehl=True)["strehls"]["filter1"]
    np.testing.assert_allclose(j_s, h_s, rtol=0, atol=ATOL)


def test_hcipy_only_coronagraph_refused_on_jax():
    """The supported_backends config-time gate still refuses hcipy-only kinds."""
    from telescope_sim.abc import Coronagraph
    from telescope_sim.registry import register, registry

    if "hcipy_only_test_coro" not in registry["coronagraph"]:

        @register("coronagraph", "hcipy_only_test_coro")
        class HcipyOnlyCoro(Coronagraph):
            name = "hcipy_only_test_coro"
            supported_backends = frozenset({"hcipy"})

            def apply(self, wf):
                return wf

    def _swap(raw):
        raw["coronagraph"] = {"type": "hcipy_only_test_coro"}

    with pytest.raises(ValueError, match="not supported on the 'jax' backend"):
        _pair(mutate=_swap)


def test_complex_lyot_stop_refused_by_kernel_builder(lyot_pair):
    _, j = lyot_pair
    fp = j.focal_planes["filter1"]
    coro = j._c.coronagraph

    class ComplexStop:
        name = "lyot"
        mask_grid = coro.mask_grid
        occulter = coro.occulter
        lyot_field = np.asarray(coro.lyot_field) * (1.0 + 0.1j)
        focal_length = 1.0

    with pytest.raises(ValueError, match="complex Lyot-stop"):
        FraunhoferMFT(
            j._c.pupil_grid,
            fp.lam_setup.focal_grid,
            fp.lam_setup.filter_lams,
            coronagraph=ComplexStop(),
        )
