"""Cross-backend parity for the ``vortex`` / ``vector_vortex`` coronagraphs.

The jax backend replays hcipy's multi-scale vortex scheme from the exact
per-level masks the bound hcipy coronagraph precomputes (level 0 as an
FFT filter, finer levels as λ=1 MFT round trips — the vortex phase is
scale-invariant, so one kernel set serves the whole band). The vector
variant runs the circular-basis decomposition of the π-retardance plate:
two half-weight scalar channels at charges ±c. Because geometry is
shared verbatim, any disagreement is propagation numerics — held to the
standard 1e-12 parity bar on max-normalized images.

The golden-digest regression for the real instrument modes (fixtures
07/08/13/14, including the F750 VVC) lives in
``tests/fixtures/test_canonical_jax.py``; this module is the
CI-runnable synthetic-geometry suite.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

pytest.importorskip("jax", reason="jax backend requires the optional [jax] extra")

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_jax_backend_parity import (  # noqa: E402
    ATOL,
    _widen_focal_planes,
    assert_image_parity,
)

from telescope_sim.config.loader import build  # noqa: E402
from telescope_sim.config.schema import SimConfig  # noqa: E402

DATA = Path(__file__).parent / "data"


def _pair(mutate=None):
    with open(DATA / "vortex_zernike.yaml") as f:
        raw = yaml.safe_load(f)
    _widen_focal_planes(raw)
    if mutate is not None:
        mutate(raw)
    config = SimConfig.model_validate(raw)
    return build(config, backend="hcipy"), build(config, backend="jax")


def _to_vector(raw):
    raw["coronagraph"]["type"] = "vector_vortex"
    raw["coronagraph"]["charge"] = 4


@pytest.fixture(scope="module")
def vortex_pair():
    return _pair()


@pytest.fixture(scope="module")
def vvc_pair():
    return _pair(mutate=_to_vector)


@pytest.fixture(scope="module")
def zdm_actuations():
    return {"zdm": np.random.default_rng(42).normal(scale=0.05, size=6)}


@pytest.mark.parametrize("pair_name", ["vortex_pair", "vvc_pair"])
def test_images_match_at_rest_and_under_actuation(pair_name, request, zdm_actuations):
    h, j = request.getfixturevalue(pair_name)
    assert_image_parity(j.sample()["images"]["psf"], h.sample()["images"]["psf"])
    h_img = h.sample(actuations=zdm_actuations)["images"]["psf"]
    j_img = j.sample(actuations=zdm_actuations)["images"]["psf"]
    assert_image_parity(j_img, h_img)


@pytest.mark.parametrize("pair_name", ["vortex_pair", "vvc_pair"])
def test_reference_psfs_match_and_are_coro_free(pair_name, request):
    h, j = request.getfixturevalue(pair_name)
    h_fp = h.focal_planes["filter1"]
    j_fp = j.focal_planes["filter1"]
    np.testing.assert_allclose(
        j_fp.reference_peak_intensity, h_fp.reference_peak_intensity, rtol=1e-12
    )
    assert_image_parity(j_fp.reference_psf, h_fp.reference_psf)
    # The vortex nulls the on-axis star; the reference (coro bypassed) does not.
    for sim, fp in ((h, h_fp), (j, j_fp)):
        rest = sim.sample()["images"]["psf"][..., 0]
        assert rest.max() / fp.reference_peak_intensity < 1e-2


def test_forward_fn_includes_vortex_train(vvc_pair, zdm_actuations):
    h, j = vvc_pair
    fwd = j.forward_fn()
    f_img = np.asarray(fwd(zdm_actuations)["filter1"])
    h_img = h.sample(actuations=zdm_actuations)["images"]["psf"][..., 0]
    j_img = j.sample(actuations=zdm_actuations)["images"]["psf"][..., 0]
    assert_image_parity(f_img, h_img)
    peak = j_img.max()
    np.testing.assert_allclose(f_img / peak, j_img / peak, rtol=0, atol=1e-14)


def test_sample_batch_includes_vortex_train(vvc_pair):
    h, j = vvc_pair
    rng = np.random.default_rng(3)
    batch = {"zdm": rng.normal(scale=0.05, size=(2, 6))}
    j_out = j.sample_batch(batch)
    for b in range(2):
        h_img = h.sample({"zdm": batch["zdm"][b]})["images"]["psf"]
        assert_image_parity(j_out["images"]["psf"][b], h_img)


def test_matched_filter_strehl_parity():
    def _strehl(raw):
        _to_vector(raw)
        raw["strehl_core_rad"] = 2.4e-6

    h, j = _pair(mutate=_strehl)
    acts = {"zdm": np.random.default_rng(5).normal(scale=0.05, size=6)}
    h_s = h.sample(actuations=acts, meas_strehl=True)["strehls"]["filter1"]
    j_s = j.sample(actuations=acts, meas_strehl=True)["strehls"]["filter1"]
    np.testing.assert_allclose(j_s, h_s, rtol=0, atol=ATOL)
