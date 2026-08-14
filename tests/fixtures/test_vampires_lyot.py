"""Validation for fixture 12 (VAMPIRES classical Lyot coronagraph).

There is deliberately no golden digest: the legacy variant this fixture
descends from never produced working output, so there is nothing to
regress against. Validation is cross-backend parity plus physics checks
on the real-instrument geometry instead — the same standard the unit
suite applies to the small synthetic Lyot config, here at full VAMPIRES
scale (256 pupil, 5-wavelength F750 band, parametric SCExAO pupil and
Lyot stop).

Marked ``slow``; needs the local ``test_fixtures`` helpers (skipped when
absent, e.g. on CI).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG = REPO_ROOT / "fixtures/configs/12_vampires_lyot.yaml"

pytestmark = [
    pytest.mark.slow,
    pytest.mark.fixture,
    pytest.mark.skipif(
        not (REPO_ROOT / "test_fixtures/helpers").exists(),
        reason="needs the local test_fixtures helpers (not shipped)",
    ),
]


@pytest.fixture(scope="module")
def hcipy_sim():
    from telescope_sim import TelescopeSim

    return TelescopeSim.from_yaml(CONFIG)


def test_lyot_physics_on_real_geometry(hcipy_sim):
    fp = hcipy_sim.focal_planes["filter1"]
    raw = fp._propagate_chain(
        hcipy_sim._c.correctors, coronagraph=hcipy_sim._c.coronagraph
    ).intensity
    # The CLC-3 spot (~6.5 lam/D) strongly suppresses the core, and the
    # stopped-down pupil rejects most of the total energy. Loose physics
    # bounds, not fitted values.
    assert raw.max() / fp.reference_peak_intensity < 1e-2
    assert raw.sum() / fp.reference_psf_sum < 0.1


@pytest.mark.skipif(
    importlib.util.find_spec("jax") is None,
    reason="backend='jax' needs the optional JAX dependency",
)
def test_cross_backend_parity_on_real_geometry(hcipy_sim):
    from telescope_sim import TelescopeSim

    jax_sim = TelescopeSim.from_yaml(CONFIG, backend="jax")
    acts = {"zernike_dm": np.random.default_rng(42).normal(scale=0.05, size=35)}

    h_img = hcipy_sim.sample(actuations=acts)["images"]["psf"]
    j_img = jax_sim.sample(actuations=acts)["images"]["psf"]
    peak = np.max(np.abs(h_img))
    np.testing.assert_allclose(j_img / peak, h_img / peak, rtol=0, atol=1e-12)

    h_fp = hcipy_sim.focal_planes["filter1"]
    j_fp = jax_sim.focal_planes["filter1"]
    np.testing.assert_allclose(
        j_fp.reference_peak_intensity, h_fp.reference_peak_intensity, rtol=1e-12
    )
