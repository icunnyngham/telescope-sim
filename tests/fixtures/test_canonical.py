"""Regression tests: v2 pipeline reproduces the committed legacy digests.

Marked ``slow`` since each test runs the full optical chain end-to-end.
Run with ``pytest --runslow tests/fixtures/`` (or ``make test-slow``).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "fixtures/runner"))

from digest_lib import all_ok, compare_digest, read_digest  # noqa: E402

CANONICAL_FIXTURES = [
    "01_canonical_2024-09",
    "02_phase_workhorse",
    "03_multi_aperture_dm_psf",
    "07_coro_original",
    "08_ffcoro_2023-02",
    "09_vampires_base",
    "10_near_canonical_A",
    "11_near_canonical_B",
    "13_vvc_flexible",
    "14_fp_rl_ff_vvc",
    # 15_fiber_mmf carries an extra ``fiber`` mark — it dominates the
    # --runslow runtime (HCIPy LP-fiber mode solve is slow) and is not
    # run on GitHub CI. Opt in locally with --runfiber.
    pytest.param("15_fiber_mmf", marks=pytest.mark.fiber),
]

# Fixture #16 is shaped differently (Strehl arrays over an actuation
# sequence, no single-sample x/y) so it has its own runner + test below.


@pytest.fixture(scope="module")
def telescope_sim_module():
    """Importing TelescopeSim is non-trivial (drags in HCIPy); cache per module."""
    import telescope_sim

    return telescope_sim


@pytest.mark.slow
@pytest.mark.fixture
@pytest.mark.parametrize("fixture_id", CANONICAL_FIXTURES)
def test_canonical_v2_reproduces_digest(fixture_id: str, telescope_sim_module) -> None:
    """Build the v2 pipeline from fixtures/configs/<fixture_id>.yaml and compare."""
    config_path = REPO_ROOT / "fixtures/configs" / f"{fixture_id}.yaml"
    digest_path = REPO_ROOT / "fixtures/runner/digests" / fixture_id / "expected.json"
    assert config_path.is_file(), f"missing v2 config at {config_path}"
    assert digest_path.is_file(), f"missing digest at {digest_path}"

    recorded = read_digest(digest_path)
    expected_outputs = set(recorded["outputs"])

    sim = telescope_sim_module.TelescopeSim.from_yaml(config_path)
    np.random.seed(42)
    out = sim.sample(meas_strehl=("strehls" in expected_outputs))

    image_name = next(iter(out["images"]))
    actuator_name = next(iter(out["actuations"]))
    actual = {
        "x": np.asarray(out["images"][image_name]),
        "y": np.asarray(out["actuations"][actuator_name]),
    }
    if "strehls" in expected_outputs and out.get("strehls"):
        actual["strehls"] = np.array(list(out["strehls"].values()))

    results = compare_digest(actual, recorded)
    assert all_ok(results), "\n".join(str(r) for r in results)


@pytest.mark.slow
@pytest.mark.fixture
def test_noisy_psf_v2_reproduces_digest(telescope_sim_module) -> None:  # noqa: ARG001
    """Fixture #17: v2 reproduces seeded-noise legacy outputs bit-for-bit.

    Pins the v2 ``NoisyIntensityOutputTap`` (single-call Field-based
    design) against the legacy ``_addNoiseToObservation`` (Wavefront-
    reconstruction). Three configurations (at-rest + two photon fluxes,
    plus a tip-tilt + flux case) captured under ``np.random.seed(42)``.

    Detector construction is forced eager at sim-build time so the
    flat-field RNG-burn matches the legacy's sampler-init timing.
    """
    fixture_id = "17_noisy_psf"
    digest_path = REPO_ROOT / "fixtures/runner/digests" / fixture_id / "expected.json"
    assert digest_path.is_file(), f"missing digest at {digest_path}"

    from run_v2_17_noisy_psf import reproduce  # noqa: PLC0415

    recorded = read_digest(digest_path)
    actual = reproduce()

    results = compare_digest(actual, recorded)
    assert all_ok(results), "\n".join(str(r) for r in results)


@pytest.mark.slow
@pytest.mark.fixture
def test_strehl_zernike_v2_reproduces_digest(telescope_sim_module) -> None:  # noqa: ARG001
    """Fixture #16: v2 reproduces both Strehl modes on an 8-case actuation sweep.

    Pins the v2 ``StrehlEstimator`` formulas against the legacy
    ``_strehl`` formulas captured in
    ``fixtures/runner/digests/16_strehl_zernike/expected.json``.
    """
    fixture_id = "16_strehl_zernike"
    digest_path = REPO_ROOT / "fixtures/runner/digests" / fixture_id / "expected.json"
    assert digest_path.is_file(), f"missing digest at {digest_path}"

    # Lazy import: the runner script is in fixtures/runner/, which is on
    # sys.path from the module top.
    from run_v2_16_strehl_zernike import reproduce  # noqa: PLC0415

    recorded = read_digest(digest_path)
    actual = reproduce()

    results = compare_digest(actual, recorded)
    assert all_ok(results), "\n".join(str(r) for r in results)
