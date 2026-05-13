"""Regression tests: v2 pipeline reproduces the canonical-family digests.

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

from digest_lib import Tolerance, all_ok, compare_digest, read_digest  # noqa: E402


CANONICAL_FIXTURES = [
    "01_canonical_2024-09",
    "02_phase_workhorse",
    "10_near_canonical_A",
    "11_near_canonical_B",
]


@pytest.fixture(scope="module")
def telescope_sim_module():
    """Importing TelescopeSim is non-trivial (drags in HCIPy); cache per module."""
    import telescope_sim
    return telescope_sim


@pytest.mark.slow
@pytest.mark.fixture
@pytest.mark.parametrize("fixture_id", CANONICAL_FIXTURES)
def test_canonical_v2_reproduces_digest(
    fixture_id: str, telescope_sim_module
) -> None:
    """Build the v2 pipeline from fixtures/configs/<fixture_id>.yaml and compare."""
    config_path = REPO_ROOT / "fixtures/configs" / f"{fixture_id}.yaml"
    digest_path = (
        REPO_ROOT / "fixtures/runner/digests" / fixture_id / "expected.json"
    )
    assert config_path.is_file(), f"missing v2 config at {config_path}"
    assert digest_path.is_file(), f"missing digest at {digest_path}"

    sim = telescope_sim_module.TelescopeSim.from_yaml(config_path)
    np.random.seed(42)
    out = sim.sample(meas_strehl=True)

    actuator_name = next(iter(out["actuations"]))
    actual = {
        "x": np.asarray(out["images"]["psf"]),
        "y": np.asarray(out["actuations"][actuator_name]),
    }
    if out.get("strehls"):
        actual["strehls"] = np.array(list(out["strehls"].values()))

    recorded = read_digest(digest_path)
    results = compare_digest(actual, recorded)
    assert all_ok(results), "\n".join(str(r) for r in results)
