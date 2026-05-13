"""Tests for the fixture digest library."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

# digest_lib lives under fixtures/runner/ (not src/), so add it to the path
RUNNER_DIR = Path(__file__).resolve().parents[2] / "fixtures" / "runner"
sys.path.insert(0, str(RUNNER_DIR))

from digest_lib import (  # noqa: E402
    DIGEST_SCHEMA_VERSION,
    Tolerance,
    all_ok,
    array_digest,
    compare_digest,
    make_digest,
    read_digest,
    write_digest,
)


def test_array_digest_basic_fields():
    arr = np.arange(64, dtype=np.float32).reshape(4, 4, 4)
    d = array_digest(arr)
    assert d["shape"] == [4, 4, 4]
    assert d["dtype"] == "float32"
    assert d["stats"]["min"] == 0.0
    assert d["stats"]["max"] == 63.0
    assert d["stats"]["sum"] == float(np.sum(arr))
    assert d["samples"]["n"] == 32
    assert len(d["samples"]["values"]) == 32


def test_make_digest_roundtrip_self_compares_ok():
    rng = np.random.RandomState(0)
    arrays = {"x": rng.normal(size=(8, 8)), "y": rng.uniform(size=(16,))}
    d = make_digest("test", arrays, rng_seed=42, hcipy_version="0.7.0")
    assert d["schema_version"] == DIGEST_SCHEMA_VERSION
    assert d["fixture_id"] == "test"
    results = compare_digest(arrays, d)
    assert all_ok(results)


def test_compare_detects_shape_mismatch():
    arr = np.zeros((4, 4))
    d = make_digest("t", {"x": arr})
    results = compare_digest({"x": np.zeros((4, 5))}, d)
    assert not all_ok(results)
    assert not results[0].shape_match


def test_compare_detects_value_drift():
    arr = np.ones((8, 8))
    d = make_digest("t", {"x": arr})
    # Perturb beyond tolerance
    perturbed = arr.copy()
    perturbed[0, 0] = 1.5
    results = compare_digest({"x": perturbed}, d)
    assert not all_ok(results)


def test_compare_within_tolerance_passes():
    arr = np.ones((8, 8))
    d = make_digest("t", {"x": arr})
    nudged = arr + 1e-10
    results = compare_digest({"x": nudged}, d)
    assert all_ok(results), [r.notes for r in results]


def test_compare_strict_tolerance_catches_tiny_drift():
    arr = np.ones((8, 8))
    d = make_digest("t", {"x": arr})
    nudged = arr + 1e-4
    tight = Tolerance(stats_rtol=1e-8, stats_atol=1e-12, sample_rtol=1e-8, sample_atol=1e-12)
    results = compare_digest({"x": nudged}, d, tol=tight)
    assert not all_ok(results)


def test_extra_output_in_actual_fails():
    arr = np.zeros((4,))
    d = make_digest("t", {"x": arr})
    results = compare_digest({"x": arr, "y": arr}, d)
    assert not all_ok(results)
    names = {r.output_name for r in results}
    assert "y" in names


def test_missing_output_in_actual_fails():
    arr = np.zeros((4,))
    d = make_digest("t", {"x": arr, "y": arr})
    results = compare_digest({"x": arr}, d)
    assert not all_ok(results)


def test_write_read_roundtrip(tmp_path: Path):
    arr = np.linspace(0, 1, 64).reshape(8, 8)
    d = make_digest("rt", {"x": arr}, rng_seed=1)
    p = tmp_path / "expected.json"
    write_digest(d, p)
    d2 = read_digest(p)
    assert d2 == d
    assert json.loads(p.read_text())["fixture_id"] == "rt"


def test_all_committed_digests_are_well_formed():
    """Every committed digest should be a v1.0 record with at least one output
    and a non-empty fixture id matching the directory name."""
    digests_root = Path(__file__).resolve().parents[2] / "fixtures/runner/digests"
    digest_files = sorted(digests_root.glob("*/expected.json"))
    if not digest_files:
        pytest.skip("no fixture digests captured in this checkout")

    for path in digest_files:
        d = read_digest(path)
        assert d["schema_version"] == DIGEST_SCHEMA_VERSION, f"{path}: bad schema version"
        assert d["fixture_id"] == path.parent.name, f"{path}: fixture_id mismatch"
        assert d["outputs"], f"{path}: no outputs"
        # Every output entry must have shape, dtype, stats, samples
        for name, out in d["outputs"].items():
            assert "shape" in out and "dtype" in out, f"{path}/{name}: missing core fields"
            assert "stats" in out and "samples" in out, f"{path}/{name}: missing stats/samples"
            assert isinstance(out["shape"], list) and len(out["shape"]) >= 1


def test_committed_fiber_digest_specifics():
    """Targeted sanity check for the fiber digest's known shape."""
    digest_path = (
        Path(__file__).resolve().parents[2] / "fixtures/runner/digests/15_fiber_mmf/expected.json"
    )
    if not digest_path.exists():
        pytest.skip("fiber digest not captured in this checkout")
    d = read_digest(digest_path)
    assert d["outputs"]["x"]["shape"] == [2, 128, 128, 1]
