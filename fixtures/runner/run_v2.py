"""Run a v2 reproduction of a fixture and compare against its committed digest.

Usage::

    python fixtures/runner/run_v2.py <fixture_id>

For example::

    python fixtures/runner/run_v2.py 01_canonical_2024-09

Looks up ``fixtures/configs/<fixture_id>.yaml``, builds the v2 pipeline,
runs ``sample(meas_strehl=True)`` with seed 42, and compares each output
against ``fixtures/runner/digests/<fixture_id>/expected.json``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "fixtures/runner"))

from digest_lib import Tolerance, all_ok, compare_digest, read_digest  # noqa: E402


def reproduce(
    fixture_id: str, *, expected_outputs: set[str] | None = None
) -> dict[str, np.ndarray]:
    """Build the v2 pipeline for a fixture and return outputs matching the digest.

    Parameters
    ----------
    fixture_id
        Fixture identifier.
    expected_outputs
        If given, only those output keys are included (matches the digest's
        ``outputs`` dict). Avoids spurious failures when v2 includes outputs
        the legacy capture didn't record (e.g. strehls when the legacy
        sample() didn't accept ``meas_strehl``).
    """
    from telescope_sim import TelescopeSim

    config_path = REPO_ROOT / "fixtures/configs" / f"{fixture_id}.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(f"no v2 config for {fixture_id} at {config_path}")

    want_strehls = expected_outputs is None or "strehls" in expected_outputs
    sim = TelescopeSim.from_yaml(config_path)
    np.random.seed(42)
    out = sim.sample(meas_strehl=want_strehls)

    # Map the v2 output dict to the legacy (x, y[, strehls]) digest layout.
    # If multiple images are produced, the first key wins (the canonical
    # path has exactly one).
    image_name = next(iter(out["images"]))
    actuator_name = next(iter(out["actuations"]))
    outputs: dict[str, np.ndarray] = {
        "x": np.asarray(out["images"][image_name]),
        "y": np.asarray(out["actuations"][actuator_name]),
    }
    strehls = out.get("strehls")
    if strehls and want_strehls:
        outputs["strehls"] = np.array(list(strehls.values()))

    if expected_outputs is not None:
        outputs = {k: v for k, v in outputs.items() if k in expected_outputs}
    return outputs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("fixture_id", help="e.g. 01_canonical_2024-09")
    parser.add_argument(
        "--stats-rtol",
        type=float,
        default=1e-6,
        help="relative tolerance for summary statistics (default 1e-6)",
    )
    parser.add_argument(
        "--sample-rtol",
        type=float,
        default=1e-4,
        help="relative tolerance for sampled pixel values (default 1e-4)",
    )
    args = parser.parse_args()

    digest_path = REPO_ROOT / "fixtures/runner/digests" / args.fixture_id / "expected.json"
    if not digest_path.is_file():
        print(f"no recorded digest at {digest_path}", file=sys.stderr)
        return 2

    recorded = read_digest(digest_path)
    actual = reproduce(args.fixture_id, expected_outputs=set(recorded["outputs"]))

    print(f"v2 outputs for {args.fixture_id}:")
    for name, arr in actual.items():
        print(
            f"  {name}: shape={arr.shape}, dtype={arr.dtype}, "
            f"range=[{arr.min():.4g}, {arr.max():.4g}]"
        )

    tol = Tolerance(
        stats_rtol=args.stats_rtol,
        sample_rtol=args.sample_rtol,
    )
    results = compare_digest(actual, recorded, tol=tol)
    print("\nComparison:")
    for r in results:
        print(f"  {r}")
    ok = all_ok(results)
    print(f"\nResult: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
