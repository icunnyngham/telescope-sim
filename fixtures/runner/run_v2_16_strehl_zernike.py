"""v2 reproduction of fixture #16 (strehl_zernike).

Mirrors :mod:`run_legacy_16_strehl_zernike`: builds two v2 ``TelescopeSim``
instances (one peak-mode, one matched-filter), runs the same 8 PTT
actuation cases through both, returns the two Strehl arrays under the
same keys the legacy digest uses (``strehls_peak``,
``strehls_matched_filter``).

Importable: :func:`reproduce` returns the outputs dict so
``tests/fixtures/test_canonical.py`` can compare against the committed
digest without spawning a subprocess.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "fixtures/runner"))


# --- Same 3-segment aperture as the legacy capture. -----------------------
N_MIR = 3

CASE_LABELS = [
    "at_rest",
    "global_tip_small",
    "global_tip_large",
    "global_tilt_small",
    "differential_piston",
    "seg0_combined",
    "per_segment_tip",
    "global_piston_large",
]


def _actuation_cases() -> list[tuple[str, np.ndarray]]:
    """Must stay in lockstep with run_legacy_16_strehl_zernike._actuation_cases()."""
    cases: list[tuple[str, np.ndarray]] = []
    Z = np.zeros((N_MIR, 3))
    cases.append(("at_rest", Z.copy()))

    a = Z.copy(); a[:, 1] = 0.1
    cases.append(("global_tip_small", a))

    a = Z.copy(); a[:, 1] = 0.5
    cases.append(("global_tip_large", a))

    a = Z.copy(); a[:, 2] = 0.1
    cases.append(("global_tilt_small", a))

    a = Z.copy(); a[0, 0] = 0.3; a[1, 0] = -0.3
    cases.append(("differential_piston", a))

    a = Z.copy(); a[0] = (0.2, 0.1, 0.1)
    cases.append(("seg0_combined", a))

    a = Z.copy(); a[:, 1] = (0.3, -0.3, 0.3)
    cases.append(("per_segment_tip", a))

    a = Z.copy(); a[:, 0] = 0.5
    cases.append(("global_piston_large", a))

    return cases


def _run(yaml_name: str, cases) -> np.ndarray:
    """Build a v2 sim and run the actuation sequence. Returns (n_cases, 1)."""
    from telescope_sim import TelescopeSim

    yaml_path = REPO_ROOT / "fixtures/configs" / yaml_name
    sim = TelescopeSim.from_yaml(yaml_path)
    strehls = []
    for _label, ptt in cases:
        # v2 PTT actuator order: (n_segments, 3) = (piston, tip, tilt),
        # same convention as the legacy ptt_actuate.
        out = sim.sample({"segments": ptt}, meas_strehl=True)
        # Single focal plane named "filter1" → one Strehl value.
        strehls.append([out["strehls"]["filter1"]])
    return np.array(strehls, dtype=np.float64)


def reproduce() -> dict[str, np.ndarray]:
    """Build both v2 sims and return the digest-shaped outputs dict."""
    cases = _actuation_cases()
    np.random.seed(42)
    strehls_peak = _run("16_strehl_zernike.yaml", cases)
    strehls_matched_filter = _run("16_strehl_zernike_matched_filter.yaml", cases)
    return {
        "strehls_peak": strehls_peak,
        "strehls_matched_filter": strehls_matched_filter,
    }


def main() -> int:
    from digest_lib import Tolerance, all_ok, compare_digest, read_digest

    digest_path = REPO_ROOT / "fixtures/runner/digests/16_strehl_zernike/expected.json"
    if not digest_path.is_file():
        print(f"no recorded digest at {digest_path}", file=sys.stderr)
        return 2
    recorded = read_digest(digest_path)
    actual = reproduce()

    print("v2 reproduction:")
    for i, label in enumerate(CASE_LABELS):
        print(f"  {label:<24s}  peak={actual['strehls_peak'][i, 0]:.4f}  "
              f"matched_filter={actual['strehls_matched_filter'][i, 0]:.4f}")

    results = compare_digest(actual, recorded, tol=Tolerance())
    print("\nComparison:")
    for r in results:
        print(f"  {r}")
    ok = all_ok(results)
    print(f"\nResult: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
