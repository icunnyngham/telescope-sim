"""v2 reproduction of fixture #17 (noisy_psf).

Mirrors :mod:`run_legacy_17_noisy_psf` exactly: same three actuation cases,
same fluxes, same ``np.random.seed(42)`` before each sample. Uses the new
``noisy_intensity`` output_tap kind with per-sample ``output_overrides``
to swap photon fluxes between cases.

Importable: :func:`reproduce` returns the digest-shaped outputs dict so
``tests/fixtures/test_canonical.py`` can compare against the committed
digest without spawning a subprocess.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "fixtures/runner"))


N_MIR = 3


def _actuation_cases() -> list[tuple[str, np.ndarray, float]]:
    """Must stay in lockstep with run_legacy_17_noisy_psf._actuation_cases()."""
    Z = np.zeros((N_MIR, 3))
    cases = []
    cases.append(("at_rest_flux1e6", Z.copy(), 1.0e6))
    cases.append(("at_rest_flux1e8", Z.copy(), 1.0e8))
    a = Z.copy()
    a[:, 1] = 0.5
    cases.append(("tip0p5_flux1e6", a, 1.0e6))
    return cases


def reproduce() -> dict[str, np.ndarray]:
    """Build the v2 sim and run the three configurations."""
    from telescope_sim import TelescopeSim

    yaml_path = REPO_ROOT / "fixtures/configs/17_noisy_psf.yaml"
    sim = TelescopeSim.from_yaml(yaml_path)

    cases = _actuation_cases()
    noisy_stack = []
    clean_stack = []
    for _label, ptt, flux in cases:
        np.random.seed(42)
        out = sim.sample(
            {"segments": ptt},
            output_overrides={"noisy_psf": {"int_phot_flux": flux}},
        )
        # Both clean_psf and noisy_psf carry shape (H, W, 1) — same as the
        # legacy capture's per-filter `out_samp[..., None]` shape.
        clean_stack.append(np.asarray(out["images"]["clean_psf"]))
        noisy_stack.append(np.asarray(out["images"]["noisy_psf"]))

    return {
        "noisy_psfs": np.array(noisy_stack, dtype=np.float64),
        "clean_psfs": np.array(clean_stack, dtype=np.float64),
    }


def main() -> int:
    from digest_lib import Tolerance, all_ok, compare_digest, read_digest

    digest_path = REPO_ROOT / "fixtures/runner/digests/17_noisy_psf/expected.json"
    if not digest_path.is_file():
        print(f"no recorded digest at {digest_path}", file=sys.stderr)
        return 2
    recorded = read_digest(digest_path)
    actual = reproduce()

    print("v2 reproduction (noisy stats per case):")
    cases = _actuation_cases()
    for i, (label, _, flux) in enumerate(cases):
        n = actual["noisy_psfs"][i]
        print(
            f"  {label:<24s}  flux={flux:.1e}  mean={n.mean():.4g}  "
            f"std={n.std():.4g}  max={n.max():.4g}"
        )

    results = compare_digest(actual, recorded, tol=Tolerance())
    print("\nComparison:")
    for r in results:
        print(f"  {r}")
    ok = all_ok(results)
    print(f"\nResult: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
