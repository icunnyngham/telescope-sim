"""Compute and compare numerical digests of simulation outputs.

A *digest* is a compact, JSON-serializable summary of one or more numpy arrays:
shape, dtype, basic statistics, and a hash-stabilized set of sampled pixel
values. Digests serve as committed regression targets in the
``fixtures/runner/digests/`` tree — keeping them in JSON form keeps the repo
diffable and small, while still giving us enough information to catch
numerical drift between captured and reproduced outputs.

The library is dependency-light (numpy + stdlib only) so it can be imported
under any era-matched conda env we may build, not just the current dev env.
"""

from __future__ import annotations

import dataclasses
import datetime as _dt
import hashlib
import json
import platform
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

# Version of the digest schema itself. Bump when fields change in
# backward-incompatible ways.
DIGEST_SCHEMA_VERSION = "1.0"

# How many sampled-pixel positions to record per array.
DEFAULT_NUM_SAMPLES = 32


@dataclasses.dataclass(frozen=True)
class Tolerance:
    """Comparison tolerances for digest fields."""

    stats_rtol: float = 1e-6
    stats_atol: float = 1e-9
    sample_rtol: float = 1e-4
    sample_atol: float = 1e-7


def _stable_pixel_indices(
    shape: tuple[int, ...],
    n_samples: int,
    seed: int,
) -> NDArray[np.int64]:
    """Pick `n_samples` flat indices into an array of `shape`, deterministically.

    Uses numpy's legacy ``RandomState`` so the indices are stable regardless of
    NEP-19 generator changes across numpy releases.
    """
    rng = np.random.RandomState(seed)
    total = int(np.prod(shape))
    if total == 0:
        return np.zeros(0, dtype=np.int64)
    n_samples = min(n_samples, total)
    return rng.choice(total, size=n_samples, replace=False).astype(np.int64)


def _index_hash(indices: NDArray[np.int64]) -> str:
    """SHA-256 hash of the sorted flat indices. Tells us if the sampling
    scheme matches between capture and reproduction without revealing the
    actual indices in the JSON."""
    return hashlib.sha256(np.sort(indices).tobytes()).hexdigest()[:16]


def array_digest(
    array: NDArray,
    *,
    n_samples: int = DEFAULT_NUM_SAMPLES,
    sample_seed: int = 0,
) -> dict[str, Any]:
    """Return a JSON-serializable digest of a single numpy array."""
    a = np.asarray(array)
    flat = a.ravel()
    indices = _stable_pixel_indices(a.shape, n_samples, sample_seed)
    sample_values = flat[indices].astype(np.float64).tolist() if indices.size else []
    return {
        "shape": list(a.shape),
        "dtype": str(a.dtype),
        "stats": {
            "mean": float(np.mean(a)) if a.size else 0.0,
            "std": float(np.std(a)) if a.size else 0.0,
            "min": float(np.min(a)) if a.size else 0.0,
            "max": float(np.max(a)) if a.size else 0.0,
            "sum": float(np.sum(a)) if a.size else 0.0,
        },
        "samples": {
            "n": int(indices.size),
            "seed": int(sample_seed),
            "indices_hash": _index_hash(indices),
            "values": sample_values,
        },
    }


def make_digest(
    fixture_id: str,
    outputs: Mapping[str, NDArray],
    *,
    rng_seed: int | None = None,
    hcipy_version: str | None = None,
    extra: Mapping[str, Any] | None = None,
    sample_seed: int = 0,
    n_samples: int = DEFAULT_NUM_SAMPLES,
) -> dict[str, Any]:
    """Build the full digest record for a fixture run."""
    return {
        "schema_version": DIGEST_SCHEMA_VERSION,
        "fixture_id": fixture_id,
        "rng_seed": rng_seed,
        "captured_at": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "platform": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "system": platform.system(),
            "hcipy": hcipy_version,
        },
        "outputs": {
            name: array_digest(arr, n_samples=n_samples, sample_seed=sample_seed)
            for name, arr in outputs.items()
        },
        "extra": dict(extra) if extra else {},
    }


def write_digest(digest: Mapping[str, Any], path: str | Path) -> None:
    """Write a digest to JSON, pretty-printed for diff-friendliness."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(digest, indent=2, sort_keys=False) + "\n")


def read_digest(path: str | Path) -> dict[str, Any]:
    """Read a digest from JSON."""
    return json.loads(Path(path).read_text())


@dataclasses.dataclass
class DigestComparison:
    """Result of comparing a reproduced array to a recorded digest."""

    output_name: str
    ok: bool
    shape_match: bool
    dtype_match: bool
    stats_match: bool
    samples_match: bool
    notes: list[str] = dataclasses.field(default_factory=list)

    def __str__(self) -> str:
        flag = "OK " if self.ok else "FAIL"
        return (
            f"[{flag}] {self.output_name}: " + "; ".join(self.notes)
            if self.notes
            else f"[{flag}] {self.output_name}"
        )


def _close(a: float, b: float, rtol: float, atol: float) -> bool:
    return abs(a - b) <= atol + rtol * max(abs(a), abs(b))


def compare_array_digest(
    actual: NDArray,
    recorded: Mapping[str, Any],
    *,
    name: str,
    tol: Tolerance,
) -> DigestComparison:
    """Compare a reproduced array against a recorded digest entry."""
    a = np.asarray(actual)
    notes: list[str] = []

    rec_shape = tuple(recorded["shape"])
    shape_match = a.shape == rec_shape
    if not shape_match:
        notes.append(f"shape mismatch: actual {a.shape} vs recorded {rec_shape}")

    rec_dtype = recorded["dtype"]
    dtype_match = str(a.dtype) == rec_dtype
    if not dtype_match:
        notes.append(f"dtype mismatch: actual {a.dtype} vs recorded {rec_dtype}")

    rec_stats = recorded["stats"]
    actual_stats = {
        "mean": float(np.mean(a)),
        "std": float(np.std(a)),
        "min": float(np.min(a)),
        "max": float(np.max(a)),
        "sum": float(np.sum(a)),
    }
    stats_match = all(
        _close(actual_stats[k], rec_stats[k], tol.stats_rtol, tol.stats_atol) for k in actual_stats
    )
    if not stats_match:
        for k in actual_stats:
            if not _close(actual_stats[k], rec_stats[k], tol.stats_rtol, tol.stats_atol):
                notes.append(
                    f"stats.{k}: actual {actual_stats[k]:.6g} vs recorded {rec_stats[k]:.6g}"
                )

    # Compare sampled pixels at the recorded indices (recompute indices from
    # the same seed).
    samples_match = True
    if shape_match:
        rec_samples = recorded["samples"]
        indices = _stable_pixel_indices(a.shape, rec_samples["n"], rec_samples["seed"])
        if _index_hash(indices) != rec_samples["indices_hash"]:
            notes.append("indices_hash mismatch (sampling scheme differs)")
            samples_match = False
        else:
            actual_vals = a.ravel()[indices].astype(np.float64)
            rec_vals = np.array(rec_samples["values"], dtype=np.float64)
            diff_mask = ~np.isclose(
                actual_vals, rec_vals, rtol=tol.sample_rtol, atol=tol.sample_atol
            )
            n_diff = int(np.sum(diff_mask))
            if n_diff > 0:
                samples_match = False
                worst = int(np.argmax(np.abs(actual_vals - rec_vals)))
                notes.append(
                    f"{n_diff}/{len(actual_vals)} sample values out of tolerance; "
                    f"worst index {indices[worst]}: actual {actual_vals[worst]:.6g} "
                    f"vs recorded {rec_vals[worst]:.6g}"
                )

    ok = shape_match and dtype_match and stats_match and samples_match
    return DigestComparison(
        output_name=name,
        ok=ok,
        shape_match=shape_match,
        dtype_match=dtype_match,
        stats_match=stats_match,
        samples_match=samples_match,
        notes=notes,
    )


def compare_digest(
    actual: Mapping[str, NDArray],
    recorded: Mapping[str, Any],
    *,
    tol: Tolerance | None = None,
) -> list[DigestComparison]:
    """Compare a full set of reproduced outputs against a recorded digest."""
    tol = tol or Tolerance()
    rec_outputs = recorded["outputs"]

    extra_actual = set(actual) - set(rec_outputs)
    extra_recorded = set(rec_outputs) - set(actual)

    results: list[DigestComparison] = []
    for name in sorted(set(actual) | set(rec_outputs)):
        if name in actual and name in rec_outputs:
            results.append(
                compare_array_digest(actual[name], rec_outputs[name], name=name, tol=tol)
            )
        elif name in extra_actual:
            results.append(
                DigestComparison(
                    output_name=name,
                    ok=False,
                    shape_match=False,
                    dtype_match=False,
                    stats_match=False,
                    samples_match=False,
                    notes=["present in reproduction but not in recorded digest"],
                )
            )
        else:
            results.append(
                DigestComparison(
                    output_name=name,
                    ok=False,
                    shape_match=False,
                    dtype_match=False,
                    stats_match=False,
                    samples_match=False,
                    notes=["present in recorded digest but missing from reproduction"],
                )
            )
    return results


def all_ok(results: list[DigestComparison]) -> bool:
    return all(r.ok for r in results)


if __name__ == "__main__":  # pragma: no cover  (smoke test)
    rng = np.random.RandomState(0)
    arr = rng.normal(size=(16, 16))
    d = make_digest("smoke", {"x": arr}, rng_seed=42)
    print(json.dumps(d, indent=2))
    cmp_ = compare_digest({"x": arr}, d)
    for r in cmp_:
        print(r)
    sys.exit(0 if all_ok(cmp_) else 1)
