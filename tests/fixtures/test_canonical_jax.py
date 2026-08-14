"""Regression tests: the ``jax`` backend reproduces the same legacy digests.

Runs the committed golden fixtures a second time with ``backend="jax"``,
comparing against the same digests and tolerances the default (hcipy)
backend is held to in :mod:`test_canonical`. Rather than restating the
comparison logic, each test forces the backend and then calls the
corresponding hcipy-backend test body, so the two suites can never drift
apart.

Fixtures whose components have no jax implementation (vortex
coronagraphs, the ``fiber_dual`` tap) are skipped with the reason derived
from their config, and the whole module skips when the optional
JAX dependency are absent.

Marked ``slow`` like its hcipy sibling: run with
``pytest --runslow tests/fixtures/``.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

import test_canonical as harness  # noqa: E402  (sibling module; also puts fixtures/runner on sys.path)

REPO_ROOT = harness.REPO_ROOT

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("jax") is None,
    reason="backend='jax' needs the optional JAX dependency",
)


def _jax_block_reason(fixture_id: str) -> str | None:
    """Why this fixture cannot run on the jax backend, or None if it can.

    Read straight from the fixture's YAML so new fixtures are classified
    automatically. Mirrors the loader's config-time gates: non-identity
    coronagraphs and the ``fiber_dual`` tap are hcipy-only. (Correctors
    are not screened here — every corrector kind the fixtures use backs
    onto a DM surface, and the loader raises a clear error otherwise.)
    """
    config_path = REPO_ROOT / "fixtures/configs" / f"{fixture_id}.yaml"
    config = yaml.safe_load(config_path.read_text())

    coronagraph = config.get("coronagraph")
    if coronagraph is not None and coronagraph.get("type") not in ("identity", "lyot"):
        return f"coronagraph {coronagraph.get('type')!r} has no jax implementation"

    for out_name, out_cfg in (config.get("outputs") or {}).items():
        tap_type = (out_cfg.get("tap") or {}).get("type")
        if tap_type == "fiber_dual":
            return f"output {out_name!r} uses the hcipy-only {tap_type!r} tap"
    return None


def _jax_params() -> list:
    """Re-parametrize the hcipy fixture list, skipping the ineligible ones."""
    params = []
    for entry in harness.CANONICAL_FIXTURES:
        # Entries are either a plain id or a pytest.param wrapping one.
        fixture_id = getattr(entry, "values", (entry,))[0]
        reason = _jax_block_reason(fixture_id)
        marks = (pytest.mark.skip(reason=reason),) if reason else ()
        params.append(pytest.param(fixture_id, marks=marks, id=fixture_id))
    return params


def _force_backend(monkeypatch: pytest.MonkeyPatch, backend: str) -> None:
    """Make every ``TelescopeSim.from_yaml`` in this test build on ``backend``.

    The fixture runners deliberately take no backend argument (they mirror
    the legacy capture scripts); patching the constructor lets this module
    reuse them verbatim instead of re-implementing their actuation
    sequences.
    """
    import telescope_sim  # noqa: PLC0415

    original = telescope_sim.TelescopeSim.from_yaml.__func__

    def from_yaml(cls, path, *, backend=backend):
        return original(cls, path, backend=backend)

    monkeypatch.setattr(telescope_sim.TelescopeSim, "from_yaml", classmethod(from_yaml))


@pytest.mark.slow
@pytest.mark.fixture
@pytest.mark.parametrize("fixture_id", _jax_params())
def test_canonical_jax_reproduces_digest(fixture_id: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Same digest comparison as the hcipy suite, built with backend='jax'."""
    import telescope_sim  # noqa: PLC0415

    _force_backend(monkeypatch, "jax")
    harness.test_canonical_v2_reproduces_digest(fixture_id, telescope_sim)


@pytest.mark.slow
@pytest.mark.fixture
def test_noisy_psf_jax_reproduces_digest(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fixture #17 (seeded detector noise) on the jax backend."""
    import telescope_sim  # noqa: PLC0415

    _force_backend(monkeypatch, "jax")
    harness.test_noisy_psf_v2_reproduces_digest(telescope_sim)


@pytest.mark.slow
@pytest.mark.fixture
def test_strehl_zernike_jax_reproduces_digest(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fixture #16 (both Strehl modes over an actuation sweep) on jax."""
    import telescope_sim  # noqa: PLC0415

    _force_backend(monkeypatch, "jax")
    harness.test_strehl_zernike_v2_reproduces_digest(telescope_sim)
