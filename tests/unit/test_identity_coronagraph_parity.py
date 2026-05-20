"""Parity tests for ``IdentityCoronagraph`` — the no-op coronagraph passthrough.

There's no legacy ``IdentityCoronagraph`` per se; the canonical reference is
the "no coronagraph" path (``self.use_coro = False`` in the coro variants,
or simply omitting the coronagraph block in canonical variants). The v2
equivalent is to place an ``IdentityCoronagraph`` in the pipeline as a
placeholder.

This audit verifies the coronagraph really is a pure passthrough — the
wavefront leaves apply() identical to the input, the bind step does no
work, and the constructor swallows any kwargs without complaint (since
the YAML loader may forward extra fields).
"""

from __future__ import annotations

import hcipy
import numpy as np
import pytest


@pytest.fixture(scope="module")
def pupil_grid():
    return hcipy.make_pupil_grid(32, 1.0)


def test_identity_apply_returns_input_wavefront_identity(pupil_grid):
    """apply(wf) must return the very same Wavefront object — not a copy.

    This is the strictest passthrough guarantee: any reference change would
    break the v2 expectation that running with IdentityCoronagraph is
    indistinguishable from passing coronagraph=None to _propagate_chain.
    """
    from telescope_sim.coronagraphs.standard import IdentityCoronagraph

    coro = IdentityCoronagraph()
    coro._bind_pupil_grid(pupil_grid)

    aper = hcipy.evaluate_supersampled(hcipy.make_circular_aperture(0.8), pupil_grid, 16)
    wf = hcipy.Wavefront(aper, 1.0e-6)

    out = coro.apply(wf)
    assert out is wf  # identity, not just equality


def test_identity_apply_preserves_aberrated_wavefront(pupil_grid):
    """And applies to an aberrated input identically — no phase mask, no Lyot stop."""
    from telescope_sim.coronagraphs.standard import IdentityCoronagraph

    coro = IdentityCoronagraph()
    coro._bind_pupil_grid(pupil_grid)

    aper = hcipy.evaluate_supersampled(hcipy.make_circular_aperture(0.8), pupil_grid, 16)
    x = np.asarray(pupil_grid.x)
    phase = 0.5 * x
    field = aper * np.exp(1j * phase)
    wf = hcipy.Wavefront(field, 1.0e-6)

    out = coro.apply(wf)
    np.testing.assert_allclose(
        np.asarray(out.electric_field), np.asarray(wf.electric_field), rtol=0, atol=0
    )


def test_identity_bind_pupil_grid_is_noop(pupil_grid):
    """_bind_pupil_grid does nothing — no caching, no error if called twice."""
    from telescope_sim.coronagraphs.standard import IdentityCoronagraph

    coro = IdentityCoronagraph()
    coro._bind_pupil_grid(pupil_grid)
    coro._bind_pupil_grid(pupil_grid)  # idempotent
    coro._bind_pupil_grid(None)  # accepts None too — no work to do


def test_identity_constructor_swallows_extra_kwargs():
    """Loader forwards `{type: identity, ...rest}` payload after popping `type`.

    If `rest` is non-empty (which can happen during YAML refactors), the
    constructor must not fail.
    """
    from telescope_sim.coronagraphs.standard import IdentityCoronagraph

    # Mimics the loader: `coro_cls(**payload_after_popping_type)`
    coro = IdentityCoronagraph(unused="value", other=42)
    assert coro is not None
    assert coro.name == "identity"
