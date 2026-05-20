"""Parity tests for ``VectorVortexCoronagraphImpl`` against direct HCIPy calls.

Legacy references (multiple variants, all 2024-05):
    variants/vampires_vvc_2024-05__coro_mas_psf.py:134
    variants/scexao_vvc_2024-05__coro_mas_psf.py:134
    variants/fp_rl_ff_vvc_2024-05__coro_mas_psf.py:134

All three are the same single line:
    self.coro = hcipy.VectorVortexCoronagraph(charge=4, lyot_stop=self.lyot)

Note the API difference from VortexCoronagraph: VVC does NOT take a
``pupil_grid`` argument. It infers the grid from the wavefront at apply
time. The Lyot stop in these variants is a *Field* (from ``vp.generate_pupil``),
NOT a supersampled callable — so the supersample-override audit from
test_vortex_coronagraph_parity does not transfer here.
"""

from __future__ import annotations

import hcipy
import numpy as np
import pytest

# Trigger the registry side-effect imports the YAML loader does normally.
import telescope_sim.apertures.external_pupil  # noqa: F401


@pytest.fixture(scope="module")
def pupil_grid():
    return hcipy.make_pupil_grid(128, 12.0)


@pytest.fixture(scope="module")
def lyot_field(pupil_grid):
    """Lyot Field — matches the VVC variants' direct `vp.generate_pupil(...)` path."""
    callable_ = hcipy.aperture.make_obstructed_circular_aperture(
        pupil_diameter=7.395,
        central_obscuration_ratio=0.3,
    )
    return hcipy.evaluate_supersampled(callable_, pupil_grid, 8)


@pytest.fixture(scope="module")
def aberrated_wf(pupil_grid):
    """Pupil-plane wavefront with a non-trivial defocus + tilt phase."""
    aper_callable = hcipy.make_circular_aperture(8.0)
    aper_field = hcipy.evaluate_supersampled(aper_callable, pupil_grid, 8)
    x = np.asarray(pupil_grid.x)
    y = np.asarray(pupil_grid.y)
    r2 = (x**2 + y**2) / (4.0**2)
    phase = 0.6 * r2 + 0.2 * x / 4.0
    field = aper_field * np.exp(1j * phase)
    return hcipy.Wavefront(field, 1.0e-6)


def _v2_vvc(pupil_grid, charge=4):
    """Build the v2 wrapper exactly as the loader would for a VVC fixture."""
    from telescope_sim.coronagraphs.standard import VectorVortexCoronagraphImpl

    coro = VectorVortexCoronagraphImpl(
        charge=charge,
        lyot={
            "type": "external_pupil",
            "module": "hcipy.aperture",
            "function": "make_obstructed_circular_aperture",
            "mode": "callable",
            "kwargs": {
                "pupil_diameter": 7.395,
                "central_obscuration_ratio": 0.3,
            },
            "supersample": 8,
        },
    )
    coro._bind_pupil_grid(pupil_grid)
    return coro


def test_vvc_wraps_hcipy_faithfully_pupil_plane(pupil_grid, lyot_field, aberrated_wf):
    """v2 VVC wrapper output matches a direct ``hcipy.VectorVortexCoronagraph`` call.

    Uses charge=4 — the value all three active legacy uses share.
    """
    legacy = hcipy.VectorVortexCoronagraph(charge=4, lyot_stop=lyot_field)
    v2 = _v2_vvc(pupil_grid, charge=4)

    legacy_out = legacy(aberrated_wf)
    v2_out = v2.apply(aberrated_wf)

    np.testing.assert_allclose(
        np.asarray(v2_out.electric_field),
        np.asarray(legacy_out.electric_field),
        rtol=0,
        atol=1e-14,
    )


def test_vvc_focal_intensity_parity(pupil_grid, lyot_field, aberrated_wf):
    """Downstream focal-plane intensity parity (the pipeline's actual surface)."""
    legacy = hcipy.VectorVortexCoronagraph(charge=4, lyot_stop=lyot_field)
    v2 = _v2_vvc(pupil_grid, charge=4)

    focal_grid = hcipy.make_uniform_grid([64, 64], 5e-6)
    prop = hcipy.FraunhoferPropagator(pupil_grid, focal_grid)

    legacy_int = np.asarray(prop(legacy(aberrated_wf)).intensity.shaped)
    v2_int = np.asarray(prop(v2.apply(aberrated_wf)).intensity.shaped)

    np.testing.assert_allclose(v2_int, legacy_int, rtol=1e-12, atol=1e-18)


def test_vvc_charge_threads_through(pupil_grid, aberrated_wf):
    """Charge changes are reflected in HCIPy output; default is 4."""
    from telescope_sim.coronagraphs.standard import VectorVortexCoronagraphImpl

    v2_default = VectorVortexCoronagraphImpl(lyot=None)
    v2_default._bind_pupil_grid(pupil_grid)
    assert v2_default.charge == 4  # default matches legacy

    # charge=2 path
    v2_2 = VectorVortexCoronagraphImpl(charge=2, lyot=None)
    v2_2._bind_pupil_grid(pupil_grid)
    legacy_2 = hcipy.VectorVortexCoronagraph(charge=2, lyot_stop=None)

    np.testing.assert_allclose(
        np.asarray(v2_2.apply(aberrated_wf).electric_field),
        np.asarray(legacy_2(aberrated_wf).electric_field),
        rtol=0,
        atol=1e-14,
    )

    # And charge=2 differs from charge=4 (so the kwarg actually does something)
    legacy_4 = hcipy.VectorVortexCoronagraph(charge=4, lyot_stop=None)
    diff = np.linalg.norm(
        np.asarray(legacy_2(aberrated_wf).electric_field)
        - np.asarray(legacy_4(aberrated_wf).electric_field)
    )
    assert diff > 1e-6


def test_vvc_unbound_raises():
    from telescope_sim.coronagraphs.standard import VectorVortexCoronagraphImpl

    coro = VectorVortexCoronagraphImpl(charge=4, lyot=None)
    with pytest.raises(RuntimeError, match="_bind_pupil_grid"):
        coro.apply(None)
