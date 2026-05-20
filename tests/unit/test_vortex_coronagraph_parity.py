"""Parity tests for ``VortexCoronagraphImpl`` against the legacy direct HCIPy call.

Legacy reference (variants/coro__coro_mas_psf.py:189):
    coro = hcipy.VortexCoronagraph(self.pupil_grid, charge=2, lyot_stop=self.lyot_mask)

where ``lyot_mask = hcipy.evaluate_supersampled(lyot_func(), pupil_grid, 8)``.

v2's :class:`VortexCoronagraphImpl` is a thin wrapper that should produce a
bit-identical HCIPy ``VortexCoronagraph`` instance when fed matching inputs.
These tests build both sides in parallel and assert that applying them to a
non-trivial (aberrated) wavefront yields the same output — both pupil-plane
output and downstream focal-plane intensity.

The audit value here is structural: the underlying math is HCIPy's, but the
wrapper has to (a) pass ``charge`` through unchanged, (b) build the Lyot stop
with the configured supersample (NOT the aperture default), and (c) apply
the coronagraph to the wavefront identically. A subtle bug in any of those
would diverge from legacy without crashing.
"""

from __future__ import annotations

import hcipy
import numpy as np
import pytest

# Trigger the registry side-effect imports that the YAML loader does
# normally. Without these the lyot field's external_pupil aperture lookup
# would fail.
import telescope_sim.apertures.external_pupil  # noqa: F401


@pytest.fixture(scope="module")
def pupil_grid():
    """Pupil grid matching fixture #07_coro_original conventions (downscaled)."""
    return hcipy.make_pupil_grid(128, 12.0)


@pytest.fixture(scope="module")
def lyot_mask(pupil_grid):
    """Lyot stop built the legacy way: supersampled circular obstructed aperture."""
    callable_ = hcipy.aperture.make_obstructed_circular_aperture(
        pupil_diameter=7.395,
        central_obscuration_ratio=0.3,
    )
    return hcipy.evaluate_supersampled(callable_, pupil_grid, 8)


@pytest.fixture(scope="module")
def aberrated_wf(pupil_grid):
    """A pupil-plane wavefront with a non-trivial defocus + tilt phase."""
    aper_callable = hcipy.make_circular_aperture(8.0)
    aper_field = hcipy.evaluate_supersampled(aper_callable, pupil_grid, 8)
    # Defocus-like (∝ r²) + a tilt component, both small relative to 2π.
    x = np.asarray(pupil_grid.x)
    y = np.asarray(pupil_grid.y)
    r2 = (x**2 + y**2) / (4.0**2)
    phase = 0.6 * r2 + 0.2 * x / 4.0
    field = aper_field * np.exp(1j * phase)
    return hcipy.Wavefront(field, 1.0e-6)


def _v2_coronagraph(pupil_grid):
    """Build the v2 wrapper exactly as the loader would for fixture #07."""
    from telescope_sim.coronagraphs.standard import VortexCoronagraphImpl

    coro = VortexCoronagraphImpl(
        charge=2,
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


def test_vortex_wraps_hcipy_faithfully_pupil_plane(pupil_grid, lyot_mask, aberrated_wf):
    """v2 wrapper output matches a direct ``hcipy.VortexCoronagraph`` call.

    The pupil-plane output (the result of the coronagraph's pupil→pupil
    operation, before any focal propagator) must agree pixel-for-pixel.
    """
    legacy_coro = hcipy.VortexCoronagraph(pupil_grid, charge=2, lyot_stop=lyot_mask)
    legacy_out = legacy_coro(aberrated_wf)

    v2_coro = _v2_coronagraph(pupil_grid)
    v2_out = v2_coro.apply(aberrated_wf)

    np.testing.assert_allclose(
        np.asarray(v2_out.electric_field),
        np.asarray(legacy_out.electric_field),
        rtol=0,
        atol=1e-14,
    )


def test_vortex_wraps_hcipy_faithfully_focal_intensity(pupil_grid, lyot_mask, aberrated_wf):
    """Same parity, but check the downstream focal-plane intensity.

    This is the failure mode the pipeline actually surfaces — even a tiny
    pupil-plane phase drift could compound through the Fraunhofer prop.
    """
    legacy_coro = hcipy.VortexCoronagraph(pupil_grid, charge=2, lyot_stop=lyot_mask)
    v2_coro = _v2_coronagraph(pupil_grid)

    focal_grid = hcipy.make_uniform_grid([64, 64], 5e-6)
    prop = hcipy.FraunhoferPropagator(pupil_grid, focal_grid)

    legacy_int = np.asarray(prop(legacy_coro(aberrated_wf)).intensity.shaped)
    v2_int = np.asarray(prop(v2_coro.apply(aberrated_wf)).intensity.shaped)

    np.testing.assert_allclose(v2_int, legacy_int, rtol=1e-12, atol=1e-18)


def test_vortex_charge_threads_through(pupil_grid, aberrated_wf):
    """Non-default charge values are passed to HCIPy unchanged.

    Tests charge=4 (high-charge vortex used by some VAMPIRES variants).
    The output for charge=4 differs significantly from charge=2; if the
    wrapper silently defaulted, this test would fail.
    """
    from telescope_sim.coronagraphs.standard import VortexCoronagraphImpl

    legacy = hcipy.VortexCoronagraph(pupil_grid, charge=4, lyot_stop=None)
    v2 = VortexCoronagraphImpl(charge=4, lyot=None)
    v2._bind_pupil_grid(pupil_grid)

    legacy_out = legacy(aberrated_wf)
    v2_out = v2.apply(aberrated_wf)

    np.testing.assert_allclose(
        np.asarray(v2_out.electric_field),
        np.asarray(legacy_out.electric_field),
        rtol=0,
        atol=1e-14,
    )
    # Sanity check that charge=4 actually differs from charge=2 (so the
    # test would catch a "wrapper drops the charge kwarg" bug).
    legacy2 = hcipy.VortexCoronagraph(pupil_grid, charge=2, lyot_stop=None)
    diff = np.linalg.norm(
        np.asarray(legacy(aberrated_wf).electric_field)
        - np.asarray(legacy2(aberrated_wf).electric_field)
    )
    assert diff > 1e-6


def test_vortex_lyot_supersample_is_honored(pupil_grid, aberrated_wf):
    """The Lyot stop's ``supersample`` from the YAML must override the aperture default.

    Legacy hardcodes ``evaluate_supersampled(lyot_func(), pupil_grid, 8)``. The
    v2 ExternalPupilAperture default is ``supersample=16``. If the wrapper
    silently used the default, the Lyot stop would be a different field and
    the coronagraph output would diverge.
    """
    from telescope_sim.coronagraphs.standard import VortexCoronagraphImpl

    # Identical config except for the supersample override
    cfg_ss8 = {
        "type": "external_pupil",
        "module": "hcipy.aperture",
        "function": "make_obstructed_circular_aperture",
        "mode": "callable",
        "kwargs": {"pupil_diameter": 7.395, "central_obscuration_ratio": 0.3},
        "supersample": 8,
    }
    cfg_default = {**cfg_ss8}
    cfg_default.pop("supersample")  # uses ExternalPupilAperture default = 16

    coro_ss8 = VortexCoronagraphImpl(charge=2, lyot=cfg_ss8)
    coro_ss8._bind_pupil_grid(pupil_grid)
    coro_default = VortexCoronagraphImpl(charge=2, lyot=cfg_default)
    coro_default._bind_pupil_grid(pupil_grid)

    out_ss8 = np.asarray(coro_ss8.apply(aberrated_wf).electric_field)
    out_default = np.asarray(coro_default.apply(aberrated_wf).electric_field)
    # The two outputs must differ (different Lyot stops). If they don't,
    # the supersample field is being ignored.
    diff = np.linalg.norm(out_ss8 - out_default)
    assert diff > 1e-10, (
        "Lyot stop supersample appears to be ignored — outputs identical "
        "between supersample=8 and supersample=16."
    )

    # And the ss=8 output must match the legacy-built reference exactly.
    legacy_mask = hcipy.evaluate_supersampled(
        hcipy.aperture.make_obstructed_circular_aperture(
            pupil_diameter=7.395, central_obscuration_ratio=0.3
        ),
        pupil_grid,
        8,
    )
    legacy_coro = hcipy.VortexCoronagraph(pupil_grid, charge=2, lyot_stop=legacy_mask)
    legacy_out = np.asarray(legacy_coro(aberrated_wf).electric_field)
    np.testing.assert_allclose(out_ss8, legacy_out, rtol=0, atol=1e-14)


def test_vortex_unbound_raises():
    """apply() before _bind_pupil_grid() must raise — the wrapper has no fallback."""
    from telescope_sim.coronagraphs.standard import VortexCoronagraphImpl

    coro = VortexCoronagraphImpl(charge=2, lyot=None)
    with pytest.raises(RuntimeError, match="_bind_pupil_grid"):
        coro.apply(None)
