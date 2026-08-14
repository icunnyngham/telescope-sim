"""Unit tests for the classical ``lyot`` coronagraph (hcipy backend).

Fixture #12 (the legacy VAMPIRES Lyot variant) never produced working
output, so there is deliberately no golden regression target here.
Validation is structural and physical instead:

- the wrapper builds exactly the same optical train as a hand-built
  ``hcipy.LyotCoronagraph`` (identical fields out);
- the occulter removes core flux and the total transmitted energy drops;
- a tighter Lyot stop transmits less energy than a looser one;
- the reference PSF always bypasses the coronagraph.

Cross-backend parity lives in ``test_jax_lyot_parity.py``.
"""

from __future__ import annotations

from pathlib import Path

import hcipy
import numpy as np
import pytest
import yaml

from telescope_sim.config.loader import build
from telescope_sim.config.schema import SimConfig
from telescope_sim.coronagraphs.lyot import LyotCoronagraphImpl
from telescope_sim.registry import lookup

DATA = Path(__file__).parent / "data"


def _sim_from_yaml(path, mutate=None):
    with open(path) as f:
        raw = yaml.safe_load(f)
    if mutate is not None:
        mutate(raw)
    return build(SimConfig.model_validate(raw), backend="hcipy")


def _widen(raw, focal_extent=2.0):
    for fp in raw["focal_planes"].values():
        fp["focal_extent"] = focal_extent


# --- Registration & config surface -------------------------------------------


def test_lyot_is_registered_for_both_backends():
    cls = lookup("coronagraph", "lyot")
    assert cls is LyotCoronagraphImpl
    assert cls.supported_backends == frozenset({"hcipy", "jax"})


def test_constructor_validation():
    with pytest.raises(ValueError, match="occulter_diameter"):
        LyotCoronagraphImpl(occulter_diameter=0.0)
    with pytest.raises(ValueError, match="mask_extent"):
        LyotCoronagraphImpl(occulter_diameter=1e-6, mask_extent=0.5e-6)
    with pytest.raises(ValueError, match="mask_resolution"):
        LyotCoronagraphImpl(occulter_diameter=1e-6, mask_resolution=1)


def test_apply_before_bind_raises():
    coro = LyotCoronagraphImpl(occulter_diameter=1e-6)
    with pytest.raises(RuntimeError, match="_bind_pupil_grid"):
        coro.apply(object())


def test_mask_extent_defaults_to_twice_the_occulter():
    coro = LyotCoronagraphImpl(occulter_diameter=3e-6)
    assert coro.mask_extent == pytest.approx(6e-6)


# --- Equivalence with a hand-built hcipy.LyotCoronagraph ----------------------


def test_matches_hand_built_hcipy_lyot_coronagraph():
    pupil_grid = hcipy.make_pupil_grid(64, 1.05)
    aperture = hcipy.evaluate_supersampled(hcipy.make_circular_aperture(1.0), pupil_grid, 4)

    diameter = 6e-6
    coro = LyotCoronagraphImpl(
        occulter_diameter=diameter,
        mask_resolution=96,
        lyot={
            "type": "external_pupil",
            "module": "hcipy",
            "function": "make_circular_aperture",
            "mode": "callable",
            "kwargs": {"diameter": 0.85},
        },
    )
    coro._bind_pupil_grid(pupil_grid)

    mask_grid = hcipy.make_uniform_grid([96] * 2, [2 * diameter] * 2)
    spot = hcipy.evaluate_supersampled(hcipy.make_circular_aperture(diameter), mask_grid, 8)
    # external_pupil's callable mode supersamples at 16 by default.
    stop = hcipy.evaluate_supersampled(hcipy.make_circular_aperture(0.85), pupil_grid, 16)
    reference = hcipy.LyotCoronagraph(
        pupil_grid,
        focal_plane_mask=1.0 - spot,
        lyot_stop=stop,
        focal_plane_mask_grid=mask_grid,
    )

    # An aberrated wavefront so the test sees more than the trivial case.
    rng = np.random.default_rng(7)
    opd = hcipy.Field(rng.normal(scale=2e-8, size=pupil_grid.size), pupil_grid)
    for lam in (0.95e-6, 1.05e-6):
        wf = hcipy.Wavefront(aperture * np.exp(2j * np.pi * opd / lam), lam)
        got = coro.apply(wf).electric_field
        want = reference(wf).electric_field
        np.testing.assert_allclose(np.asarray(got), np.asarray(want), rtol=0, atol=1e-15)


# --- Physics ------------------------------------------------------------------


@pytest.fixture(scope="module")
def lyot_sim():
    return _sim_from_yaml(DATA / "lyot_zernike.yaml", mutate=_widen)


def test_occulter_removes_core_flux_and_energy(lyot_sim):
    fp = lyot_sim.focal_planes["filter1"]
    rest = lyot_sim.sample()["images"]["psf"][..., 0]
    # The on-axis core is strongly suppressed relative to the coro-free
    # reference PSF, and most of the total energy is rejected. The bounds
    # are loose physics sanity levels, not fitted values.
    assert rest.max() / fp.reference_peak_intensity < 1e-2
    assert rest.sum() / fp.reference_psf_sum < 0.05


def test_reference_psf_bypasses_coronagraph(lyot_sim):
    def _drop_coro(raw):
        _widen(raw)
        raw.pop("coronagraph")

    open_sim = _sim_from_yaml(DATA / "lyot_zernike.yaml", mutate=_drop_coro)
    ref_coro = lyot_sim.focal_planes["filter1"].reference_psf
    ref_open = open_sim.focal_planes["filter1"].reference_psf
    np.testing.assert_allclose(ref_coro, ref_open, rtol=1e-12)


def test_tighter_lyot_stop_transmits_less(lyot_sim):
    def _tighten(raw):
        _widen(raw)
        raw["coronagraph"]["lyot"]["kwargs"]["diameter"] = 0.6

    tight = _sim_from_yaml(DATA / "lyot_zernike.yaml", mutate=_tighten)
    loose_sum = lyot_sim.sample()["images"]["psf"][..., 0].sum()
    tight_sum = tight.sample()["images"]["psf"][..., 0].sum()
    assert 0 < tight_sum < loose_sum


def test_no_stop_transmits_more_than_stopped(lyot_sim):
    def _no_stop(raw):
        _widen(raw)
        raw["coronagraph"].pop("lyot")

    open_stop = _sim_from_yaml(DATA / "lyot_zernike.yaml", mutate=_no_stop)
    stopped_sum = lyot_sim.sample()["images"]["psf"][..., 0].sum()
    open_sum = open_stop.sample()["images"]["psf"][..., 0].sum()
    fp = open_stop.focal_planes["filter1"]
    assert stopped_sum < open_sum
    # Even without a stop, Babinet subtraction alone cannot create energy.
    assert open_sum < fp.reference_psf_sum
