"""Unit tests for the ``fourier`` corrector (Fourier-mode DM).

Pins the mode-count formula from the module docstring, piston removal
and truncation, peak normalization, the caller-units round trip, and the
diagnostic frequency labeling. Fit-surface behavior is covered by the
cross-kind contract suite (``test_corrector_fit_contract.py``).
"""

from __future__ import annotations

import hcipy
import numpy as np
import pytest

from telescope_sim.config.loader import build
from telescope_sim.config.schema import SimConfig
from telescope_sim.correctors.fourier import FourierCorrector
from telescope_sim.registry import lookup


@pytest.fixture(scope="module")
def pupil():
    grid = hcipy.make_pupil_grid(64, 1.05)
    field = hcipy.evaluate_supersampled(hcipy.make_circular_aperture(1.0), grid, 8)
    return grid, field


def _bound(pupil, **kwargs) -> FourierCorrector:
    grid, field = pupil
    corr = FourierCorrector(**kwargs)
    corr._bind_pupil_grid(grid, field)
    return corr


def test_registered_as_fourier():
    assert lookup("corrector", "fourier") is FourierCorrector


@pytest.mark.parametrize(
    ("n_axis", "raw_count"),
    [(4, 23), (6, 47), (8, 79)],  # n^2 + 2n - 1 on an even-parity grid
)
def test_mode_count_formula(pupil, n_axis, raw_count):
    assert _bound(pupil, n_axis=n_axis, remove_piston=False).n_actuators == raw_count
    assert _bound(pupil, n_axis=n_axis).n_actuators == raw_count - 1


def test_default_is_46_modes(pupil):
    assert _bound(pupil).n_actuators == 46


def test_piston_mode_is_dc_and_removed_by_default(pupil):
    kept = _bound(pupil, remove_piston=False)
    grid, field = pupil
    mode0 = np.asarray(kept.basis[0])[np.asarray(field) > 0]
    np.testing.assert_allclose(mode0, mode0[0])  # constant inside the aperture
    assert kept.mode_frequencies[0] == 0.0
    dropped = _bound(pupil)
    assert dropped.mode_frequencies[0] > 0.0
    assert dropped.n_actuators == kept.n_actuators - 1


def test_truncation_keeps_the_lowest_frequencies(pupil):
    corr = _bound(pupil, n_axis=6, n_modes=10)
    assert corr.n_actuators == 10
    assert corr.actuators.shape == (10,)
    assert len(corr.mode_frequencies) == 10
    full = _bound(pupil, n_axis=6)
    np.testing.assert_allclose(corr.mode_frequencies, full.mode_frequencies[:10])
    assert np.all(np.diff(full.mode_frequencies) >= 0)


def test_truncation_beyond_available_modes_raises(pupil):
    with pytest.raises(ValueError, match="increase n_axis"):
        _bound(pupil, n_axis=4, n_modes=30)


def test_modes_are_peak_normalized(pupil):
    corr = _bound(pupil)
    for mode in corr.basis:
        assert np.max(np.abs(np.asarray(mode))) == pytest.approx(1.0)


def test_actuator_round_trip_and_scale(pupil):
    corr = _bound(pupil, actuate_scale=2.5e-7)
    values = np.random.default_rng(0).normal(size=corr.n_actuators)
    corr.set_actuators(values)
    np.testing.assert_allclose(corr.actuators, values, rtol=1e-12)
    np.testing.assert_allclose(corr._dm.actuators, values * 2.5e-7, rtol=1e-12)
    # Peak surface of a unit single-mode command is actuate_scale.
    corr.set_actuators(np.eye(corr.n_actuators)[3])
    assert np.max(np.abs(np.asarray(corr._dm.surface))) == pytest.approx(2.5e-7)
    corr.flatten()
    assert np.all(corr.actuators == 0)
    with pytest.raises(ValueError, match="expected"):
        corr.set_actuators(np.zeros(corr.n_actuators + 1))


def test_builds_from_yaml_and_samples():
    data = {
        "pupil": {"resolution": 64, "extent": 1.05},
        "aperture": {
            "type": "external_pupil",
            "module": "hcipy",
            "function": "make_circular_aperture",
            "mode": "callable",
            "kwargs": {"diameter": 1.0},
            "area": 0.785,
        },
        "correctors": {
            "dm": {
                "type": "fourier",
                "n_axis": 4,
                "actuate_scale": 1.0e-7,
                "target_strategy": "actuators",
                "target": True,
            }
        },
        "corrector_chain": ["dm"],
        "focal_planes": {
            "filter1": {
                "type": "angular",
                "central_lam": 1.0e-6,
                "focal_extent": 2.0,
                "focal_res": 64,
                "fractional_bandwidth": 0.0,
                "num_samples": 1,
            }
        },
        "outputs": {"psf": {"tap": {"type": "intensity", "focal_planes": ["filter1"]}}},
        "strehl_core_rad": None,
    }
    sim = build(SimConfig.model_validate(data))
    dm = sim._c.correctors[0]
    assert dm.n_actuators == 22
    acts = np.random.default_rng(1).normal(size=22)
    out = sim.sample({"dm": acts})
    np.testing.assert_allclose(out["actuations"]["dm"], acts, rtol=1e-12)
    ref = np.asarray(sim.focal_planes["filter1"].reference_psf)
    assert np.max(np.abs(out["images"]["psf"][..., 0] - ref)) / ref.max() > 1e-3
