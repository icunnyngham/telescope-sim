"""Tests for the v1.x SimulateMultiApertureTelescope compatibility shim."""

from __future__ import annotations

import warnings

import pytest


def test_shim_emits_deprecation_warning():
    from telescope_sim.legacy import SimulateMultiApertureTelescope

    with pytest.warns(DeprecationWarning, match="deprecated"):
        SimulateMultiApertureTelescope(
            mirror_layout="elf",
            telescope_radius=1.25,
            sub_aperture_count=15,
        )


def test_shim_elf_get_observation_returns_xy():
    from telescope_sim.legacy import SimulateMultiApertureTelescope

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        sim = SimulateMultiApertureTelescope(
            mirror_layout="elf",
            telescope_radius=1.25,
            sub_aperture_count=15,
            pupil_res=64,
            focal_res=32,
            filter_num_samples=1,
        )
    x, y = sim.get_observation()
    assert x.ndim == 3
    assert y.shape == (15, 3)


def test_shim_elf_with_strehl():
    from telescope_sim.legacy import SimulateMultiApertureTelescope

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        sim = SimulateMultiApertureTelescope(
            mirror_layout="elf",
            telescope_radius=1.25,
            sub_aperture_count=15,
            pupil_res=64,
            focal_res=32,
            filter_num_samples=1,
        )
    x, y, strehls = sim.get_observation(meas_strehl=True)
    assert strehls.shape == (1,)
    # at-rest strehl is 1.0
    assert pytest.approx(1.0) == float(strehls[0])


def test_shim_monolithic():
    from telescope_sim.legacy import SimulateMultiApertureTelescope

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        sim = SimulateMultiApertureTelescope(
            mirror_layout="monolithic",
            telescope_radius=1.8,
            pupil_res=64,
            focal_res=32,
            filter_num_samples=1,
        )
    x, y = sim.get_observation()
    assert y.shape == (1, 3)  # one segment


def test_shim_rejects_unsupported_kwarg():
    from telescope_sim.legacy import SimulateMultiApertureTelescope

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        with pytest.raises(NotImplementedError, match="aren't yet supported"):
            SimulateMultiApertureTelescope(
                mirror_layout="elf",
                telescope_radius=1.25,
                sub_aperture_count=15,
                dm_actuator_num=11,  # not supported by shim
            )


def test_shim_rejects_unknown_layout():
    from telescope_sim.legacy import SimulateMultiApertureTelescope

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        with pytest.raises(NotImplementedError, match="not yet supported"):
            SimulateMultiApertureTelescope(mirror_layout="keck")
