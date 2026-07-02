"""Tests for ``ActuatorGridCorrector`` — geometry, misalignment, and wiring.

The misalignment semantics are the point of this corrector, so the
geometric conventions are pinned explicitly:

  - A shaped ``(N, N)`` command indexes ``cmd[iy, ix]`` — axis 0 walks y
    ascending, axis 1 walks x ascending, row-major flattening matches
    HCIPy's actuator-position ordering (x varies fastest).
  - Positive ``rotation_deg`` rotates the DM counterclockwise relative
    to the pupil (x right, y up): a poked actuator at lattice position p
    renders its surface bump at R(+theta) @ p.
  - ``flip_x`` / ``flip_y`` mirror the command indexing (``fliplr`` /
    ``flipud``) before the rotated geometry renders it (flip-then-rotate),
    and the ``actuators`` readback un-applies them (caller round-trip).

Geometry tests use the gaussian influence model (non-negative pokes, so
the surface-weighted centroid is well-defined); xinetics gets a smoke test.
"""

from __future__ import annotations

from pathlib import Path

import hcipy
import numpy as np
import pytest
from hcipy.optics.deformable_mirror import make_actuator_positions

from telescope_sim.correctors.actuator_grid import ActuatorGridCorrector
from telescope_sim.registry import lookup

YAML_PATH = Path(__file__).parent / "data" / "actuator_grid_dm.yaml"

N_ACT = 8
PITCH = 1.0 / 8


@pytest.fixture(scope="module")
def pupil_grid():
    return hcipy.make_pupil_grid(64, 1.05)


@pytest.fixture(scope="module")
def aper_field(pupil_grid):
    aper_callable = hcipy.make_circular_aperture(1.0)
    return hcipy.evaluate_supersampled(aper_callable, pupil_grid, 16)


def _corrector(pupil_grid, aper_field, **kwargs):
    kwargs.setdefault("actuate_scale", 1.0e-7)
    c = ActuatorGridCorrector(N_ACT, PITCH, **kwargs)
    c._bind_pupil_grid(pupil_grid, aper_field)
    return c


def _lattice_position(ix, iy):
    """Unrotated lattice position for command index ``(iy, ix)``."""
    offset = -0.5 * N_ACT * PITCH + 0.5 * PITCH
    return np.array([offset + ix * PITCH, offset + iy * PITCH])


def _poke(corrector, ix, iy, value=1.0):
    cmd = np.zeros((N_ACT, N_ACT))
    cmd[iy, ix] = value
    corrector.set_actuators(cmd)


def _surface_centroid(corrector, grid):
    """Surface-weighted centroid of the DM surface (gaussian pokes >= 0)."""
    w = np.asarray(corrector._dm.surface, dtype=float)
    total = w.sum()
    assert total > 0, "expected a non-zero DM surface"
    return np.array([(w * grid.x).sum() / total, (w * grid.y).sum() / total])


def _rot(theta_deg):
    t = np.deg2rad(theta_deg)
    return np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])


# --- Registry / construction / binding -------------------------------------


def test_registry_lookup():
    assert lookup("corrector", "actuator_grid") is ActuatorGridCorrector


def test_unknown_influence_raises():
    with pytest.raises(ValueError, match="influence"):
        ActuatorGridCorrector(N_ACT, PITCH, influence="bogus")


def test_unbound_state(pupil_grid):
    c = ActuatorGridCorrector(N_ACT, PITCH)
    assert c.n_actuators == N_ACT**2
    np.testing.assert_array_equal(c.actuators, np.zeros(N_ACT**2))
    with pytest.raises(RuntimeError):
        c.apply(hcipy.Wavefront(hcipy.Field(np.ones(pupil_grid.size), pupil_grid)))
    with pytest.raises(RuntimeError):
        c.set_actuators(np.zeros(N_ACT**2))
    c.flatten()  # no-op before bind, must not raise


def test_command_axis_convention():
    """Pin the documented (iy, ix) mapping against HCIPy's actuator ordering.

    HCIPy places actuators via ``make_uniform_grid`` (x varies fastest), so
    flat index ``iy * N + ix`` — i.e. row-major flattening of a command
    whose axis 0 is y ascending and axis 1 is x ascending.
    """
    positions = make_actuator_positions(N_ACT, PITCH)
    for iy, ix in [(0, 0), (0, 5), (4, 1), (7, 7)]:
        k = iy * N_ACT + ix
        np.testing.assert_allclose(
            [positions.x[k], positions.y[k]], _lattice_position(ix, iy), atol=1e-12
        )


def test_xinetics_influence_constructs(pupil_grid, aper_field):
    c = _corrector(pupil_grid, aper_field, influence="xinetics")
    _poke(c, 4, 3)
    assert np.max(np.abs(np.asarray(c._dm.surface))) > 0


# --- Poke geometry ----------------------------------------------------------


def test_poke_localizes_at_actuator_position(pupil_grid, aper_field):
    c = _corrector(pupil_grid, aper_field)
    ix, iy = 5, 4
    _poke(c, ix, iy)
    np.testing.assert_allclose(
        _surface_centroid(c, pupil_grid), _lattice_position(ix, iy), atol=5e-3
    )


def test_surface_scales_linearly_with_actuate_scale(pupil_grid, aper_field):
    c1 = _corrector(pupil_grid, aper_field, actuate_scale=1.0e-7)
    c2 = _corrector(pupil_grid, aper_field, actuate_scale=2.0e-7)
    _poke(c1, 5, 4)
    _poke(c2, 5, 4)
    np.testing.assert_allclose(
        2.0 * np.asarray(c1._dm.surface), np.asarray(c2._dm.surface), rtol=1e-12
    )
    # And linear in the command value on a single corrector.
    _poke(c1, 5, 4, value=3.0)
    np.testing.assert_allclose(
        np.asarray(c1._dm.surface), 3.0 * 0.5 * np.asarray(c2._dm.surface), rtol=1e-12
    )


def test_rotation_90_rotates_poke_counterclockwise(pupil_grid, aper_field):
    """The rotation sign pin: +90 deg maps lattice position p to R(+90) @ p."""
    c = _corrector(pupil_grid, aper_field, rotation_deg=90.0)
    ix, iy = 5, 4  # deliberately off-center, x != y so the sign is unambiguous
    _poke(c, ix, iy)
    expected = _rot(90.0) @ _lattice_position(ix, iy)
    np.testing.assert_allclose(_surface_centroid(c, pupil_grid), expected, atol=5e-3)


def test_flip_x_mirrors_bump(pupil_grid, aper_field):
    c = _corrector(pupil_grid, aper_field, flip_x=True)
    ix, iy = 5, 4
    _poke(c, ix, iy)
    x0, y0 = _lattice_position(ix, iy)
    np.testing.assert_allclose(_surface_centroid(c, pupil_grid), [-x0, y0], atol=5e-3)


def test_flip_y_mirrors_bump(pupil_grid, aper_field):
    c = _corrector(pupil_grid, aper_field, flip_y=True)
    ix, iy = 5, 4
    _poke(c, ix, iy)
    x0, y0 = _lattice_position(ix, iy)
    np.testing.assert_allclose(_surface_centroid(c, pupil_grid), [x0, -y0], atol=5e-3)


def test_flip_then_rotate_composition(pupil_grid, aper_field):
    """flip_x + rotation: the command is mirrored FIRST, then rotated."""
    c = _corrector(pupil_grid, aper_field, flip_x=True, rotation_deg=90.0)
    ix, iy = 5, 4
    _poke(c, ix, iy)
    x0, y0 = _lattice_position(ix, iy)
    expected = _rot(90.0) @ np.array([-x0, y0])  # flip-then-rotate
    np.testing.assert_allclose(_surface_centroid(c, pupil_grid), expected, atol=5e-3)


# --- Command shapes / readback ----------------------------------------------


def test_flat_and_shaped_commands_equivalent(pupil_grid, aper_field):
    c = _corrector(pupil_grid, aper_field)
    rng = np.random.default_rng(0)
    cmd = rng.normal(size=(N_ACT, N_ACT))
    c.set_actuators(cmd)
    from_shaped = np.asarray(c._dm.actuators).copy()
    c.set_actuators(cmd.reshape(-1))
    np.testing.assert_array_equal(from_shaped, np.asarray(c._dm.actuators))


def test_readback_round_trip(pupil_grid, aper_field):
    c = _corrector(pupil_grid, aper_field)
    rng = np.random.default_rng(1)
    cmd = rng.normal(size=N_ACT**2)
    c.set_actuators(cmd)
    np.testing.assert_allclose(c.actuators, cmd, rtol=1e-12)


def test_readback_round_trip_with_flips(pupil_grid, aper_field):
    """Flips mirror the DM-facing command, but callers read back their own values."""
    c = _corrector(pupil_grid, aper_field, flip_x=True, flip_y=True)
    rng = np.random.default_rng(2)
    cmd = rng.normal(size=(N_ACT, N_ACT))
    c.set_actuators(cmd)
    np.testing.assert_allclose(c.actuators, cmd.reshape(-1), rtol=1e-12)
    # The internal DM state really is the mirrored command.
    internal = np.asarray(c._dm.actuators).reshape(N_ACT, N_ACT) / c.actuate_scale
    np.testing.assert_allclose(internal, np.flipud(np.fliplr(cmd)), rtol=1e-12)


def test_wrong_size_raises(pupil_grid, aper_field):
    c = _corrector(pupil_grid, aper_field)
    with pytest.raises(ValueError, match="actuators"):
        c.set_actuators(np.zeros(N_ACT**2 - 1))
    with pytest.raises(ValueError, match="actuators"):
        c.set_actuators(np.zeros((N_ACT, N_ACT - 1)))


# --- YAML round-trip ---------------------------------------------------------


def test_yaml_round_trip_actuation_changes_image():
    from telescope_sim import TelescopeSim

    sim = TelescopeSim.from_yaml(YAML_PATH)
    out_rest = sim.sample()
    img_rest = out_rest["images"]["psf"]

    cmd = np.zeros((8, 8))
    cmd[3, 4] = 0.5
    cmd[5, 2] = -0.3
    out_poke = sim.sample(actuations={"dm": cmd})
    img_poke = out_poke["images"]["psf"]

    assert img_poke.shape == img_rest.shape
    assert np.max(np.abs(img_poke - img_rest)) > 1e-6 * np.max(img_rest)
    # target_strategy=actuators echoes the caller-facing command back.
    np.testing.assert_allclose(out_poke["actuations"]["dm"], cmd.reshape(-1), rtol=1e-12)
