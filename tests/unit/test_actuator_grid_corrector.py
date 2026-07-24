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
FIT_YAML_PATH = Path(__file__).parent / "data" / "actuator_grid_fit_dm.yaml"

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


# --- fit_surface -------------------------------------------------------------


def _masked_mean_removed(opd, mask):
    opd = np.asarray(opd, dtype=float).ravel()
    out = opd.copy()
    out[mask] -= opd[mask].mean()
    return out


def _low_order_screen(pupil_grid, rms_m=5e-8, seed=3):
    """Smooth low-order OPD screen (meters, Z2-Z6) an 8x8 DM can fit well.

    The Zernike disk is oversized (1.2 vs the 1.0 aperture) so the screen
    stays smooth over every aperture-mask pixel — HCIPy hard-zeroes modes
    outside their disk, and a step at the rim would be unfittable.
    """
    basis = hcipy.make_zernike_basis(5, 1.2, pupil_grid, starting_mode=2)
    rng = np.random.default_rng(seed)
    opd = sum(rng.normal() * np.asarray(m, dtype=float) for m in basis)
    return opd * (rms_m / np.std(opd))


def test_fit_surface_before_bind_raises():
    c = ActuatorGridCorrector(N_ACT, PITCH)
    with pytest.raises(RuntimeError):
        c.fit_surface(np.zeros(64 * 64))


def test_fit_surface_reproduces_opd(pupil_grid, aper_field):
    """fit -> set_actuators reproduces the (mean-removed) input OPD over the aperture."""
    c = _corrector(pupil_grid, aper_field)
    mask = np.asarray(aper_field, dtype=float).ravel() > 0
    opd = _low_order_screen(pupil_grid)

    fit = c.fit_surface(opd)
    c.set_actuators(fit)
    reproduced = 2.0 * np.asarray(c._dm.surface, dtype=float)

    target = _masked_mean_removed(opd, mask)
    err = np.std((reproduced - target)[mask])
    # Empirically ~12% rms: dominated by the aperture rim, where the mask
    # extends past the outermost actuator ring and the gaussian influence
    # basis rolls off. Interior fitting is at the percent level.
    assert err < 0.2 * np.std(target[mask])


def test_fit_surface_ignores_piston(pupil_grid, aper_field):
    """A uniform OPD offset is never commanded (aperture-masked mean subtraction)."""
    c = _corrector(pupil_grid, aper_field)
    # Fitting the offset naively would command ~offset/(2*scale) per actuator;
    # demand at least 9 orders of magnitude below that (roundoff territory).
    naive = 3.7e-7 / (2.0 * c.actuate_scale)
    fit = c.fit_surface(np.full(pupil_grid.size, 3.7e-7))
    np.testing.assert_allclose(fit, 0.0, atol=1e-9 * naive)
    # ...and an offset added to a structured screen changes nothing.
    opd = _low_order_screen(pupil_grid)
    np.testing.assert_allclose(c.fit_surface(opd + 2.2e-7), c.fit_surface(opd), atol=1e-9 * naive)


def test_fit_surface_matches_reference_lstsq(pupil_grid, aper_field):
    """The regularized solve agrees with a dense lstsq reference fit.

    Coefficients of poorly-illuminated (edge) actuators are not unique, so
    the comparison is on the reproduced surface over the aperture — the
    quantity the pipeline consumes — not on raw coefficients.
    """
    c = _corrector(pupil_grid, aper_field, actuate_scale=1.0)
    mask = np.asarray(aper_field, dtype=float).ravel() > 0
    opd = _masked_mean_removed(_low_order_screen(pupil_grid), mask)

    matrix = c._dm.influence_functions.transformation_matrix
    dense = np.asarray(matrix.toarray() if hasattr(matrix, "toarray") else matrix)
    ref_amps, _, _, _ = np.linalg.lstsq(dense[mask], opd[mask], rcond=None)

    fit_amps = 2.0 * c.fit_surface(opd)  # DM-facing amplitudes (no flips, scale 1)
    # Tikhonov vs minimum-norm differ only in the near-null space (dark edge
    # actuators); the reproduced surfaces agree to well below a picometer.
    ref_fit = dense[mask] @ ref_amps
    got_fit = dense[mask] @ fit_amps
    np.testing.assert_allclose(got_fit, ref_fit, rtol=0, atol=1e-12)


def test_fit_surface_unapplies_flips(pupil_grid, aper_field):
    """Fitted commands are caller-facing: flipped configs return flipped values
    but reproduce the identical surface through set_actuators."""
    opd = _low_order_screen(pupil_grid)
    plain = _corrector(pupil_grid, aper_field)
    flipped = _corrector(pupil_grid, aper_field, flip_x=True, flip_y=True)

    fit_plain = plain.fit_surface(opd).reshape(N_ACT, N_ACT)
    fit_flipped = flipped.fit_surface(opd).reshape(N_ACT, N_ACT)
    np.testing.assert_allclose(fit_flipped, np.flipud(np.fliplr(fit_plain)), rtol=1e-10)

    plain.set_actuators(fit_plain)
    flipped.set_actuators(fit_flipped)
    np.testing.assert_allclose(
        np.asarray(flipped._dm.surface), np.asarray(plain._dm.surface), rtol=1e-10
    )


def test_yaml_fit_role_cancels_atmosphere():
    """End-to-end: a fit-role actuator_grid DM corrects an atmosphere OPD.

    Pins the full chain — atmosphere seeds the cumulative-OPD stream,
    fit_surface matches it, the pipeline negates at the apply site — and
    checks the pipeline's fitted state equals an independent fit_surface
    call (pipeline wiring adds nothing beyond the documented negation).
    """
    from telescope_sim import TelescopeSim

    class OPDScreen:
        def __init__(self, opd_m):
            self._opd = np.asarray(opd_m, dtype=float).ravel()

        def phase_for(self, lam):
            return 2.0 * np.pi * self._opd / float(lam)

        def __call__(self, wf):
            field = hcipy.Field(
                np.asarray(wf.electric_field) * np.exp(1j * self.phase_for(wf.wavelength)),
                wf.grid,
            )
            return hcipy.Wavefront(field, wf.wavelength)

    sim = TelescopeSim.from_yaml(FIT_YAML_PATH)
    grid = hcipy.make_pupil_grid(64, 1.05)
    opd = _low_order_screen(grid, rms_m=5e-8)

    out = sim.sample(atmos=OPDScreen(opd), meas_strehl=True)
    corr = sim._c.correctors[0]

    # Pipeline sets -fit_surface(cumulative pre-self OPD) on the corrector.
    np.testing.assert_allclose(corr.actuators, -corr.fit_surface(opd), rtol=1e-10)

    # The DM's correction shrinks the aperture-masked residual OPD (~12%
    # rms survives, dominated by influence-basis rolloff at the rim).
    mask = corr._aperture_mask
    residual = _masked_mean_removed(opd, mask) + 2.0 * np.asarray(corr._dm.surface, dtype=float)
    assert np.std(residual[mask]) < 0.2 * np.std(opd[mask] - opd[mask].mean())
    # ~50nm rms in, few-nm residual out: strongly corrected PSF.
    assert out["strehls"]["filter1"] > 0.95


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
