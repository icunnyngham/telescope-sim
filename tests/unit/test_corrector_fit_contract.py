"""Cross-kind contract tests for ``Corrector.fit_surface`` implementations.

Every corrector that overrides ``fit_surface`` must honor the same
convention (see :meth:`telescope_sim.abc.Corrector.fit_surface`):

- input is pupil-plane OPD in **meters** (path length);
- output is **matching** caller-facing actuator values — setting them
  reproduces the input OPD as the corrector's surface contribution,
  including the surface→OPD round-trip factor of 2 and the caller-units
  scaling;
- a uniform OPD offset is never commanded (aperture-masked mean
  subtraction): a global phase is unobservable, and each basis's
  imperfect approximation of one would print through as an observable
  artifact.

Implementations are free to differ in solver (dense lstsq, sparse
Tikhonov normal equations, per-segment projection) — this suite pins the
*behavior*, not the algorithm. A guard test fails if a registered
corrector kind grows a ``fit_surface`` override without a contract case
here, so the suite stays exhaustive as kinds are added.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from telescope_sim.abc import Corrector
from telescope_sim.config.loader import build
from telescope_sim.config.schema import SimConfig
from telescope_sim.registry import available, lookup

_CIRC_APERTURE = {
    "type": "external_pupil",
    "module": "hcipy",
    "function": "make_circular_aperture",
    "mode": "callable",
    "kwargs": {"diameter": 1.0},
    "area": 0.7853981633974483,
}

_SEG_APERTURE = {
    "type": "segmented_circular",
    "segment_diameter": 0.3,
    "layout": "elf",
    "n_segments": 6,
    "ring_radius": 0.33,
    "supersample": 4,
}

# One case per corrector kind implementing fit_surface: aperture config,
# corrector config, and the config fields carrying caller-units scaling
# (used by the scale-linearity test).
CASES: dict[str, dict[str, Any]] = {
    "zernike": {
        "aperture": _CIRC_APERTURE,
        "corrector": {
            "type": "zernike",
            "n_modes": 8,
            "zernike_diameter": 1.0,
            "actuate_scale": 1.0e-7,
        },
        "scale_fields": ["actuate_scale"],
        "roundtrip_tol": 0.02,
    },
    "actuator_grid": {
        "aperture": _CIRC_APERTURE,
        "corrector": {
            "type": "actuator_grid",
            "num_actuators": 8,
            "actuator_pitch": 0.125,
            "actuate_scale": 1.0e-7,
        },
        "scale_fields": ["actuate_scale"],
        "roundtrip_tol": 0.02,
    },
    "segmented_ptt": {
        "aperture": _SEG_APERTURE,
        "corrector": {
            "type": "segmented_ptt",
            "piston_scale": 1.0e-6,
            "tip_tilt_scale": 1.0e-6,
        },
        "scale_fields": ["piston_scale", "tip_tilt_scale"],
        # The own-surface probe is anti-aliased: rim pixels carry
        # fractional surface, dragging the (legacy-faithful, unweighted)
        # per-segment lstsq to ~0.9 recovery — ~17% round-trip rms at this
        # resolution. Decisive convention checks for this kind live in
        # test_segmented_ptt_recovers_global_ramp (exact for smooth input).
        "roundtrip_tol": 0.25,
    },
}


def _build_case(kind: str, scale_factor: float = 1.0):
    """Build a one-corrector sim for a contract case; return (corrector, aperture_field)."""
    case = CASES[kind]
    corrector = dict(case["corrector"])
    for field in case["scale_fields"]:
        corrector[field] = corrector.get(field, 1.0) * scale_factor
    data = {
        "pupil": {"resolution": 64, "extent": 1.05},
        "aperture": case["aperture"],
        "correctors": {"c": corrector},
        "corrector_chain": ["c"],
        "focal_planes": {
            "filter1": {
                "type": "angular",
                "central_lam": 1.0e-6,
                "focal_extent": 1.0e-5,
                "focal_res": 32,
                "fractional_bandwidth": 0.0,
                "num_samples": 1,
            }
        },
        "outputs": {
            "psf": {"tap": {"type": "intensity", "focal_planes": ["filter1"]}},
        },
        "strehl_core_rad": None,
    }
    sim = build(SimConfig.model_validate(data))
    corr = sim._c.correctors[0]
    field = np.asarray(sim.aperture.field, dtype=float).ravel()
    return corr, field


@pytest.fixture(scope="module", params=sorted(CASES))
def fit_case(request):
    corr, field = _build_case(request.param)
    return request.param, corr, field


def _surface_opd(corrector) -> np.ndarray:
    mirror = getattr(corrector, "_dm", None) or getattr(corrector, "_sm", None)
    return 2.0 * np.asarray(mirror.surface, dtype=float)


def _own_surface_opd(corrector, scale=0.1, seed=11) -> np.ndarray:
    """A guaranteed-fittable OPD: the corrector's own surface from random commands."""
    rng = np.random.default_rng(seed)
    corrector.set_actuators(rng.normal(scale=scale, size=corrector.n_actuators))
    opd = _surface_opd(corrector)
    corrector.flatten()
    return opd


def test_all_fit_capable_kinds_have_contract_cases():
    """Guard: a corrector kind overriding fit_surface MUST appear in CASES."""
    fit_capable = {
        name
        for name in available("corrector")
        if lookup("corrector", name).fit_surface is not Corrector.fit_surface
    }
    assert fit_capable == set(CASES), (
        "corrector kinds overriding fit_surface must have a contract case in "
        f"test_corrector_fit_contract.CASES (missing: {fit_capable - set(CASES)}, "
        f"stale: {set(CASES) - fit_capable})"
    )


def test_fit_reproduces_surface_with_round_trip_factor(fit_case):
    """fit → set_actuators reproduces the input OPD's observable content.

    Catches: wrong sign (matching vs cancelling), a missing/extra factor
    of 2 (surface vs OPD), missing caller-units scaling — each inflates
    the mismatch by ≥2x, far beyond the tolerance.
    """
    kind, corr, field = fit_case
    mask = field > 0
    opd = _own_surface_opd(corr)

    fit = np.asarray(corr.fit_surface(opd), dtype=float).reshape(-1)
    corr.set_actuators(fit)
    reproduced = _surface_opd(corr)
    corr.flatten()

    diff = reproduced - opd
    diff = diff - diff[mask].mean()  # global offset is unobservable
    ref = opd - opd[mask].mean()
    assert np.std(diff[mask]) < CASES[kind]["roundtrip_tol"] * np.std(ref[mask]), kind


def test_constant_opd_fits_to_zero(fit_case):
    """Piston is never commanded: a uniform OPD produces ~zero actuators."""
    kind, corr, field = fit_case
    offset = 3.7e-7
    fit = np.asarray(corr.fit_surface(np.full(field.size, offset)), dtype=float)
    # Fitting the offset naively would command ~offset / (2 * scale);
    # demand at least 6 orders of magnitude below that.
    naive = offset / 2.0e-6  # most sensitive caller-units scale in CASES
    assert np.max(np.abs(fit)) < 1e-6 * naive, kind


def test_global_offset_does_not_change_fit(fit_case):
    """fit(opd + c) == fit(opd): mean subtraction on structured input."""
    kind, corr, field = fit_case
    opd = _own_surface_opd(corr)
    f0 = np.asarray(corr.fit_surface(opd), dtype=float)
    f1 = np.asarray(corr.fit_surface(opd + 2.2e-7), dtype=float)
    scale_ref = max(np.max(np.abs(f0)), 1e-12)
    np.testing.assert_allclose(f1, f0, rtol=1e-7, atol=1e-7 * scale_ref, err_msg=kind)


def test_segmented_ptt_recovers_global_ramp():
    """A global tilt plane is exactly per-segment PTT; recovery is analytic.

    A linear function is fit exactly by per-segment lstsq regardless of
    pixel weighting, so this pins the segmented kind's tip/tilt axis
    conventions and piston vs tip/tilt scaling tightly — the checks the
    anti-aliased own-surface round-trip can't do.
    """
    corr, field = _build_case("segmented_ptt")
    grid = corr._sm.input_grid
    a, b = 3.0e-7, -1.7e-7  # OPD slopes, m per pupil-plane meter
    ramp = a * np.asarray(grid.x) + b * np.asarray(grid.y)

    fit = np.asarray(corr.fit_surface(ramp), dtype=float).reshape(-1, 3)
    tts = corr.tip_tilt_scale
    np.testing.assert_allclose(fit[:, 1], a / (2.0 * tts), rtol=1e-6)
    np.testing.assert_allclose(fit[:, 2], b / (2.0 * tts), rtol=1e-6)
    # Pistons follow the segment centers (global mean removed).
    centers = np.asarray(corr.segment_coords, dtype=float)
    expected_p = a * centers[:, 0] + b * centers[:, 1]
    expected_p = (expected_p - expected_p.mean()) / (2.0 * corr.piston_scale)
    np.testing.assert_allclose(fit[:, 0], expected_p, rtol=1e-3, atol=1e-6)


@pytest.mark.parametrize("kind", sorted(CASES))
def test_fit_values_scale_inversely_with_actuate_scale(kind):
    """Caller-units convention: 10x the scale fields → fit values / 10."""
    corr1, _ = _build_case(kind, scale_factor=1.0)
    corr10, _ = _build_case(kind, scale_factor=10.0)
    opd = _own_surface_opd(corr1)
    f1 = np.asarray(corr1.fit_surface(opd), dtype=float)
    f10 = np.asarray(corr10.fit_surface(opd), dtype=float)
    np.testing.assert_allclose(f10, f1 / 10.0, rtol=1e-6, atol=1e-12 * np.max(np.abs(f1)))
