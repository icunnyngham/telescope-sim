"""Unit tests for cumulative-OPD tracking + fit/residual-fit roles.

Headline regression: three identical Zernike DMs over a clean circular
aperture — dm1 and dm2 set to random actuators (`impose`), dm3 in the
`fit` role with `fit_source="cumulative_phase_pre_self"`. After the
pipeline negates the matching `fit_surface` result, dm3 exactly cancels
dm1 + dm2; PSF matches the at-rest reference to numerical precision.

The residual-fit target-strategy tests then exercise the Y-echo paths
that report wavefront state (legacy v1 semantics:
``out_actuate = caller + matching_fit(atmos)``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

from telescope_sim.config.loader import build, build_from_yaml
from telescope_sim.config.schema import SimConfig

YAML_PATH = Path(__file__).parent / "data" / "three_zernike_residual_fit.yaml"


def _load_yaml_dict() -> dict[str, Any]:
    with open(YAML_PATH) as f:
        return yaml.safe_load(f)


def _build_with_dm3_overrides(**dm3_updates: Any):
    """Build a sim from the headline YAML with dm3 fields overridden.

    Lets each test vary `wavefront_role`, `target_strategy`, etc. without
    duplicating the whole config.
    """
    data = _load_yaml_dict()
    data["correctors"]["dm3"].update(dm3_updates)
    cfg = SimConfig.model_validate(data)
    return build(cfg)


# ---------------------------------------------------------------------------
# Headline regression — the user's "three identical DMs, residual = 0" test
# ---------------------------------------------------------------------------


def test_three_identical_zernike_fit_cancels_impose():
    """dm3 with role=fit should cancel dm1+dm2 → PSF matches reference."""
    sim = build_from_yaml(YAML_PATH)

    rng = np.random.default_rng(0)
    a1 = rng.normal(scale=0.1, size=8)
    a2 = rng.normal(scale=0.1, size=8)

    res = sim.sample({"dm1": a1, "dm2": a2})

    # dm3's actuators end up at ~-(a1+a2): pipeline applied
    # -fit_surface(2*(S1+S2)). The atol=1e-5 (instead of machine
    # precision) reflects a tiny rank-1 perturbation introduced by
    # the pre-fit mean-subtract — Zernike modes Z_2..Z_n aren't
    # exactly zero-mean over a *discrete* aperture, so subtracting
    # the input mean shifts the lstsq solution by O(1e-6).
    np.testing.assert_allclose(res["actuations"]["dm3"], -(a1 + a2), atol=1e-5)

    # And the PSF should equal the at-rest reference, since residual
    # cumulative OPD across the chain is zero.
    ref = sim.focal_planes["filter1"].reference_psf
    psf = res["images"]["psf"][..., 0]  # drop the per-filter axis
    np.testing.assert_allclose(psf, ref, rtol=1e-10, atol=1e-10 * ref.max())


# ---------------------------------------------------------------------------
# fit_surface convention tests — should return matching (positive) values
# ---------------------------------------------------------------------------


def test_fit_surface_matches_zernike():
    """fit_surface(2*surface) returns the actuator vector that produced
    the surface — *matching*, not cancellation. The pipeline negates at
    apply sites; fit_surface itself reports wavefront state.
    """
    sim = build_from_yaml(YAML_PATH)
    dm = sim.correctors["dm1"]

    rng = np.random.default_rng(1)
    a = rng.normal(scale=0.1, size=8)
    dm.set_actuators(a)

    opd = 2.0 * np.asarray(dm._dm.surface)  # surface → OPD
    fit = dm.fit_surface(opd)

    # See test_three_identical_zernike_fit_cancels_impose for the
    # rationale behind the 1e-5 (instead of machine-precision)
    # tolerance.
    np.testing.assert_allclose(fit, a, atol=1e-5)


def test_fit_surface_matches_segmented_ptt():
    """Per-piston matching for SegmentedPTTCorrector.

    Uses the `elf_15seg` preset, synthesizes a piecewise-constant OPD
    field (one uniform value per disjoint segment — like an atmospheric
    disturbance fed to the legacy ``_measure_atmos_ptt``), and asserts
    ``fit_surface(OPD)[:, 0]`` recovers the input pistons with the
    global mean removed.

    Also exercises the fixed actuator round-trip: set caller-facing
    pistons via ``set_actuators``, read back via the ``actuators``
    property, confirm both agree on the block layout.
    """
    from telescope_sim.pipeline import TelescopeSim

    sim = TelescopeSim.from_preset("elf_15seg")
    seg = sim.correctors["segments"]

    rng = np.random.default_rng(2)
    n_seg = seg._n_segments
    piston_caller = rng.normal(scale=0.1, size=n_seg)

    # Round-trip check on the corrected layout: caller → HCIPy → caller
    ptt_in = np.zeros((n_seg, 3))
    ptt_in[:, 0] = piston_caller
    seg.set_actuators(ptt_in)
    np.testing.assert_allclose(seg.actuators, ptt_in, atol=1e-12)

    # Synthetic flat-per-segment OPD (each pixel of segment i gets
    # exactly ``2 * piston[i] * scale``; disjoint masks since
    # _bind_pupil_grid now argmax-assigns boundary pixels).
    opd = np.zeros(sim._c.pupil_grid.size, dtype=np.float64)
    for i, sp in enumerate(seg._segment_pixel_data):
        opd[sp["inds"]] = 2.0 * piston_caller[i] * seg.piston_scale

    fit = seg.fit_surface(opd)
    expected = piston_caller - piston_caller.mean()
    np.testing.assert_allclose(fit[:, 0], expected, atol=1e-10)
    # Tip/tilt fits should be ~zero (input has no tip/tilt component).
    assert np.max(np.abs(fit[:, 1:])) < 1e-10


# ---------------------------------------------------------------------------
# Target-strategy tests — the ML training semantic
# ---------------------------------------------------------------------------


def test_actuators_plus_residual_fit_reports_residual_error():
    """Y = c.actuators + fit_surface(cum_pre_self). Drives home the
    semantic: Y reports the residual wavefront error after dm3's own
    actuation. ML model is trained to predict this; controller applies
    -Y to drive corrections.
    """
    sim = _build_with_dm3_overrides(
        wavefront_role="actuate",
        target_strategy="actuators_plus_residual_fit",
        fit_source=None,  # default: cumulative_phase_pre_self
        target=True,
    )

    rng = np.random.default_rng(3)
    a1 = rng.normal(scale=0.1, size=8)
    a2 = rng.normal(scale=0.1, size=8)

    # (a) dm3 idle: Y reports the full disturbance
    res = sim.sample({"dm1": a1, "dm2": a2, "dm3": np.zeros(8)})
    np.testing.assert_allclose(res["actuations"]["dm3"], a1 + a2, atol=1e-5)

    # (b) dm3 perfectly cancels: Y reports zero residual
    res = sim.sample({"dm1": a1, "dm2": a2, "dm3": -(a1 + a2)})
    np.testing.assert_allclose(res["actuations"]["dm3"], np.zeros(8), atol=1e-5)

    # (c) dm3 imperfectly cancels — model overshoots/undershoots by `r`.
    # Y isolates the residual: actuators + matching = -(a1+a2) + r + (a1+a2) = r.
    r = rng.normal(scale=0.01, size=8)
    res = sim.sample({"dm1": a1, "dm2": a2, "dm3": -(a1 + a2) + r})
    np.testing.assert_allclose(res["actuations"]["dm3"], r, atol=1e-5)


def test_target_strategy_residual_fit_only():
    """Y = fit_surface(cum_pre_self). Reports cumulative disturbance
    regardless of dm3's own actuation (the `+ c.actuators` term is
    dropped vs. `actuators_plus_residual_fit`).
    """
    sim = _build_with_dm3_overrides(
        wavefront_role="actuate",
        target_strategy="residual_fit_only",
        fit_source=None,
        target=True,
    )

    rng = np.random.default_rng(4)
    a1 = rng.normal(scale=0.1, size=8)
    a2 = rng.normal(scale=0.1, size=8)
    arbitrary_a3 = rng.normal(scale=0.1, size=8)

    res = sim.sample({"dm1": a1, "dm2": a2, "dm3": arbitrary_a3})

    # Y is the matching fit of cum_pre_self = a1+a2, regardless of a3.
    np.testing.assert_allclose(res["actuations"]["dm3"], a1 + a2, atol=1e-5)


# ---------------------------------------------------------------------------
# fit_source variants
# ---------------------------------------------------------------------------


def test_fit_source_by_corrector_name():
    """fit_source pointing at an earlier corrector by name: dm3 should
    fit-and-negate dm1's surface, cancelling dm1 only (ignoring dm2).
    """
    sim = _build_with_dm3_overrides(
        wavefront_role="fit",
        fit_source="dm1",
        target_strategy="actuators",
        target=True,
    )

    rng = np.random.default_rng(5)
    a1 = rng.normal(scale=0.1, size=8)
    a2 = rng.normal(scale=0.1, size=8)

    res = sim.sample({"dm1": a1, "dm2": a2})

    # dm3 cancels dm1 (not dm2).
    np.testing.assert_allclose(res["actuations"]["dm3"], -a1, atol=1e-5)


def test_fit_source_forward_reference_raises():
    """fit_source naming a later corrector in the chain should raise."""
    # dm1 has wavefront_role=fit pointing at dm3 (which is later).
    data = _load_yaml_dict()
    data["correctors"]["dm1"]["wavefront_role"] = "fit"
    data["correctors"]["dm1"]["fit_source"] = "dm3"
    cfg = SimConfig.model_validate(data)
    sim = build(cfg)

    with pytest.raises(ValueError, match="later in the chain"):
        sim.sample({"dm2": np.zeros(8)})


def test_fit_source_unknown_name_raises():
    """fit_source naming a non-existent corrector should raise clearly."""
    sim = _build_with_dm3_overrides(
        wavefront_role="fit",
        fit_source="nonexistent_dm",
        target_strategy="actuators",
        target=True,
    )
    with pytest.raises(ValueError, match="unknown fit_source"):
        sim.sample({"dm1": np.zeros(8), "dm2": np.zeros(8)})


# ---------------------------------------------------------------------------
# Pre-fit mean-subtract tests — defensive against non-zero-mean inputs
# ---------------------------------------------------------------------------


def test_zernike_fit_surface_immune_to_constant_offset():
    """A constant offset added to the input must not leak into the
    non-piston Zernike modes. The corrector's ``starting_mode=2``
    means piston isn't in the basis, so without pre-fit mean-subtract
    the offset would distort modes 2..N.
    """
    sim = build_from_yaml(YAML_PATH)
    dm = sim.correctors["dm1"]

    rng = np.random.default_rng(10)
    a = rng.normal(scale=0.1, size=8)
    dm.set_actuators(a)
    opd_clean = 2.0 * np.asarray(dm._dm.surface)

    fit_clean = dm.fit_surface(opd_clean)
    # Add a large constant offset to the input.
    fit_offset = dm.fit_surface(opd_clean + 1.0e-5)

    # Same discretization-noise tolerance as test_fit_surface_matches_zernike.
    np.testing.assert_allclose(fit_clean, a, atol=1e-5)
    np.testing.assert_allclose(fit_offset, a, atol=1e-5)
    # The two recoveries should match each other to *machine* precision —
    # the offset gets cleanly removed before the lstsq, so the rest is
    # bit-identical.
    np.testing.assert_allclose(fit_offset, fit_clean, atol=1e-12)


def test_segmented_ptt_fit_surface_immune_to_constant_offset():
    """A constant offset added to the input must not change the
    recovered per-segment PTT. The existing post-fit mean removal
    already handled this; the new pre-fit subtract is idempotent
    with it, so behaviour is unchanged but verified.
    """
    from telescope_sim.pipeline import TelescopeSim

    sim = TelescopeSim.from_preset("elf_15seg")
    seg = sim.correctors["segments"]

    rng = np.random.default_rng(11)
    n_seg = seg._n_segments
    piston_caller = rng.normal(scale=0.1, size=n_seg)

    opd_clean = np.zeros(sim._c.pupil_grid.size, dtype=np.float64)
    for i, sp in enumerate(seg._segment_pixel_data):
        opd_clean[sp["inds"]] = 2.0 * piston_caller[i] * seg.piston_scale

    fit_clean = seg.fit_surface(opd_clean)
    fit_offset = seg.fit_surface(opd_clean + 1.0e-5)

    expected = piston_caller - piston_caller.mean()
    np.testing.assert_allclose(fit_clean[:, 0], expected, atol=1e-10)
    np.testing.assert_allclose(fit_offset[:, 0], expected, atol=1e-10)
    np.testing.assert_allclose(fit_offset, fit_clean, atol=1e-12)
