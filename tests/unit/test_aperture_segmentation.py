"""Pupil segmentation by connected components, and ``segmented_ptt`` on top.

Covers :mod:`telescope_sim.apertures.segmentation` (the partitioner and
its merge / refusal rules), the loader's ``aperture.segmentation`` block,
and the ``segmented_ptt`` corrector driving a spider-partitioned
monolithic pupil — the low-wind-effect / petal-mode configuration —
including the ``piston_only`` variant.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import hcipy
import numpy as np
import pytest

from telescope_sim.apertures.segmentation import partition_connected_components
from telescope_sim.config.loader import build
from telescope_sim.config.schema import SimConfig

STUB_PUPIL = str(Path(__file__).parent / "data" / "stub_pupil.py")


def _config(
    *,
    stub: bool = False,
    segmentation: dict[str, Any] | None = None,
    corrector: dict[str, Any] | None = None,
) -> dict[str, Any]:
    aperture: dict[str, Any] = {
        "type": "external_pupil",
        "module": STUB_PUPIL,
        "function": "generate_pupil",
        "mode": "field",
        "kwargs": {"stub": stub},
        "area": 0.75,
    }
    if segmentation is not None:
        aperture["segmentation"] = segmentation
    correctors = {} if corrector is None else {"petals": corrector}
    return {
        "pupil": {"resolution": 96, "extent": 1.05},
        "aperture": aperture,
        "correctors": correctors,
        "corrector_chain": list(correctors),
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


def _build(**kwargs):
    return build(SimConfig.model_validate(_config(**kwargs)))


def _ptt(*, piston_scale=1.0e-7, piston_only=False, **extra) -> dict[str, Any]:
    return {
        "type": "segmented_ptt",
        "piston_scale": piston_scale,
        "tip_tilt_scale": 1.0e-7,
        "piston_only": piston_only,
        "wavefront_role": "impose",
        "target_strategy": "actuators",
        "target": True,
        **extra,
    }


# --- The partitioner ----------------------------------------------------------


@pytest.fixture(scope="module")
def grid_and_fields():
    grid = hcipy.make_pupil_grid(96, 1.05)
    import importlib.util

    spec = importlib.util.spec_from_file_location("stub_pupil", STUB_PUPIL)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return grid, mod.generate_pupil(grid), mod.generate_pupil(grid, stub=True)


def test_four_vane_pupil_partitions_into_four_angle_ordered_segments(grid_and_fields):
    grid, clean, _ = grid_and_fields
    res = partition_connected_components(grid, clean, n_segments=4)

    assert res.n_raw_regions == 4
    assert res.merge_info == []
    assert len(res.segments) == 4
    assert res.segment_coords.shape == (4, 2)
    # Ascending polar angle from the origin: (-,-), (+,-), (+,+), (-,+).
    signs = np.sign(res.segment_coords)
    np.testing.assert_array_equal(signs, [[-1, -1], [1, -1], [1, 1], [-1, 1]])
    angles = np.arctan2(res.segment_coords[:, 1], res.segment_coords[:, 0])
    assert np.all(np.diff(angles) > 0)

    masks = np.column_stack([np.asarray(s) for s in res.segments])
    assert set(np.unique(masks)) == {0.0, 1.0}
    assert np.all(masks.sum(axis=1) <= 1)  # disjoint
    np.testing.assert_array_equal(masks.sum(axis=1) > 0, np.asarray(clean) > 0)  # covering


def test_requesting_more_segments_than_regions_raises(grid_and_fields):
    grid, clean, _ = grid_and_fields
    with pytest.raises(ValueError, match="found 4 connected region"):
        partition_connected_components(grid, clean, n_segments=5)


def test_fragment_is_merged_into_its_nearest_segment(grid_and_fields):
    grid, clean, stubbed = grid_and_fields
    res = partition_connected_components(grid, stubbed, n_segments=4)

    assert res.n_raw_regions == 5
    assert len(res.segments) == 4
    assert len(res.merge_info) == 1
    (rec,) = res.merge_info
    # The stub vane lives in the first quadrant (+,+): index 2 in angle order.
    assert rec["segment_index"] == 2
    assert rec["fragment_size_px"] > 0
    cx, cy = rec["fragment_centroid"]
    assert cx > 0 and cy > 0

    masks = np.column_stack([np.asarray(s) for s in res.segments])
    np.testing.assert_array_equal(masks.sum(axis=1) > 0, np.asarray(stubbed) > 0)
    # Merged quadrant equals the clean quadrant minus the stub vane's pixels.
    clean_masks = np.column_stack(
        [np.asarray(s) for s in partition_connected_components(grid, clean, n_segments=4).segments]
    )
    assert np.all(masks[:, 2] <= clean_masks[:, 2])
    assert masks[:, 2].sum() > 0.9 * clean_masks[:, 2].sum()


def test_merge_fragments_false_refuses_extra_regions(grid_and_fields):
    grid, _, stubbed = grid_and_fields
    with pytest.raises(ValueError, match="merge_fragments"):
        partition_connected_components(grid, stubbed, n_segments=4, merge_fragments=False)


def test_threshold_controls_what_counts_as_lit(grid_and_fields):
    grid, clean, _ = grid_and_fields
    lo = partition_connected_components(grid, clean, n_segments=4, threshold=0.0)
    hi = partition_connected_components(grid, clean, n_segments=4, threshold=0.5)
    n_lo = sum(int(np.asarray(s).sum()) for s in lo.segments)
    n_hi = sum(int(np.asarray(s).sum()) for s in hi.segments)
    assert n_hi < n_lo  # anti-aliased rim pixels dropped
    assert n_hi == int((np.asarray(clean) > 0.5).sum())


# --- Loader integration -------------------------------------------------------


def test_loader_fills_segments_from_the_block():
    sim = _build(segmentation={"n_segments": 4}, corrector=_ptt())
    petals = sim._c.correctors[0]
    assert petals.n_segments == 4
    assert petals.n_actuators == 12
    assert petals.segment_coords.shape == (4, 2)


def test_loader_records_segmentation_metadata():
    sim = _build(stub=True, segmentation={"n_segments": 4}, corrector=_ptt())
    petals = sim._c.correctors[0]
    assert petals.n_segments == 4
    # Masks are the same whether read through the corrector or the partitioner.
    assert len(petals._sm.segments) == 4


def test_segmentation_block_on_a_segmented_aperture_is_refused():
    data = _config(segmentation={"n_segments": 4})
    data["aperture"] = {
        "type": "segmented_circular",
        "segment_diameter": 0.3,
        "layout": "elf",
        "n_segments": 6,
        "ring_radius": 0.33,
        "supersample": 4,
        "segmentation": {"n_segments": 6},
    }
    with pytest.raises(ValueError, match="already defines segments"):
        build(SimConfig.model_validate(data))


def test_segmented_ptt_without_segments_still_errors_clearly():
    with pytest.raises(ValueError, match="requires aperture.segments"):
        _build(corrector=_ptt())


def test_segmentation_block_rejects_unknown_fields():
    with pytest.raises(ValueError):
        SimConfig.model_validate(_config(segmentation={"n_segments": 4, "angle": 45.0}))


# --- segmented_ptt on a partitioned pupil -------------------------------------


@pytest.fixture(scope="module")
def petal_sim():
    return _build(segmentation={"n_segments": 4}, corrector=_ptt())


def _psf(sim, acts):
    return np.asarray(sim.sample({"petals": acts})["images"]["psf"][..., 0], dtype=float)


def test_equal_pistons_are_invisible(petal_sim):
    ref = np.asarray(petal_sim.focal_planes["filter1"].reference_psf, dtype=float)
    equal = np.zeros((4, 3))
    equal[:, 0] = 3.0
    img = _psf(petal_sim, equal)
    np.testing.assert_allclose(img / ref.max(), ref / ref.max(), rtol=0, atol=1e-12)


def test_single_petal_piston_and_tip_kicks_are_distinct(petal_sim):
    ref = np.asarray(petal_sim.focal_planes["filter1"].reference_psf, dtype=float)
    piston = np.zeros((4, 3))
    piston[1, 0] = 2.0
    tip = np.zeros((4, 3))
    tip[1, 1] = 2.0
    img_p = _psf(petal_sim, piston)
    img_t = _psf(petal_sim, tip)
    assert np.max(np.abs(img_p - ref)) / ref.max() > 1e-3
    assert np.max(np.abs(img_t - ref)) / ref.max() > 1e-3
    assert np.max(np.abs(img_p - img_t)) / ref.max() > 1e-3


def test_fit_surface_recovers_commanded_petal_ptt(petal_sim):
    """Per-segment lstsq on binary petal masks is exact for the mirror's own surface."""
    petals = petal_sim._c.correctors[0]
    rng = np.random.default_rng(5)
    cmd = rng.normal(size=(4, 3))
    petals.set_actuators(cmd)
    opd = 2.0 * np.asarray(petals._sm.surface, dtype=float)
    petals.flatten()

    fit = np.asarray(petals.fit_surface(opd))
    expected = cmd.copy()
    expected[:, 0] -= expected[:, 0].mean()  # mean piston is unobservable
    np.testing.assert_allclose(fit, expected, rtol=1e-6, atol=1e-9)


def test_echo_reports_caller_facing_ptt(petal_sim):
    rng = np.random.default_rng(6)
    cmd = rng.normal(size=(4, 3))
    out = petal_sim.sample({"petals": cmd})
    np.testing.assert_allclose(out["actuations"]["petals"], cmd, rtol=1e-12)


# --- piston_only --------------------------------------------------------------


@pytest.fixture(scope="module")
def piston_only_sim():
    return _build(segmentation={"n_segments": 4}, corrector=_ptt(piston_only=True))


def test_piston_only_shapes(piston_only_sim):
    petals = piston_only_sim._c.correctors[0]
    assert petals.piston_only
    assert petals.n_segments == 4
    assert petals.n_actuators == 4
    assert petals.actuators.shape == (4,)
    petals.set_actuators(np.array([[1.0], [2.0], [3.0], [4.0]]))  # (n, 1) accepted
    np.testing.assert_allclose(petals.actuators, [1.0, 2.0, 3.0, 4.0])
    petals.flatten()
    with pytest.raises(ValueError, match="piston_only"):
        petals.set_actuators(np.zeros((4, 3)))


def test_piston_only_matches_full_ptt_with_zero_tip_tilt(petal_sim, piston_only_sim):
    p = np.array([0.7, -1.2, 0.3, 0.9])
    full = np.column_stack([p, np.zeros(4), np.zeros(4)])
    img_full = _psf(petal_sim, full)
    img_po = _psf(piston_only_sim, p)
    np.testing.assert_array_equal(img_po, img_full)
    np.testing.assert_allclose(piston_only_sim.sample({"petals": p})["actuations"]["petals"], p)


def test_piston_only_fit_is_the_piston_column(petal_sim, piston_only_sim):
    petals = petal_sim._c.correctors[0]
    rng = np.random.default_rng(7)
    petals.set_actuators(rng.normal(size=(4, 3)))
    opd = 2.0 * np.asarray(petals._sm.surface, dtype=float)
    petals.flatten()
    full_fit = np.asarray(petals.fit_surface(opd))
    po_fit = np.asarray(piston_only_sim._c.correctors[0].fit_surface(opd))
    assert po_fit.shape == (4,)
    np.testing.assert_allclose(po_fit, full_fit[:, 0], rtol=1e-12, atol=1e-15)
