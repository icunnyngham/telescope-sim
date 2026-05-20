"""Parity tests for ``IntensityOutputTap`` against the canonical legacy stacking.

Legacy reference (TelescopeSim/.../multi_aperture_psf.py:520-523):

    Xs += [ out_samp[..., None] ]
    ...
    Xs = np.concatenate(Xs, axis=2)

Each filter's PSF is given a trailing channel axis and the filters are
concatenated along it, producing the canonical ``(H, W, n_filters)`` shape.

v2's ``IntensityOutputTap`` does this in one step via ``np.stack(psfs,
axis=-1)`` over the per-focal-plane summed intensities. These tests
verify the stacking convention matches legacy for the single- and
multi-focal-plane cases.
"""

from __future__ import annotations

import numpy as np
import pytest

import telescope_sim.outputs.intensity  # noqa: F401 (registers)


def _fp_result(arr):
    from telescope_sim.focal_planes.physical import FocalPlaneResult

    return FocalPlaneResult(intensity=arr, wavefronts=[])


def test_intensity_tap_single_focal_plane_shape():
    """One focal plane → shape (H, W, 1)."""
    from telescope_sim.outputs.intensity import IntensityOutputTap

    psf = np.arange(32 * 32).reshape(32, 32).astype(np.float64)
    tap = IntensityOutputTap(focal_plane_names=["f1"])
    out = tap.extract({"f1": _fp_result(psf)})
    assert out.shape == (32, 32, 1)
    np.testing.assert_array_equal(out[:, :, 0], psf)


def test_intensity_tap_multi_focal_plane_stacks_channels_last():
    """Multiple focal planes → channels-last stack matching legacy concat-axis-2."""
    from telescope_sim.outputs.intensity import IntensityOutputTap

    psf1 = np.full((16, 16), 1.0)
    psf2 = np.full((16, 16), 2.0)
    psf3 = np.full((16, 16), 3.0)
    tap = IntensityOutputTap(focal_plane_names=["a", "b", "c"])
    out = tap.extract({"a": _fp_result(psf1), "b": _fp_result(psf2), "c": _fp_result(psf3)})
    assert out.shape == (16, 16, 3)
    # Channel order matches the focal_plane_names list (NOT the dict iteration order)
    np.testing.assert_array_equal(out[:, :, 0], psf1)
    np.testing.assert_array_equal(out[:, :, 1], psf2)
    np.testing.assert_array_equal(out[:, :, 2], psf3)


def test_intensity_tap_channel_order_follows_focal_plane_names():
    """Reordering focal_plane_names reorders channels — dict order is irrelevant."""
    from telescope_sim.outputs.intensity import IntensityOutputTap

    psf_a = np.full((8, 8), 1.0)
    psf_b = np.full((8, 8), 2.0)
    fp_results = {"b": _fp_result(psf_b), "a": _fp_result(psf_a)}

    tap_ab = IntensityOutputTap(focal_plane_names=["a", "b"])
    tap_ba = IntensityOutputTap(focal_plane_names=["b", "a"])

    out_ab = tap_ab.extract(fp_results)
    out_ba = tap_ba.extract(fp_results)

    np.testing.assert_array_equal(out_ab[:, :, 0], psf_a)
    np.testing.assert_array_equal(out_ab[:, :, 1], psf_b)
    np.testing.assert_array_equal(out_ba[:, :, 0], psf_b)
    np.testing.assert_array_equal(out_ba[:, :, 1], psf_a)


def test_intensity_tap_uses_focal_plane_result_intensity_not_wavefronts():
    """Tap reads ``.intensity`` (summed across wavelengths), not per-WF list.

    Mirrors the legacy `out_samp = psf` path where the wavelength sum was
    already accumulated in ``_psf()`` before reaching the output stage.
    """
    from telescope_sim.focal_planes.physical import FocalPlaneResult
    from telescope_sim.outputs.intensity import IntensityOutputTap

    psf = np.full((8, 8), 5.0)
    # Even with NO wavefronts in the FocalPlaneResult, extraction works
    # because we only read .intensity.
    fp = FocalPlaneResult(intensity=psf, wavefronts=[])
    tap = IntensityOutputTap(focal_plane_names=["f1"])
    out = tap.extract({"f1": fp})
    np.testing.assert_array_equal(out[:, :, 0], psf)


def test_intensity_tap_empty_focal_plane_names_raises():
    from telescope_sim.outputs.intensity import IntensityOutputTap

    with pytest.raises(ValueError, match="at least one"):
        IntensityOutputTap(focal_plane_names=[])


def test_intensity_tap_missing_focal_plane_raises():
    from telescope_sim.outputs.intensity import IntensityOutputTap

    tap = IntensityOutputTap(focal_plane_names=["a", "b"])
    with pytest.raises(KeyError, match=r"\[.*'b'.*\]"):
        tap.extract({"a": _fp_result(np.zeros((4, 4)))})


def test_intensity_tap_wrong_input_type_raises():
    from telescope_sim.outputs.intensity import IntensityOutputTap

    tap = IntensityOutputTap(focal_plane_names=["f1"])
    with pytest.raises(TypeError, match="dict"):
        tap.extract(np.zeros((4, 4)))


def test_intensity_tap_source_string_lists_focal_planes():
    from telescope_sim.outputs.intensity import IntensityOutputTap

    tap = IntensityOutputTap(focal_plane_names=["x", "y", "z"])
    assert tap.source == "focal:x,y,z"
    # And focal_plane_names attribute mirrors the input
    assert tap.focal_plane_names == ["x", "y", "z"]
