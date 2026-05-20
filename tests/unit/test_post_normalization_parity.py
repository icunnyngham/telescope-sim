"""Parity tests for the four post-processors in ``post/normalization.py``.

Covers audits 11-14 of the v2.0 → legacy parity sweep. Each post-processor
is a thin wrapper around a one-liner from the legacy ``extra_processing``
block — the audit value is structural: did the v2 implementation honor
the right axis, the right scope, and the right per-channel vs global
semantics?

Legacy references:

- **MaxIntensityNorm** (`multi_aperture_psf.py:486-487` and similar):
    if self.extra_processing['max_inten_norm']:
        psf /= lam_setup['peak_int']
  → divide each filter's PSF by *its own reference peak intensity*. v2
  applies this per-channel using ``context.extras['reference_peak_intensities']``.

- **PerSampleNorm** (`multi_aperture_psf.py:516-518`):
    samp_min, samp_max = out_samp.min(), out_samp.max()
    out_samp = (out_samp - samp_min) / (samp_max - samp_min)
  → per-filter (single channel) min-max to [0, 1]. v2 supports the same
  semantic in the (H, W) path; for stacked (H, W, C) it normalizes
  *per-channel*, which matches legacy's "per-filter then concat" behavior
  for single-channel-per-filter inputs.

- **MaxImageNorm** (`variants/coro__coro_mas_psf.py:386-387`):
    if self.extra_processing['max_im_norm']:
        out_samp /= out_samp.max()
  → divide by the current image's max (NOT the reference peak — useful
  for coronagraphs where the reference peak is suppressed).

- **ChannelsFirst** (canonical 2024-09 addition): transpose ``(H, W, C)`` →
  ``(C, H, W)`` for PyTorch. The pinning targets are: the axis order, the
  2D passthrough, and the (C-first) shape of the result.
"""

from __future__ import annotations

import numpy as np
import pytest

from telescope_sim.abc import PipelineContext


def _ctx(**extras) -> PipelineContext:
    return PipelineContext(
        output_name="psf",
        focal_plane_name="filter1",
        reference_peak_intensity=extras.get("reference_peak_intensity"),
        reference_psf_sum=extras.get("reference_psf_sum"),
        extras=extras,
    )


# --- MaxIntensityNorm ----------------------------------------------------


def test_max_intensity_norm_divides_each_channel_by_its_reference_peak():
    """Legacy: psf /= lam_setup['peak_int']. v2 applies per-channel."""
    from telescope_sim.post.normalization import MaxIntensityNorm

    img = np.ones((8, 8, 3)) * np.array([2.0, 4.0, 8.0])  # broadcast → (8,8,3)
    ctx = _ctx(reference_peak_intensities=np.array([2.0, 4.0, 8.0]))
    out = MaxIntensityNorm()(img, ctx)
    # Each channel now == 1.0
    np.testing.assert_allclose(out, np.ones_like(img))


def test_max_intensity_norm_requires_extras_peak_array():
    from telescope_sim.post.normalization import MaxIntensityNorm

    img = np.ones((4, 4, 1))
    with pytest.raises(RuntimeError, match="reference_peak_intensities"):
        MaxIntensityNorm()(img, _ctx())


def test_max_intensity_norm_channel_count_mismatch_raises():
    from telescope_sim.post.normalization import MaxIntensityNorm

    img = np.ones((4, 4, 3))
    ctx = _ctx(reference_peak_intensities=np.array([1.0, 2.0]))
    with pytest.raises(ValueError, match="channel count"):
        MaxIntensityNorm()(img, ctx)


# --- MaxImageNorm --------------------------------------------------------


def test_max_image_norm_2d_matches_legacy_global_divide():
    """Legacy coro: `out_samp /= out_samp.max()` — global max over the 2D PSF."""
    from telescope_sim.post.normalization import MaxImageNorm

    img = np.array([[1.0, 2.0], [3.0, 4.0]])
    out = MaxImageNorm()(img, _ctx())
    np.testing.assert_allclose(out, img / 4.0)


def test_max_image_norm_3d_normalizes_per_channel():
    """For stacked (H, W, C) input, each channel is normalized by its OWN max.

    This is the natural per-filter generalization of the legacy `out_samp.max()`
    when filters are stacked along the channel axis BEFORE this step.
    """
    from telescope_sim.post.normalization import MaxImageNorm

    ch0 = np.array([[1.0, 2.0], [3.0, 4.0]])
    ch1 = np.array([[10.0, 20.0], [30.0, 40.0]])
    img = np.stack([ch0, ch1], axis=-1)
    out = MaxImageNorm()(img, _ctx())
    np.testing.assert_allclose(out[..., 0], ch0 / 4.0)
    np.testing.assert_allclose(out[..., 1], ch1 / 40.0)


def test_max_image_norm_zero_max_returns_input_unchanged():
    """Guards: an all-zero channel divides safely (returns the input untouched)."""
    from telescope_sim.post.normalization import MaxImageNorm

    img = np.zeros((4, 4))
    out = MaxImageNorm()(img, _ctx())
    np.testing.assert_array_equal(out, img)

    img_3d = np.zeros((4, 4, 2))
    out_3d = MaxImageNorm()(img_3d, _ctx())
    np.testing.assert_array_equal(out_3d, img_3d)


# --- PerSampleNorm -------------------------------------------------------


def test_per_sample_norm_2d_matches_legacy_minmax():
    """Legacy: (out_samp - out_samp.min()) / (out_samp.max() - out_samp.min())."""
    from telescope_sim.post.normalization import PerSampleNorm

    img = np.array([[1.0, 2.0], [3.0, 5.0]])
    out = PerSampleNorm()(img, _ctx())
    expected = (img - 1.0) / (5.0 - 1.0)
    np.testing.assert_allclose(out, expected)


def test_per_sample_norm_3d_normalizes_per_channel():
    """Per-channel min-max in the (H, W, C) path — matches legacy per-filter behavior."""
    from telescope_sim.post.normalization import PerSampleNorm

    ch0 = np.array([[1.0, 2.0], [3.0, 5.0]])
    ch1 = np.array([[10.0, 20.0], [30.0, 50.0]])
    img = np.stack([ch0, ch1], axis=-1)
    out = PerSampleNorm()(img, _ctx())
    np.testing.assert_allclose(out[..., 0], (ch0 - 1.0) / 4.0)
    np.testing.assert_allclose(out[..., 1], (ch1 - 10.0) / 40.0)


def test_per_sample_norm_constant_channel_safely_zeros():
    """A constant channel (max == min) avoids division-by-zero."""
    from telescope_sim.post.normalization import PerSampleNorm

    img = np.full((4, 4), 5.0)
    out = PerSampleNorm()(img, _ctx())
    # Returns input - min (= zeros) when range is zero
    np.testing.assert_array_equal(out, np.zeros_like(img))


# --- ChannelsFirst -------------------------------------------------------


def test_channels_first_transposes_3d_input():
    """(H, W, C) → (C, H, W) — the canonical 2024-09 PyTorch convention."""
    from telescope_sim.post.normalization import ChannelsFirst

    img = np.arange(2 * 3 * 4).reshape(2, 3, 4)  # (H=2, W=3, C=4)
    out = ChannelsFirst()(img, _ctx())
    assert out.shape == (4, 2, 3)  # (C, H, W)
    # And the values are correctly permuted
    for c in range(4):
        np.testing.assert_array_equal(out[c], img[..., c])


def test_channels_first_passes_through_2d():
    """A 2D image isn't transposed (no channel axis to move)."""
    from telescope_sim.post.normalization import ChannelsFirst

    img = np.arange(12).reshape(3, 4)
    out = ChannelsFirst()(img, _ctx())
    assert out.shape == (3, 4)
    np.testing.assert_array_equal(out, img)


def test_channels_first_inverse_with_transpose_recovers_input():
    """Round-trip sanity: ChannelsFirst then transpose(1,2,0) yields the input."""
    from telescope_sim.post.normalization import ChannelsFirst

    img = np.arange(2 * 3 * 4).reshape(2, 3, 4)
    out = ChannelsFirst()(img, _ctx())
    inv = np.transpose(out, (1, 2, 0))
    np.testing.assert_array_equal(inv, img)
