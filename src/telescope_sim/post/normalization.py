"""Normalization post-processors used by the canonical-family fixtures."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from telescope_sim.abc import PipelineContext, PostProcessor
from telescope_sim.registry import register


@register("post_processor", "max_intensity_norm")
class MaxIntensityNorm(PostProcessor):
    """Divide each channel by the corresponding reference PSF peak intensity.

    Mirrors the canonical `max_inten_norm` flag — the per-filter peak of the
    at-rest reference PSF is precomputed by the pipeline; this transform
    divides each channel by that scalar. Channels here = last axis of an
    ``(H, W, C)`` image stack, where C corresponds to the focal planes
    listed in the parent IntensityOutputTap.
    """

    name = "max_intensity_norm"

    def __call__(
        self, image: NDArray[np.floating], context: PipelineContext
    ) -> NDArray[np.floating]:
        peaks = context.extras.get("reference_peak_intensities")
        if peaks is None:
            raise RuntimeError(
                "max_intensity_norm requires context.extras['reference_peak_intensities']"
            )
        peaks = np.asarray(peaks, dtype=np.float64)
        # image is (H, W, C); peaks is (C,)
        if image.shape[-1] != peaks.shape[0]:
            raise ValueError(
                f"channel count {image.shape[-1]} != n_peaks {peaks.shape[0]}"
            )
        return image / peaks[None, None, :]


@register("post_processor", "max_image_norm")
class MaxImageNorm(PostProcessor):
    """Divide each channel by its own max value.

    Mirrors the coronagraph-variant `max_im_norm` flag — useful when the
    per-image dynamic range varies a lot (e.g. coronagraph variants where
    the reference PSF peak isn't a meaningful normalization target).
    """

    name = "max_image_norm"

    def __call__(
        self, image: NDArray[np.floating], context: PipelineContext
    ) -> NDArray[np.floating]:
        if image.ndim == 2:
            mx = image.max()
            return image / mx if mx > 0 else image
        # (H, W, C) — per-channel
        maxes = image.reshape(-1, image.shape[-1]).max(axis=0)
        safe = np.where(maxes > 0, maxes, 1.0)
        return image / safe[None, None, :]


@register("post_processor", "per_sample_norm")
class PerSampleNorm(PostProcessor):
    """Min-max normalize each channel to [0, 1]."""

    name = "per_sample_norm"

    def __call__(
        self, image: NDArray[np.floating], context: PipelineContext
    ) -> NDArray[np.floating]:
        if image.ndim == 2:
            mn, mx = image.min(), image.max()
            return (image - mn) / (mx - mn) if mx > mn else image - mn
        # Per-channel along last axis
        flat = image.reshape(-1, image.shape[-1])
        mins = flat.min(axis=0)
        maxes = flat.max(axis=0)
        denom = np.where(maxes > mins, maxes - mins, 1.0)
        return (image - mins[None, None, :]) / denom[None, None, :]


@register("post_processor", "channels_first")
class ChannelsFirst(PostProcessor):
    """Transpose ``(H, W, C)`` → ``(C, H, W)`` for PyTorch-style consumers."""

    name = "channels_first"

    def __call__(
        self, image: NDArray[np.floating], context: PipelineContext
    ) -> NDArray[np.floating]:
        if image.ndim == 3:
            return np.transpose(image, (2, 0, 1))
        return image


__all__ = [
    "MaxIntensityNorm",
    "MaxImageNorm",
    "PerSampleNorm",
    "ChannelsFirst",
]
