"""Convolve-with-image post-processor.

Closes the ``convolve_im`` feature gap from the v2.0.0a6 audit.

Legacy behavior (multi_aperture_psf.py:489-491):

    if convolve_im is not None:
        out_samp = fftconvolve(convolve_im, psf / lam_setup['ref_psf_sum'],
                               mode='same')
    else:
        out_samp = psf

The PSF is normalized by the at-rest reference's sum (so the kernel
integrates to ~1, preserving the input image's flux scale), then
``scipy.signal.fftconvolve(image, kernel, mode='same')`` produces an
output with the same shape as the input image.

v2 ships this as a post-processor (rather than a tap option) so it
composes cleanly with the new ``noisy_detector`` post-processor:
``intensity → convolve → noise`` is the physical chain. Per-sample image
override via the ``PipelineContext.overrides`` channel.

Single-focal-plane only — the kernel normalization needs a single
``reference_psf_sum``. Multi-channel outputs that want convolution must
split into one output per filter.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.signal import fftconvolve

from telescope_sim.abc import LoaderBindable, PipelineContext, PostProcessor
from telescope_sim.registry import register


@register("post_processor", "convolve_image")
class ConvolveImagePostProcessor(PostProcessor, LoaderBindable):
    """Convolve a PSF image with a caller-supplied scene.

    Parameters
    ----------
    image
        Default scene to convolve with. 2D ``ndarray``. ``None`` means no
        default — the caller must supply ``convolve_image`` via
        ``output_overrides`` per sample.
    """

    name = "convolve_image"

    def __init__(self, *, image: Any = None) -> None:
        self._default_image: NDArray[np.floating] | None = (
            None if image is None else np.asarray(image, dtype=np.float64)
        )
        # Populated by the loader via _bind_loader_dependencies
        self._reference_psf_sum: float | None = None

    def _bind_loader_dependencies(
        self,
        *,
        aperture_result: Any,  # noqa: ARG002
        focal_planes: dict[str, Any],
        focal_plane_names: list[str],
    ) -> None:
        if len(focal_plane_names) != 1:
            raise ValueError(
                "convolve_image post-processor takes exactly one focal_plane; "
                f"the output references {focal_plane_names!r}. For multi-filter "
                "convolution, declare one output (with its own convolve_image) "
                "per filter."
            )
        fp = focal_planes[focal_plane_names[0]]
        if fp.reference_psf_sum is None or fp.reference_psf_sum <= 0:
            raise RuntimeError(
                f"convolve_image needs a positive reference_psf_sum on focal "
                f"plane {focal_plane_names[0]!r}; got {fp.reference_psf_sum!r}. "
                "Did compute_reference_psf() run during sim build?"
            )
        self._reference_psf_sum = float(fp.reference_psf_sum)

    def __call__(
        self,
        image: NDArray[np.floating],
        context: PipelineContext,
    ) -> NDArray[np.floating]:
        if self._reference_psf_sum is None:
            raise RuntimeError(
                "ConvolveImagePostProcessor must be bound via "
                "_bind_loader_dependencies() (normally done by the YAML "
                "loader) before being called."
            )

        # Per-sample override beats YAML default
        scene = context.overrides.get("convolve_image", self._default_image)
        if scene is None:
            # No image configured at all → passthrough (lets users keep the
            # convolve processor in the YAML and toggle it on per-sample).
            return image

        scene = np.asarray(scene, dtype=np.float64)
        if scene.ndim != 2:
            raise ValueError(f"convolve_image expects a 2D scene; got shape {scene.shape}")

        # Input shape: (H_psf, W_psf, 1). Strip channel axis, normalize PSF
        # kernel by the at-rest reference sum, convolve, restore channel.
        if image.ndim != 3 or image.shape[-1] != 1:
            raise ValueError(
                "ConvolveImagePostProcessor expects a single-channel image "
                f"of shape (H, W, 1); got {image.shape}."
            )
        psf = np.asarray(image[..., 0], dtype=np.float64)
        kernel = psf / self._reference_psf_sum
        convolved = fftconvolve(scene, kernel, mode="same")
        return convolved[..., None]


__all__ = ["ConvolveImagePostProcessor"]
