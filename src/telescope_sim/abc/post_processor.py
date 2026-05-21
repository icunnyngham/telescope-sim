"""PostProcessor ABC — ordered image transforms applied per output.

Each output declares a list of post-processors that run in order. A processor
takes the current image plus a :class:`PipelineContext` (which exposes
reference PSFs, peak intensities, and any per-sample state needed for
normalization) and returns the next image. Processors may change the array's
shape (e.g. ``fft_channels`` appends channels; ``channels_first`` transposes).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass
class PipelineContext:
    """Per-sample state available to post-processors.

    Populated by the pipeline before running the post-processing list for each
    output. Includes reference values (peak intensity, normalized reference PSF)
    used by normalization steps, plus any per-sample overrides the caller
    supplied via ``sim.sample(output_overrides={...})``.
    """

    output_name: str
    focal_plane_name: str
    reference_peak_intensity: float | None = None
    reference_psf_sum: float | None = None
    overrides: dict[str, Any] = field(default_factory=dict)
    extras: dict[str, Any] = field(default_factory=dict)


class LoaderBindable:
    """Marker mixin for post-processors that need loader-injected dependencies.

    The loader walks each output's post-processing list and calls
    ``_bind_loader_dependencies(aperture_result, focal_planes, focal_plane_names)``
    on any processor exposing the hook. Used by ``noisy_detector`` (needs the
    focal grid + aperture area) and ``convolve_image`` (needs the reference
    PSF sum). Processors without runtime dependencies simply don't expose
    the hook and the loader skips them.
    """

    def _bind_loader_dependencies(
        self,
        *,
        aperture_result: Any,
        focal_planes: dict[str, Any],
        focal_plane_names: list[str],
    ) -> None:
        raise NotImplementedError


class PostProcessor(ABC):
    """ABC for image-space transforms applied after wavefront extraction."""

    name: str

    @abstractmethod
    def __call__(
        self,
        image: NDArray[np.floating],
        context: PipelineContext,
    ) -> NDArray[np.floating]:
        """Transform an image, optionally changing its shape."""
        ...


__all__ = ["PostProcessor", "PipelineContext", "LoaderBindable"]
