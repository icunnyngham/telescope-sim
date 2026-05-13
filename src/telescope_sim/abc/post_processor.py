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
    used by normalization steps.
    """

    output_name: str
    focal_plane_name: str
    reference_peak_intensity: float | None = None
    reference_psf_sum: float | None = None
    extras: dict[str, Any] = field(default_factory=dict)


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


__all__ = ["PostProcessor", "PipelineContext"]
