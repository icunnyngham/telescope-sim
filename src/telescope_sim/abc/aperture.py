"""Aperture ABC — constructs the static pupil-plane transmission mask.

Implementations are registered with ``@register("aperture", "<name>")`` and
return an :class:`ApertureResult` from :meth:`Aperture.build`. The pipeline
calls ``build()`` once at construction time.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass
class ApertureResult:
    """The outputs of an :class:`Aperture` build step.

    Attributes
    ----------
    field
        HCIPy ``Field`` representing the pupil-plane transmission (real-valued).
    area
        Effective collecting area in pupil-grid units squared (used for photon
        flux scaling in noise calculations).
    segments
        Optional ``SegmentedAperture`` (HCIPy) for segmented mirrors. ``None``
        for monolithic apertures.
    segment_coords
        Optional ``(n_segments, 2)`` array of segment center coordinates.
        Used by per-segment PTT measurement (``_measure_atmos_ptt``-style fits).
    """

    field: Any  # hcipy.Field
    area: float
    segments: Any | None = None
    segment_coords: NDArray[np.floating] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class Aperture(ABC):
    """ABC for pupil-plane aperture constructors.

    Subclasses are typically registered via :func:`telescope_sim.registry.register`::

        @register("aperture", "segmented_circular")
        class SegmentedCircularAperture(Aperture):
            def build(self, pupil_grid):
                ...
    """

    @abstractmethod
    def build(self, pupil_grid: Any) -> ApertureResult:
        """Construct the aperture on the given pupil grid.

        Parameters
        ----------
        pupil_grid
            HCIPy pupil grid (typically ``make_pupil_grid(resolution, extent)``).

        Returns
        -------
        ApertureResult
            The transmission field, collecting area, and optional segmentation.
        """
        ...


__all__ = ["Aperture", "ApertureResult"]
