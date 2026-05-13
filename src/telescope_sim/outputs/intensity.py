"""IntensityOutputTap — channels-last stack of per-focal-plane PSF intensities.

Mirrors the canonical-family behaviour where one OutputTap collects the
PSF intensities from one or more named focal planes and stacks them along
the last axis (shape ``(H, W, n_focal_planes)``).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from telescope_sim.abc import OutputTap
from telescope_sim.registry import register


@register("output_tap", "intensity")
class IntensityOutputTap(OutputTap):
    """Stack per-focal-plane PSF intensities along the channel (last) axis.

    Parameters
    ----------
    focal_plane_names
        One or more named focal planes whose summed intensities feed this tap.
    name
        Output name (used as the key in ``sample()``'s returned ``images`` dict).
    """

    def __init__(
        self,
        focal_plane_names: list[str],
        *,
        name: str = "psf",
    ) -> None:
        if not focal_plane_names:
            raise ValueError("IntensityOutputTap needs at least one focal_plane name")
        self.name = name
        self.focal_plane_names = list(focal_plane_names)
        # The ABC's `source` field is informational here; we expose the
        # composite source string for diagnostics.
        self.source = "focal:" + ",".join(self.focal_plane_names)

    def extract(self, wf: Any) -> NDArray[np.floating]:
        """Extract from a dict mapping focal_plane name → 2D PSF intensity.

        The pipeline orchestrator builds this dict by running each focal
        plane's chain-propagation once per ``sample()``.
        """
        if not isinstance(wf, dict):
            raise TypeError(
                "IntensityOutputTap.extract expects a dict {focal_plane_name: psf}; "
                f"got {type(wf).__name__}"
            )
        missing = [n for n in self.focal_plane_names if n not in wf]
        if missing:
            raise KeyError(
                f"focal planes {missing} not in available outputs {list(wf)}"
            )
        # Stack channels-last to match the canonical layout
        psfs = [np.asarray(wf[n], dtype=np.float64) for n in self.focal_plane_names]
        return np.stack(psfs, axis=-1)


__all__ = ["IntensityOutputTap"]
