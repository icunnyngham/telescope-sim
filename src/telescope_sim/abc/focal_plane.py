"""FocalPlane ABC — owns a propagator, focal grid, wavelength sampling, optional detector.

Multiple named focal planes can fan out from the same ``pupil_post_coro`` wavefront,
each with its own propagator and grid. This supports per-filter PSFs and
mixed-physics setups (e.g. an angular focal plane for the science detector
alongside a physical focal plane for fiber coupling).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from numpy.typing import NDArray


class FocalPlane(ABC):
    """ABC for focal-plane setups (propagator + grid + optional detector)."""

    name: str

    @abstractmethod
    def propagate(self, wf: Any) -> Any:
        """Propagate a pupil-plane wavefront to this focal plane.

        Returns the focal-plane ``Wavefront`` (keeping phase info, so downstream
        taps like fiber coupling can use it).
        """
        ...

    def integrate(self, wf: Any, t_exp: float = 1.0) -> NDArray[np.floating]:
        """Integrate a focal-plane wavefront through the detector if present.

        Default impl: returns ``wf.intensity.shaped`` (no detector noise).
        Implementations with detectors override this to call the in-chain
        ``NoisyDetector``.
        """
        return np.asarray(wf.intensity.shaped)


__all__ = ["FocalPlane"]
