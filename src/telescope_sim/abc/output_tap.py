"""OutputTap ABC — extracts a numpy array from a named intermediate wavefront.

Output taps reference a named wavefront in the pipeline (``pupil_post_coro``,
focal-plane names, ``pupil_pre_<corrector>``, etc.) and produce an array that
becomes one of the entries in the returned ``images`` dict.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from numpy.typing import NDArray


class OutputTap(ABC):
    """ABC for output extractors on named intermediate wavefronts."""

    name: str
    source: str  # name of the intermediate wavefront to consume

    @abstractmethod
    def extract(self, wf: Any, *, overrides: dict[str, Any] | None = None) -> NDArray[np.floating]:
        """Return a numpy array representing this tap's view of the wavefront.

        Parameters
        ----------
        wf
            The tap's input — typically the ``fp_results`` dict mapping focal-
            plane name to ``FocalPlaneResult``.
        overrides
            Per-sample tap-config overrides from ``sim.sample(output_overrides=...)``.
            Most taps ignore this; taps that have per-sample state (e.g. photon
            flux for a noisy detector) read named fields here.
        """
        ...


__all__ = ["OutputTap"]
