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
    def extract(self, wf: Any) -> NDArray[np.floating]:
        """Return a numpy array representing this tap's view of the wavefront."""
        ...


__all__ = ["OutputTap"]
