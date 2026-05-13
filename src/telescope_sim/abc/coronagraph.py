"""Coronagraph ABC — optional pupil-plane stage between correctors and focal plane.

The pipeline always includes a coronagraph slot; the ``identity`` implementation
is a no-op that keeps the chain uniform when no coronagraph is configured. The
reference PSF (for Strehl normalization) is always generated with the identity
coronagraph regardless of what's installed.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class Coronagraph(ABC):
    """ABC for pupil-plane coronagraph elements."""

    name: str

    @abstractmethod
    def apply(self, wf: Any) -> Any:
        """Apply the coronagraph to a pupil-plane wavefront, returning the modified wavefront."""
        ...


__all__ = ["Coronagraph"]
