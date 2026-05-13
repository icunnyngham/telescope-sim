"""Atmosphere — special chain element wrapping HCIPy atmospheric layers.

Atmosphere is conceptually an ``impose`` corrector at the front of the chain,
but it owns a time-evolution method (``evolve_until``) that the caller drives.
This preserves the RL-friendly "environment controls time" separation while
letting the YAML config declare the atmosphere's parameters statically.
"""

from __future__ import annotations

from typing import Any


class Atmosphere:
    """Wraps an HCIPy ``InfiniteAtmosphericLayer`` or ``MultiLayerAtmosphere``."""

    def __init__(self, config: Any) -> None:
        raise NotImplementedError("Atmosphere construction is not yet implemented.")

    def evolve_until(self, t: float) -> None:
        """Step the atmosphere to absolute time ``t`` (seconds)."""
        raise NotImplementedError("Atmosphere time evolution is not yet implemented.")


__all__ = ["Atmosphere"]
