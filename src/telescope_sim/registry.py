"""Tiny dict-of-dicts registry for pluggable pipeline stages.

Usage::

    from telescope_sim.registry import register
    from telescope_sim.abc import Aperture

    @register("aperture", "segmented_circular")
    class SegmentedCircularAperture(Aperture):
        ...

    cls = registry["aperture"]["segmented_circular"]

Users may register their own implementations in their own packages without
modifying telescope-sim.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T", bound=type)


_KINDS = (
    "aperture",
    "corrector",
    "coronagraph",
    "focal_plane",
    "output_tap",
    "post_processor",
)


registry: dict[str, dict[str, type]] = {kind: {} for kind in _KINDS}


def register(kind: str, name: str) -> Callable[[T], T]:
    """Decorator: register a class as a named implementation of a pipeline-stage kind."""
    if kind not in registry:
        raise ValueError(f"Unknown registry kind {kind!r}; valid kinds: {sorted(registry)}")

    def _decorator(cls: T) -> T:
        existing = registry[kind].get(name)
        if existing is not None and existing is not cls:
            raise ValueError(
                f"{kind}/{name} already registered to {existing!r}; cannot reassign to {cls!r}"
            )
        registry[kind][name] = cls
        return cls

    return _decorator


def lookup(kind: str, name: str) -> type:
    """Resolve a registered implementation, raising KeyError with a helpful message."""
    if kind not in registry:
        raise KeyError(f"Unknown registry kind {kind!r}")
    if name not in registry[kind]:
        available = sorted(registry[kind])
        raise KeyError(f"{kind}/{name} is not registered. Available {kind} types: {available}")
    return registry[kind][name]


def available(kind: str) -> list[str]:
    """List registered implementation names for a given kind."""
    return sorted(registry.get(kind, {}))


__all__ = ["register", "registry", "lookup", "available"]
