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

# Per-backend overlay: implementations registered for a specific compute
# backend (e.g. "jax") shadow the backend-agnostic table above for that
# backend only. Keyed [backend][kind][name].
backend_registry: dict[str, dict[str, dict[str, type]]] = {}


def register(kind: str, name: str, *, backend: str | None = None) -> Callable[[T], T]:
    """Decorator: register a class as a named implementation of a pipeline-stage kind.

    With ``backend=None`` (the default) the implementation is backend-agnostic
    and used by every backend that has no specific override. Passing a backend
    name (e.g. ``backend="jax"``) registers a backend-specific implementation
    that shadows the agnostic one when the pipeline is built for that backend.
    """
    if kind not in registry:
        raise ValueError(f"Unknown registry kind {kind!r}; valid kinds: {sorted(registry)}")

    table = (
        registry[kind]
        if backend is None
        else backend_registry.setdefault(backend, {kind: {} for kind in _KINDS})[kind]
    )

    def _decorator(cls: T) -> T:
        existing = table.get(name)
        if existing is not None and existing is not cls:
            raise ValueError(
                f"{kind}/{name} already registered to {existing!r}; cannot reassign to {cls!r}"
            )
        table[name] = cls
        return cls

    return _decorator


def lookup(kind: str, name: str, *, backend: str | None = None) -> type:
    """Resolve a registered implementation, raising KeyError with a helpful message.

    Backend-specific registrations (see :func:`register`) take precedence for
    their backend; otherwise the backend-agnostic table is consulted.
    """
    if kind not in registry:
        raise KeyError(f"Unknown registry kind {kind!r}")
    if backend is not None:
        cls = backend_registry.get(backend, {}).get(kind, {}).get(name)
        if cls is not None:
            return cls
    if name not in registry[kind]:
        available = sorted(registry[kind])
        raise KeyError(f"{kind}/{name} is not registered. Available {kind} types: {available}")
    return registry[kind][name]


def available(kind: str) -> list[str]:
    """List registered implementation names for a given kind."""
    return sorted(registry.get(kind, {}))


__all__ = ["register", "registry", "backend_registry", "lookup", "available"]
