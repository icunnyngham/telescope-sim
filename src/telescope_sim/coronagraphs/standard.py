"""Standard coronagraph implementations: identity, vortex, vector_vortex.

A coronagraph wraps an HCIPy coronagraph element and applies it after the
corrector chain but before propagation. The pipeline always generates the
reference PSF (for Strehl normalization) with the coronagraph bypassed.

Each implementation takes an optional ``lyot`` Aperture config that
specifies the downstream Lyot stop's transmission field.
"""

from __future__ import annotations

from typing import Any

import hcipy
from telescope_sim.abc import Aperture, Coronagraph
from telescope_sim.registry import lookup, register


def _resolve_lyot_field(lyot_cfg: Any, pupil_grid: Any) -> Any | None:
    """Build a Lyot-stop transmission field from an aperture sub-config."""
    if lyot_cfg is None:
        return None
    payload = dict(lyot_cfg) if not hasattr(lyot_cfg, "model_dump") else lyot_cfg.model_dump()
    type_name = payload.pop("type")
    cls = lookup("aperture", type_name)
    ap: Aperture = cls(**payload)
    result = ap.build(pupil_grid)
    return result.field


@register("coronagraph", "identity")
class IdentityCoronagraph(Coronagraph):
    """Passthrough — used as the "no coronagraph" placeholder."""

    name = "identity"

    def __init__(self, **_: Any) -> None:
        pass

    def apply(self, wf: Any) -> Any:
        return wf

    def _bind_pupil_grid(self, pupil_grid: Any) -> None:
        pass


@register("coronagraph", "vortex")
class VortexCoronagraphImpl(Coronagraph):
    """HCIPy VortexCoronagraph at a configurable charge.

    The Lyot stop is built from an ``Aperture`` sub-config; the
    coronagraph is constructed lazily once the pipeline binds a pupil
    grid via :meth:`_bind_pupil_grid`.
    """

    name = "vortex"

    def __init__(self, charge: int = 2, lyot: dict | None = None) -> None:
        self.charge = int(charge)
        self._lyot_cfg = lyot
        self._coro: Any | None = None

    def _bind_pupil_grid(self, pupil_grid: Any) -> None:
        lyot_field = _resolve_lyot_field(self._lyot_cfg, pupil_grid)
        self._coro = hcipy.VortexCoronagraph(pupil_grid, charge=self.charge, lyot_stop=lyot_field)

    def apply(self, wf: Any) -> Any:
        if self._coro is None:
            raise RuntimeError("VortexCoronagraph must be bound via _bind_pupil_grid()")
        return self._coro(wf)


@register("coronagraph", "vector_vortex")
class VectorVortexCoronagraphImpl(Coronagraph):
    """HCIPy VectorVortexCoronagraph at a configurable charge."""

    name = "vector_vortex"

    def __init__(self, charge: int = 4, lyot: dict | None = None) -> None:
        self.charge = int(charge)
        self._lyot_cfg = lyot
        self._coro: Any | None = None

    def _bind_pupil_grid(self, pupil_grid: Any) -> None:
        lyot_field = _resolve_lyot_field(self._lyot_cfg, pupil_grid)
        # VectorVortexCoronagraph in HCIPy takes lyot_stop as a positional/kwarg
        self._coro = hcipy.VectorVortexCoronagraph(charge=self.charge, lyot_stop=lyot_field)

    def apply(self, wf: Any) -> Any:
        if self._coro is None:
            raise RuntimeError("VectorVortexCoronagraph must be bound via _bind_pupil_grid()")
        return self._coro(wf)


__all__ = [
    "IdentityCoronagraph",
    "VortexCoronagraphImpl",
    "VectorVortexCoronagraphImpl",
]
