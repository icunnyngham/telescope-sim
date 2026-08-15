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

    Supported on both backends: the jax backend replays the exact
    multi-scale masks this hcipy object precomputes (see
    :meth:`_jax_multi_scale_sources`) through matched λ=1 transforms —
    the vortex phase is scale-invariant, so hcipy evaluates the whole
    train at unit wavelength and so does the jax port.
    """

    name = "vortex"
    supported_backends = frozenset({"hcipy", "jax"})

    def __init__(self, charge: int = 2, lyot: dict | None = None) -> None:
        self.charge = int(charge)
        self._lyot_cfg = lyot
        self._coro: Any | None = None
        self.lyot_field: Any | None = None
        self._pupil_grid: Any | None = None

    def _bind_pupil_grid(self, pupil_grid: Any) -> None:
        self._pupil_grid = pupil_grid
        self.lyot_field = _resolve_lyot_field(self._lyot_cfg, pupil_grid)
        self._coro = hcipy.VortexCoronagraph(
            pupil_grid, charge=self.charge, lyot_stop=self.lyot_field
        )

    def apply(self, wf: Any) -> Any:
        if self._coro is None:
            raise RuntimeError("VortexCoronagraph must be bound via _bind_pupil_grid()")
        return self._coro(wf)

    def _jax_multi_scale_sources(self) -> list[tuple[float, Any]]:
        """(weight, hcipy MultiScaleCoronagraph) channels for the jax train.

        The scalar vortex is a single unit-weight channel. The harvested
        object's ``focal_masks``/``props`` carry the windowed,
        correction-subtracted per-level masks; the Lyot stop is applied
        separately by the jax builder (``lyot_field``), matching where
        hcipy applies it.
        """
        if self._coro is None:
            raise RuntimeError("VortexCoronagraph must be bound via _bind_pupil_grid()")
        return [(1.0, self._coro)]


@register("coronagraph", "vector_vortex")
class VectorVortexCoronagraphImpl(Coronagraph):
    """HCIPy VectorVortexCoronagraph at a configurable charge.

    Supported on both backends. The jax port uses the circular-basis
    decomposition of the π-retardance vector vortex on unpolarized
    input: intensity = ½|V₊c E|² + ½|V₋c E|², i.e. two scalar
    multi-scale vortex trains at charges ±c (validated against hcipy's
    Jones-matrix propagation to ~1e-15). Non-π retardances are not
    expressible this way; this implementation fixes retardance at
    hcipy's default π.
    """

    name = "vector_vortex"
    supported_backends = frozenset({"hcipy", "jax"})

    def __init__(self, charge: int = 4, lyot: dict | None = None) -> None:
        self.charge = int(charge)
        self._lyot_cfg = lyot
        self._coro: Any | None = None
        self.lyot_field: Any | None = None
        self._pupil_grid: Any | None = None
        self._jax_sources: list[tuple[float, Any]] | None = None

    def _bind_pupil_grid(self, pupil_grid: Any) -> None:
        self._pupil_grid = pupil_grid
        self.lyot_field = _resolve_lyot_field(self._lyot_cfg, pupil_grid)
        # VectorVortexCoronagraph in HCIPy takes lyot_stop as a positional/kwarg
        self._coro = hcipy.VectorVortexCoronagraph(charge=self.charge, lyot_stop=self.lyot_field)

    def apply(self, wf: Any) -> Any:
        if self._coro is None:
            raise RuntimeError("VectorVortexCoronagraph must be bound via _bind_pupil_grid()")
        return self._coro(wf)

    def _jax_multi_scale_sources(self) -> list[tuple[float, Any]]:
        """Two half-weight scalar vortex channels at charges ±c.

        Masks for −c must come from their own multi-scale build (the
        correction-subtraction transforms do not commute with
        conjugation), so each channel gets a stop-free hcipy
        VortexCoronagraph whose precomputed masks the jax builder
        harvests. Built lazily and cached — the hcipy backend never
        pays for these.
        """
        if self._pupil_grid is None:
            raise RuntimeError("VectorVortexCoronagraph must be bound via _bind_pupil_grid()")
        if self._jax_sources is None:
            self._jax_sources = [
                (0.5, hcipy.VortexCoronagraph(self._pupil_grid, charge=self.charge)),
                (0.5, hcipy.VortexCoronagraph(self._pupil_grid, charge=-self.charge)),
            ]
        return self._jax_sources


__all__ = [
    "IdentityCoronagraph",
    "VortexCoronagraphImpl",
    "VectorVortexCoronagraphImpl",
]
