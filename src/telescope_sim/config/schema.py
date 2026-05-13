"""Pydantic v2 schemas for the telescope-sim configuration.

This module exposes the top-level :class:`SimConfig` and (eventually) the
per-stage sub-schemas. The current placeholder is permissive while the
concrete schema is under development.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict


class SimConfig(BaseModel):
    """Top-level simulation config. Permissive placeholder during development."""

    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)

    pupil: Any = None
    aperture: Any = None
    atmosphere: Any = None
    correctors: dict[str, Any] | None = None
    corrector_chain: list[str] | None = None
    coronagraph: Any = None
    focal_planes: dict[str, Any] | None = None
    outputs: dict[str, Any] | None = None
    cache: Any = None


__all__ = ["SimConfig"]
