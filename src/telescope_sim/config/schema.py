"""Pydantic v2 schemas for the telescope-sim configuration.

This module exposes the top-level :class:`SimConfig` and per-stage
sub-schemas. The schema is intentionally permissive on the per-stage
config payloads: those are passed straight through to the registered
implementation's constructor, which is free to validate further.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StageConfig(BaseModel):
    """Common shape: ``{type: <name>, ...payload}``.

    The ``type`` field selects which registered implementation to use;
    everything else is forwarded to its constructor.
    """

    model_config = ConfigDict(extra="allow")

    type: str


class CorrectorConfig(StageConfig):
    """A corrector + its role/target-strategy settings."""

    wavefront_role: Literal["actuate", "impose", "fit"] = "actuate"
    target_strategy: Literal[
        "none", "actuators", "actuators_plus_residual_fit", "residual_fit_only"
    ] = "none"
    fit_source: str | None = None
    target: bool = False


class PostProcessorConfig(BaseModel):
    """Either ``{type: <name>, ...payload}`` or just a bare string for no-arg processors."""

    model_config = ConfigDict(extra="allow")

    type: str


class OutputConfig(BaseModel):
    """One named output: a tap plus its ordered post-processing list."""

    model_config = ConfigDict(extra="forbid")

    tap: StageConfig
    post_processing: list[StageConfig | str] = Field(default_factory=list)


class PupilConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    resolution: int
    extent: float


class SimConfig(BaseModel):
    """Top-level simulation config."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    backend: Literal["hcipy", "jax"] = "hcipy"
    # Compute precision for the jax backend's propagation kernels; the
    # hcipy backend is float64-only, so "float32" requires backend=jax.
    precision: Literal["float64", "float32"] = "float64"
    pupil: PupilConfig
    aperture: StageConfig
    correctors: dict[str, CorrectorConfig] = Field(default_factory=dict)
    corrector_chain: list[str] = Field(default_factory=list)
    coronagraph: StageConfig | None = None
    focal_planes: dict[str, StageConfig]
    outputs: dict[str, OutputConfig]
    strehl_method: Literal["peak", "matched_filter"] | None = None
    strehl_core_rad: float | None = None

    @model_validator(mode="after")
    def _resolve_strehl(self) -> SimConfig:
        # Back-compat: if the user set strehl_core_rad but not strehl_method,
        # promote to matched_filter (the old "core_rad is not None" branch).
        # Otherwise default to peak.
        if self.strehl_method is None:
            self.strehl_method = "matched_filter" if self.strehl_core_rad is not None else "peak"
        if self.strehl_method == "matched_filter" and (
            self.strehl_core_rad is None or self.strehl_core_rad <= 0
        ):
            raise ValueError("strehl_method='matched_filter' requires a positive strehl_core_rad")
        return self


__all__ = [
    "SimConfig",
    "StageConfig",
    "CorrectorConfig",
    "PostProcessorConfig",
    "OutputConfig",
    "PupilConfig",
]
