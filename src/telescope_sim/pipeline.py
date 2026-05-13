"""Pipeline orchestrator — holds the optical chain and runs `sample()`.

This module exposes the top-level :class:`TelescopeSim` class, the entry
point for constructing a simulation from a YAML config, a preset, or a
validated pydantic config object.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from telescope_sim.abc import (
    Aperture,
    ApertureResult,
    Corrector,
    FocalPlane,
    OutputTap,
    PipelineContext,
    PostProcessor,
)
from telescope_sim.strehl import core_integral_strehl, peak_pixel_strehl


@dataclass
class _OutputSpec:
    """Per-output configuration assembled from the YAML schema."""

    name: str
    tap: OutputTap
    post_processors: list[PostProcessor] = field(default_factory=list)
    focal_plane_names: list[str] = field(default_factory=list)


@dataclass
class _PipelineComponents:
    """Resolved objects of a built pipeline. Held by :class:`TelescopeSim`."""

    pupil_grid: Any
    aperture: ApertureResult
    correctors: list[Corrector]
    focal_planes: dict[str, FocalPlane]
    outputs: list[_OutputSpec]
    strehl_core_rad: float | None = None


class TelescopeSim:
    """Composable telescope-PSF simulator.

    Construct via one of the classmethods (not via ``__init__`` directly):

    - :meth:`from_preset` — load a packaged preset by name
    - :meth:`from_yaml`   — load a user-supplied YAML config
    - :meth:`from_components` — instantiate from already-resolved components
      (primarily for the pipeline's own internal use and for advanced users
      bypassing the YAML/pydantic layer)
    """

    def __init__(self, components: _PipelineComponents) -> None:
        self._c = components
        self.atmosphere: Any = None  # No atmosphere wiring yet

    # --- Construction entry points -----------------------------------------

    @classmethod
    def from_components(cls, components: _PipelineComponents) -> TelescopeSim:
        """Instantiate from already-resolved pipeline components."""
        return cls(components)

    @classmethod
    def from_yaml(cls, path: str | Path) -> TelescopeSim:
        """Load a configuration YAML and build the pipeline."""
        from telescope_sim.config.loader import build_from_yaml

        return build_from_yaml(path)

    @classmethod
    def from_preset(cls, name: str) -> TelescopeSim:
        """Load a packaged preset by name."""
        from telescope_sim.config.loader import build_from_preset

        return build_from_preset(name)

    # --- Convenience accessors ---------------------------------------------

    @property
    def correctors(self) -> dict[str, Corrector]:
        return {c.name: c for c in self._c.correctors}

    @property
    def focal_planes(self) -> dict[str, FocalPlane]:
        return self._c.focal_planes

    @property
    def aperture(self) -> ApertureResult:
        return self._c.aperture

    # --- Main entry point --------------------------------------------------

    def sample(
        self,
        actuations: Mapping[str, ArrayLike] | None = None,
        *,
        meas_strehl: bool = False,
    ) -> dict[str, Any]:
        """Run the optical chain and return a dict of outputs.

        Parameters
        ----------
        actuations
            Per-corrector actuator state. Keys are corrector names (matching
            those declared in the config); each value is whatever shape that
            corrector's ``set_actuators`` accepts.
        meas_strehl
            If True, includes a ``strehls`` entry in the returned dict.

        Returns
        -------
        dict
            ``images``       — dict of output_name → numpy array
            ``actuations``   — dict of corrector_name → numpy array
                              (only for correctors with ``target=True``)
            ``strehls``      — present iff ``meas_strehl`` is True
        """
        actuations = dict(actuations or {})

        # 1) Apply actuator state to "actuate" correctors.
        for c in self._c.correctors:
            if c.wavefront_role == "actuate":
                values = actuations.get(c.name)
                if values is None:
                    c.flatten()
                else:
                    c.set_actuators(values)
            elif c.wavefront_role == "impose":
                values = actuations.get(c.name)
                if values is None:
                    c.flatten()
                else:
                    c.set_actuators(values)
            # "fit" correctors are resolved below

        # 2) Resolve "fit" correctors (lstsq fit to another corrector's surface
        #    or to a named wavefront). For Phase 2 (canonical family) this is
        #    a no-op; the path lands when Xinetics / fit-source patterns come
        #    in. Stub kept for clarity.
        for c in self._c.correctors:
            if c.wavefront_role == "fit":
                raise NotImplementedError(
                    f"corrector {c.name!r} has wavefront_role='fit' but "
                    "fit-source resolution is not yet implemented."
                )

        # 3) Propagate each focal plane and collect summed-intensity PSFs.
        psf_by_focal: dict[str, NDArray[np.floating]] = {}
        for name, fp in self._c.focal_planes.items():
            psf_by_focal[name] = fp._propagate_chain(self._c.correctors)

        # 4) Run output taps + per-output post-processors.
        images: dict[str, NDArray] = {}
        for out_spec in self._c.outputs:
            arr = out_spec.tap.extract(psf_by_focal)

            # Build context for post-processors
            ref_peaks = [
                self._c.focal_planes[n].reference_peak_intensity
                for n in out_spec.focal_plane_names
            ]
            ref_sums = [
                self._c.focal_planes[n].reference_psf_sum
                for n in out_spec.focal_plane_names
            ]
            ctx = PipelineContext(
                output_name=out_spec.name,
                focal_plane_name=",".join(out_spec.focal_plane_names),
                reference_peak_intensity=ref_peaks[0] if ref_peaks else None,
                reference_psf_sum=ref_sums[0] if ref_sums else None,
                extras={
                    "reference_peak_intensities": np.array(ref_peaks, dtype=np.float64),
                    "reference_psf_sums": np.array(ref_sums, dtype=np.float64),
                },
            )
            for pp in out_spec.post_processors:
                arr = pp(arr, ctx)
            images[out_spec.name] = np.asarray(arr)

        # 5) Build the actuation echo / Y output.
        actuator_echo: dict[str, NDArray] = {}
        for c in self._c.correctors:
            if not c.target:
                continue
            if c.target_strategy == "actuators":
                actuator_echo[c.name] = np.asarray(c.actuators)
            elif c.target_strategy == "none":
                continue
            elif c.target_strategy in (
                "actuators_plus_residual_fit",
                "residual_fit_only",
            ):
                raise NotImplementedError(
                    f"target_strategy={c.target_strategy!r} not yet wired up "
                    "(needs cumulative-phase tracking)."
                )

        result: dict[str, Any] = {
            "images": images,
            "actuations": actuator_echo,
        }

        # 6) Strehl.
        if meas_strehl:
            strehls: dict[str, float] = {}
            for name, fp in self._c.focal_planes.items():
                ref_peak = fp.reference_peak_intensity
                if ref_peak is None:
                    continue
                if self._c.strehl_core_rad is None:
                    strehls[name] = peak_pixel_strehl(psf_by_focal[name], ref_peak)
                else:
                    strehls[name] = core_integral_strehl(
                        psf_by_focal[name],
                        fp.reference_psf,
                        fp.lam_setup.focal_grid,
                        self._c.strehl_core_rad,
                    )
            result["strehls"] = strehls

        return result


__all__ = ["TelescopeSim"]
