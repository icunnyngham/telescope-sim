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
    ApertureResult,
    Coronagraph,
    Corrector,
    FocalPlane,
    OutputTap,
    PipelineContext,
    PostProcessor,
)
from telescope_sim.strehl import StrehlEstimator


def _mirror_of(corrector: Any) -> Any | None:
    """Return the HCIPy DM-like object backing a corrector, or None.

    Looks for ``_dm`` (``hcipy.DeformableMirror``, used by Zernike +
    custom basis correctors) then ``_sm``
    (``hcipy.SegmentedDeformableMirror``, used by ``SegmentedPTTCorrector``).
    Both expose ``.surface`` in meters of surface displacement. Returns
    ``None`` for correctors that don't back onto a DM, so the pipeline
    skips them in OPD bookkeeping.
    """
    return getattr(corrector, "_dm", None) or getattr(corrector, "_sm", None)


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
    strehl_estimators: dict[str, StrehlEstimator] = field(default_factory=dict)
    coronagraph: Coronagraph | None = None


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
        # Deferred to avoid the loader → pipeline import cycle.
        from telescope_sim.config.loader import build_from_yaml  # noqa: PLC0415

        return build_from_yaml(path)

    @classmethod
    def from_preset(cls, name: str) -> TelescopeSim:
        """Load a packaged preset by name."""
        # Deferred to avoid the loader → pipeline import cycle.
        from telescope_sim.config.loader import build_from_preset  # noqa: PLC0415

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

    def sample(  # noqa: PLR0912,PLR0915  (orchestration: branches + statements reflect chain stages)
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

        # 1) Apply actuator state to "actuate" and "impose" correctors.
        for c in self._c.correctors:
            if c.wavefront_role in {"actuate", "impose"}:
                values = actuations.get(c.name)
                if values is None:
                    c.flatten()
                else:
                    c.set_actuators(values)
            # "fit" correctors are resolved below

        # 2) Walk the chain in order, accumulating per-corrector cumulative
        #    pupil-plane OPD (= 2 × surface) "just before this corrector"
        #    and resolving fit-role correctors against it as we go. The
        #    snapshots are also saved for step 5 (residual-fit target
        #    strategies). Convention: fit_surface returns *matching*
        #    actuator values; we negate at the apply site for fit-role.
        correctors_by_name = {c.name: c for c in self._c.correctors}
        running_opd = np.zeros(self._c.pupil_grid.size, dtype=np.float64)
        cum_opd_pre: list[NDArray[np.float64]] = []
        seen_names: set[str] = set()

        for c in self._c.correctors:
            cum_opd_pre.append(running_opd.copy())

            if c.wavefront_role == "fit":
                fs = c.fit_source
                if fs is None or fs == "cumulative_phase_pre_self":
                    phase_in = cum_opd_pre[-1]
                elif fs in correctors_by_name and fs in seen_names:
                    other = correctors_by_name[fs]
                    other_mirror = _mirror_of(other)
                    if other_mirror is None:
                        raise ValueError(
                            f"corrector {c.name!r} fit_source={fs!r} "
                            "refers to a corrector with no DM surface to "
                            "fit to."
                        )
                    phase_in = 2.0 * np.asarray(other_mirror.surface)
                elif fs in correctors_by_name:
                    raise ValueError(
                        f"corrector {c.name!r} has wavefront_role='fit' with "
                        f"fit_source={fs!r}, but that corrector appears "
                        "later in the chain. fit_source must reference an "
                        "earlier corrector or use "
                        "'cumulative_phase_pre_self'."
                    )
                else:
                    raise ValueError(
                        f"corrector {c.name!r} has unknown fit_source={fs!r}. "
                        "Use 'cumulative_phase_pre_self' or the name of an "
                        "earlier corrector in the chain."
                    )
                fit_values = np.asarray(c.fit_surface(phase_in))
                c.set_actuators(-fit_values)

            # After this corrector's state is finalized, add its OPD
            # contribution to the running cumulative. Any corrector
            # exposing an HCIPy DM-like object via ``_dm`` (regular DM)
            # or ``_sm`` (segmented DM) contributes here.
            mirror = _mirror_of(c)
            if mirror is not None:
                running_opd = running_opd + 2.0 * np.asarray(mirror.surface)

            seen_names.add(c.name)

        # 3) Propagate each focal plane and collect FocalPlaneResult objects
        #    (each holds both summed intensity and per-wavelength wavefronts
        #    so downstream taps can pick what they need).
        fp_results: dict[str, Any] = {}
        for name, fp in self._c.focal_planes.items():
            fp_results[name] = fp._propagate_chain(
                self._c.correctors, coronagraph=self._c.coronagraph
            )

        # 4) Run output taps + per-output post-processors.
        images: dict[str, NDArray] = {}
        for out_spec in self._c.outputs:
            arr = out_spec.tap.extract(fp_results)

            # Build context for post-processors
            ref_peaks = [
                self._c.focal_planes[n].reference_peak_intensity for n in out_spec.focal_plane_names
            ]
            ref_sums = [
                self._c.focal_planes[n].reference_psf_sum for n in out_spec.focal_plane_names
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

        # 5) Build the actuation echo / Y output. Residual-fit strategies
        #    use the per-corrector cumulative-OPD snapshot computed in
        #    step 2. Convention: fit_surface returns *matching* values
        #    (the wavefront state in the corrector's basis), so the Y
        #    formulas below are unnegated — Y reports "what is at the
        #    pupil"; the ML model trainer applies -Y downstream to drive
        #    corrections. Matches legacy v1 ``out_actuate = caller +
        #    matching_fit(atmos)``.
        actuator_echo: dict[str, NDArray] = {}
        for i, c in enumerate(self._c.correctors):
            if not c.target:
                continue
            if c.target_strategy == "none":
                continue
            if c.target_strategy == "actuators":
                actuator_echo[c.name] = np.asarray(c.actuators)
            elif c.target_strategy == "actuators_plus_residual_fit":
                residual = np.asarray(c.fit_surface(cum_opd_pre[i]))
                actuator_echo[c.name] = np.asarray(c.actuators) + residual
            elif c.target_strategy == "residual_fit_only":
                actuator_echo[c.name] = np.asarray(c.fit_surface(cum_opd_pre[i]))

        result: dict[str, Any] = {
            "images": images,
            "actuations": actuator_echo,
        }

        # 6) Strehl. Estimators were built once at construction with the
        #    reference-PSF argmax / core mask / weighted sums cached, so
        #    this loop is O(1) (peak) or O(core_pixels) (matched_filter)
        #    per focal plane — same as the legacy `_strehl` call site.
        if meas_strehl:
            strehls: dict[str, float] = {}
            for name in self._c.focal_planes:
                est = self._c.strehl_estimators.get(name)
                if est is None:
                    continue
                strehls[name] = est.compute(fp_results[name].intensity)
            result["strehls"] = strehls

        return result


__all__ = ["TelescopeSim"]
