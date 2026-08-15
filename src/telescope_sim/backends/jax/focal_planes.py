"""jax-backend focal planes — JAX propagation behind the standard interfaces.

These subclass the hcipy focal planes so construction parameters, grid
metadata (``lam_setup``), reference-PSF bookkeeping, and Strehl/detector
integration are identical; only ``_propagate_chain`` differs. Instead of
walking correctors with per-wavelength ``apply()`` calls, the corrector
chain is composed as a single summed pupil-plane OPD (every corrector is a
thin phase screen, and thin phase screens commute) and propagated in one
jitted, wavelength-vmapped matrix Fourier transform.

Consequences vs the hcipy backend:

- ``FocalPlaneResult.wavefronts`` is empty (no per-wavelength hcipy
  wavefront objects exist on this path); taps that consume them
  (``fiber_dual``) are gated off at config time.
- ``atmos`` must expose ``.phase_for(lam)`` (OPD-defined screens); a plain
  wavefront-callable cannot be applied to a summed-OPD propagation.
- Coronagraphs: ``identity``, ``lyot``, ``vortex``, and
  ``vector_vortex`` are supported. The coronagraph train is folded into
  the propagation kernels at build time (the loader hands the bound
  coronagraph to the focal plane before ``build()``), so the science
  path applies it in-graph while the reference PSF keeps the plain
  path. Any other coronagraph kind is rejected at config time via
  ``supported_backends``; this module double-checks at sample time.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from telescope_sim.backends.jax.propagation import FraunhoferMFT
from telescope_sim.focal_planes.angular import AngularFocalPlane
from telescope_sim.focal_planes.physical import FocalPlaneResult, PhysicalFocalPlane
from telescope_sim.pipeline import _mirror_of
from telescope_sim.registry import register


def _chain_opd(
    corrector_chain: list[Any],
    atmos: Any | None,
    n_pix: int,
) -> np.ndarray:
    """Total pupil-plane OPD (meters, flat) for the current chain state.

    Mirrors the pipeline's cumulative-OPD bookkeeping: atmosphere seed
    (via ``phase_for``) plus 2 × surface per mirror-backed corrector.
    """
    opd = np.zeros(n_pix, dtype=np.float64)
    if atmos is not None:
        if not hasattr(atmos, "phase_for"):
            raise NotImplementedError(
                "the 'jax' backend requires an atmosphere exposing "
                ".phase_for(lam) (an OPD-defined phase screen); arbitrary "
                "wavefront callables are hcipy-backend only."
            )
        opd += np.asarray(atmos.phase_for(1.0), dtype=np.float64) / (2.0 * np.pi)
    for c in corrector_chain:
        mirror = _mirror_of(c)
        if mirror is None:
            raise ValueError(
                f"corrector {getattr(c, 'name', c)!r} exposes no mirror "
                "surface; it cannot run on the 'jax' backend."
            )
        opd += 2.0 * np.asarray(mirror.surface, dtype=np.float64)
    return opd


_JAX_CORONAGRAPHS = ("identity", "lyot", "vortex", "vector_vortex")


def _check_coronagraph(coronagraph: Any | None) -> None:
    if coronagraph is not None and getattr(coronagraph, "name", None) not in _JAX_CORONAGRAPHS:
        raise NotImplementedError(
            f"coronagraph {getattr(coronagraph, 'name', coronagraph)!r} is "
            "not supported on the 'jax' backend."
        )


def _wants_coronagraph(coronagraph: Any | None) -> bool:
    """True when the science path must propagate through a coronagraph train."""
    return coronagraph is not None and getattr(coronagraph, "name", None) != "identity"


class _JaxPropagationMixin:
    """Shared jax propagation path for the angular/physical focal planes."""

    _mft: FraunhoferMFT | None = None
    _amplitude: np.ndarray | None = None
    # The base build()'s hcipy propagator + wavefronts would go unused on
    # this path (everything propagates through the MFT kernels), so skip
    # constructing them; ``lam_setup.propagator`` is None here.
    _build_hcipy_propagator = False
    # Compute precision, set by the loader from the config's ``precision``
    # field before build(); float64 is the parity-first default.
    _precision: str = "float64"

    def _build_mft(self, *, focal_length: float, amplitude_scale: float = 1.0) -> None:
        aperture = np.asarray(self._aperture_field)
        if np.iscomplexobj(aperture):
            raise ValueError(
                f"focal plane {self.name!r}: complex aperture fields are not "
                "supported on the 'jax' backend (the MFT path propagates a "
                "real transmission amplitude); use the hcipy backend for "
                "apodized/complex pupils."
            )
        # The loader hands the pipeline's bound coronagraph to jax focal
        # planes before build() (like _precision) so the coronagraph train
        # can be folded into this plane's per-wavelength kernels.
        coronagraph = getattr(self, "_coronagraph", None)
        _check_coronagraph(coronagraph)
        setup = self.lam_setup
        self._mft = FraunhoferMFT(
            self._pupil_grid,
            setup.focal_grid,
            setup.filter_lams,
            focal_length=focal_length,
            dtype=self._precision,
            coronagraph=coronagraph if _wants_coronagraph(coronagraph) else None,
        )
        self._amplitude = aperture.astype(np.float64) * amplitude_scale

    def _propagate_chain(
        self,
        corrector_chain: list[Any],
        *,
        coronagraph: Any | None = None,
        atmos: Any | None = None,
    ) -> FocalPlaneResult:
        assert self._mft is not None and self._amplitude is not None
        _check_coronagraph(coronagraph)
        use_coro = _wants_coronagraph(coronagraph)
        if use_coro and self._mft._summed_intensity_coro is None:
            raise RuntimeError(
                f"focal plane {self.name!r} was built without the coronagraph "
                "bound; build the pipeline through the loader (build/from_yaml) "
                "so the coronagraph train is folded into the kernels."
            )
        opd = _chain_opd(corrector_chain, atmos, self._amplitude.size)
        intensity = self._mft.summed_intensity(self._amplitude, opd, coronagraph=use_coro)
        return FocalPlaneResult(intensity=intensity, wavefronts=[])

    def propagate(self, wf: Any) -> Any:
        raise NotImplementedError(
            "per-wavefront propagate() is hcipy-backend only; the jax focal "
            "planes propagate summed pupil-plane OPD through MFT kernels."
        )


@register("focal_plane", "angular", backend="jax")
class JaxAngularFocalPlane(_JaxPropagationMixin, AngularFocalPlane):
    """Angular focal plane propagated on JAX (see module docstring)."""

    def build(self, pupil_grid: Any, aperture_field: Any) -> None:
        super().build(pupil_grid, aperture_field)
        # Angular convention: focal coordinates in radians, focal_length=1.
        self._build_mft(focal_length=1.0)


@register("focal_plane", "physical", backend="jax")
class JaxPhysicalFocalPlane(_JaxPropagationMixin, PhysicalFocalPlane):
    """Physical focal plane propagated on JAX (see module docstring)."""

    def build(self, pupil_grid: Any, aperture_field: Any) -> None:
        super().build(pupil_grid, aperture_field)
        self._pupil_grid = pupil_grid
        self._aperture_field = aperture_field
        # wavefront_total_power rescales each monochromatic wavefront so its
        # integrated power equals the requested value — equivalent to
        # scaling the (λ-independent) aperture amplitude once.
        scale = 1.0
        if self.wavefront_total_power is not None:
            aper = np.asarray(aperture_field, dtype=np.float64)
            weights = np.atleast_1d(np.asarray(pupil_grid.weights, dtype=np.float64))
            current = float((np.abs(aper) ** 2 * weights).sum())
            scale = float(np.sqrt(self.wavefront_total_power / current))
        self._build_mft(focal_length=self.focal_length, amplitude_scale=scale)


__all__ = ["JaxAngularFocalPlane", "JaxPhysicalFocalPlane"]
