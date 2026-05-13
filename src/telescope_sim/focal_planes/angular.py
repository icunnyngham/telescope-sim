"""Angular focal plane: arcsec-extent focal grid with FraunhoferPropagator.

This is the canonical-family focal plane: extent in arcsec, broadband
sampling via N monochromatic wavefronts across a fractional bandwidth.
The reference PSF (at-rest, no actuators, no atmosphere) is built once at
construction time and exposed via :attr:`reference_psf`,
:attr:`reference_peak_intensity`, and :attr:`reference_psf_sum`.

Detector noise is currently deferred; we'll add a NoisyDetector path when
the canonical fixtures that exercise it land. The canonical-family
fixtures #01/#02/#10/#11 don't use detectors.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

import hcipy
from telescope_sim.abc import FocalPlane
from telescope_sim.focal_planes.physical import FocalPlaneResult
from telescope_sim.registry import register


@dataclass
class _LamSetup:
    """Per-filter HCIPy artefacts (mirrors the legacy lam_setup dict)."""

    name: str
    central_lam: float
    fractional_bandwidth: float
    num_samples: int
    focal_grid: Any
    propagator: Any
    wavefronts: list[Any]
    filter_lams: NDArray[np.floating]
    reference_psf: NDArray[np.floating] | None = None
    reference_peak_intensity: float | None = None
    reference_psf_sum: float | None = None
    detector: Any | None = None


@register("focal_plane", "angular")
class AngularFocalPlane(FocalPlane):
    """One filter on an arcsec-extent focal grid.

    Parameters
    ----------
    central_lam
        Central wavelength in meters.
    focal_extent
        Full focal-plane extent in arcsec (square).
    focal_res
        Number of pixels per side (square).
    fractional_bandwidth
        Width of the broadband sampling range, fraction of central_lam.
    num_samples
        Number of monochromatic wavelength samples across the bandwidth.
    name
        Filter name, used as the wavefront name in the chain
        (e.g. ``"focal:H_band"``).
    """

    def __init__(
        self,
        *,
        central_lam: float,
        focal_extent: float,
        focal_res: int,
        fractional_bandwidth: float = 0.0,
        num_samples: int = 1,
        name: str = "filter",
    ) -> None:
        self.name = name
        self.central_lam = float(central_lam)
        self.focal_extent = float(focal_extent)
        self.focal_res = int(focal_res)
        self.fractional_bandwidth = float(fractional_bandwidth)
        self.num_samples = int(num_samples)
        # Populated by :meth:`build`
        self._lam_setup: _LamSetup | None = None
        self._aperture_field: Any | None = None
        self._pupil_grid: Any | None = None

    # --- Construction -------------------------------------------------------

    def build(self, pupil_grid: Any, aperture_field: Any) -> None:
        """Build the propagator, wavefronts, and reference PSF for this filter."""
        self._pupil_grid = pupil_grid
        self._aperture_field = aperture_field

        # Focal grid: convert arcsec extent to radians (HCIPy convention)
        fov_rad = self.focal_extent * np.pi / (180.0 * 3600.0)
        focal_grid = hcipy.make_uniform_grid([self.focal_res] * 2, fov_rad)
        prop = hcipy.FraunhoferPropagator(pupil_grid, focal_grid)

        if self.num_samples > 1:
            half = self.fractional_bandwidth / 2.0
            filter_lams = self.central_lam * np.linspace(1.0 - half, 1.0 + half, self.num_samples)
        else:
            filter_lams = np.array([self.central_lam])

        wavefronts = [hcipy.Wavefront(aperture_field, lam) for lam in filter_lams]

        self._lam_setup = _LamSetup(
            name=self.name,
            central_lam=self.central_lam,
            fractional_bandwidth=self.fractional_bandwidth,
            num_samples=self.num_samples,
            focal_grid=focal_grid,
            propagator=prop,
            wavefronts=wavefronts,
            filter_lams=filter_lams,
        )

    def compute_reference_psf(self, corrector_chain: list[Any]) -> None:
        """Walk the chain at-rest (no coronagraph) for normalization/strehl."""
        if self._lam_setup is None:
            raise RuntimeError("call build() before compute_reference_psf()")
        for c in corrector_chain:
            c.flatten()
        # Reference PSF always bypasses the coronagraph (legacy convention).
        result = self._propagate_chain(corrector_chain, coronagraph=None)
        ref = result.intensity
        self._lam_setup.reference_psf = ref
        self._lam_setup.reference_peak_intensity = float(ref.max())
        self._lam_setup.reference_psf_sum = float(ref.sum())

    # --- Per-sample -------------------------------------------------------

    def _propagate_chain(
        self,
        corrector_chain: list[Any],
        *,
        coronagraph: Any | None = None,
    ) -> FocalPlaneResult:
        """Propagate this filter's wavefronts through the chain.

        Order of operations per monochromatic wavefront:
            wf → c1 → c2 → ... → cN → (coronagraph?) → focal propagator → |.|^2
        Returns the summed intensity and the per-wavelength focal Wavefronts.
        """
        assert self._lam_setup is not None
        focal_wavefronts: list[Any] = []
        total = np.zeros((self.focal_res, self.focal_res), dtype=np.float64)
        for wf in self._lam_setup.wavefronts:
            wf_out = wf
            for c in corrector_chain:
                wf_out = c.apply(wf_out)
            if coronagraph is not None:
                wf_out = coronagraph.apply(wf_out)
            wf_focal = self._lam_setup.propagator(wf_out)
            focal_wavefronts.append(wf_focal)
            total += np.asarray(wf_focal.intensity.shaped)
        return FocalPlaneResult(intensity=total, wavefronts=focal_wavefronts)

    def propagate(self, wf: Any) -> Any:  # ABC method, not used by orchestrator directly
        assert self._lam_setup is not None
        return self._lam_setup.propagator(wf)

    # --- Convenience accessors ---------------------------------------------

    @property
    def reference_psf(self) -> NDArray[np.floating] | None:
        return None if self._lam_setup is None else self._lam_setup.reference_psf

    @property
    def reference_peak_intensity(self) -> float | None:
        return None if self._lam_setup is None else self._lam_setup.reference_peak_intensity

    @property
    def reference_psf_sum(self) -> float | None:
        return None if self._lam_setup is None else self._lam_setup.reference_psf_sum

    @property
    def lam_setup(self) -> _LamSetup:
        assert self._lam_setup is not None
        return self._lam_setup


__all__ = ["AngularFocalPlane"]
