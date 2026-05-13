"""Physical focal plane — focal-plane grid in physical units (meters).

Used when the focal-plane physics depends on absolute spot size rather
than angular extent (e.g. fiber coupling, where the MMF mode field is
defined in physical coordinates). Construction requires an explicit
``focal_length`` so the FraunhoferPropagator can convert between pupil
and focal-plane geometries.

This focal plane retains the per-wavelength focal-plane wavefronts so
downstream taps (notably ``fiber_coupled``) can use the full complex
field rather than just summed intensity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

import hcipy
from telescope_sim.abc import FocalPlane
from telescope_sim.registry import register


@dataclass
class _PhysicalLamSetup:
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


@dataclass
class FocalPlaneResult:
    """Per-sample focal-plane output retained for downstream taps."""

    intensity: NDArray[np.floating]  # summed intensity across wavelengths
    wavefronts: list[Any]  # per-wavelength focal Wavefronts


@register("focal_plane", "physical")
class PhysicalFocalPlane(FocalPlane):
    """Focal-plane grid in physical (metres) units.

    Parameters
    ----------
    central_lam
        Central wavelength in meters.
    focal_extent
        Full focal-plane extent in meters (square).
    focal_res
        Number of pixels per side.
    focal_length
        Optical focal length (meters); passed to ``FraunhoferPropagator``.
    fractional_bandwidth, num_samples
        Broadband sampling — same semantics as :class:`AngularFocalPlane`.
    """

    def __init__(
        self,
        *,
        central_lam: float,
        focal_extent: float,
        focal_res: int,
        focal_length: float,
        fractional_bandwidth: float = 0.0,
        num_samples: int = 1,
        wavefront_total_power: float | None = None,
        name: str = "filter",
    ) -> None:
        self.name = name
        self.central_lam = float(central_lam)
        self.focal_extent = float(focal_extent)
        self.focal_res = int(focal_res)
        self.focal_length = float(focal_length)
        self.fractional_bandwidth = float(fractional_bandwidth)
        self.num_samples = int(num_samples)
        # If set, force each monochromatic wavefront's total_power to this
        # value. Used by fiber-coupling variants that explicitly normalize.
        self.wavefront_total_power = wavefront_total_power
        self._lam_setup: _PhysicalLamSetup | None = None

    def build(self, pupil_grid: Any, aperture_field: Any) -> None:
        # Physical focal grid: extent in meters, build via make_pupil_grid
        # (which gives us a uniform metric grid; the legacy fiber variant
        # uses this exact construction).
        focal_grid = hcipy.make_pupil_grid(self.focal_res, self.focal_extent)
        prop = hcipy.FraunhoferPropagator(pupil_grid, focal_grid, focal_length=self.focal_length)

        if self.num_samples > 1:
            half = self.fractional_bandwidth / 2.0
            filter_lams = self.central_lam * np.linspace(1.0 - half, 1.0 + half, self.num_samples)
        else:
            filter_lams = np.array([self.central_lam])

        wavefronts = [hcipy.Wavefront(aperture_field, lam) for lam in filter_lams]
        if self.wavefront_total_power is not None:
            for wf in wavefronts:
                wf.total_power = float(self.wavefront_total_power)

        self._lam_setup = _PhysicalLamSetup(
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
        if self._lam_setup is None:
            raise RuntimeError("call build() before compute_reference_psf()")
        for c in corrector_chain:
            c.flatten()
        result = self._propagate_chain(corrector_chain, coronagraph=None)
        ref = result.intensity
        self._lam_setup.reference_psf = ref
        self._lam_setup.reference_peak_intensity = float(ref.max())
        self._lam_setup.reference_psf_sum = float(ref.sum())

    def _propagate_chain(
        self,
        corrector_chain: list[Any],
        *,
        coronagraph: Any | None = None,
    ) -> FocalPlaneResult:
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

    def propagate(self, wf: Any) -> Any:
        assert self._lam_setup is not None
        return self._lam_setup.propagator(wf)

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
    def lam_setup(self) -> _PhysicalLamSetup:
        assert self._lam_setup is not None
        return self._lam_setup


__all__ = ["PhysicalFocalPlane", "FocalPlaneResult"]
