"""NoisyIntensityOutputTap — single-integrate noise through an HCIPy ``NoisyDetector``.

Closes the long-standing ``NoisyDetector`` feature gap from the v2.0.0a6 audit.

The legacy fiber/canonical variants applied noise via a Wavefront-
reconstruction workaround (sqrt the summed PSF intensity, fake a Wavefront,
override ``total_power``, integrate). The "fake Wavefront" step was
unnecessary: HCIPy's ``Detector.integrate(wavefront, dt, weight)`` accepts
either a Wavefront OR an array-like as the power input
([detector.py:226-234]). We pass the wavelength-summed intensity through as
a ``hcipy.Field`` (intensity × grid.weights) — same single-call exposure
semantic the legacy intended, none of the reconstruction overhead.

Per-sample overrides (from ``sim.sample(output_overrides=...)``) can replace
the YAML default for ``int_phot_flux``. RNG is HCIPy's global ``np.random.*``;
for deterministic outputs call ``np.random.seed(N)`` immediately before
``sim.sample()``.

Output shape: ``(H, W, 1)`` — matches the canonical channels-last convention
so downstream post-processors (e.g., ``max_intensity_norm``) operate
uniformly across noisy and clean taps.
"""

from __future__ import annotations

from typing import Any

import hcipy
import numpy as np
from numpy.typing import NDArray

from telescope_sim.abc import OutputTap
from telescope_sim.registry import register


@register("output_tap", "noisy_intensity")
class NoisyIntensityOutputTap(OutputTap):
    """Noisy focal-plane intensity via HCIPy's ``NoisyDetector``.

    Parameters
    ----------
    focal_planes
        One focal-plane name to consume. (Single-focal-plane only — for
        multi-filter noisy outputs, configure one tap per filter.)
    int_phot_flux
        Photon flux in photons/m². The detector integrates
        ``int_phot_flux * aperture_area`` photons total per sample (spread
        across all wavelength samples). ``None`` disables flux scaling — the
        wavefront's natural ``total_power`` is used as-is, useful for unit-
        power references.
    aperture_area
        Effective collecting area in m². Defaults to ``None``; the YAML loader
        injects ``ApertureResult.area`` if left unset.
    detector
        Sub-config forwarded to ``hcipy.NoisyDetector``. Common keys:
        ``read_noise``, ``dark_current_rate``, ``flat_field``,
        ``include_photon_noise``, ``subsampling``.
    clamp_nonnegative
        If ``True``, apply ``np.abs`` to the read-out image. Mirrors the
        legacy ``_addNoiseToObservation`` workaround for Gaussian read noise
        producing negative pixels. Disable for science-faithful analyses.
    name
        Output name (used as the key in ``sample()``'s returned ``images``).
    """

    def __init__(
        self,
        focal_plane_names: list[str] | str,
        *,
        int_phot_flux: float | None = None,
        aperture_area: float | None = None,
        detector: dict[str, Any] | None = None,
        clamp_nonnegative: bool = True,
        name: str = "noisy_psf",
    ) -> None:
        if isinstance(focal_plane_names, str):
            focal_plane_names = [focal_plane_names]
        if len(focal_plane_names) != 1:
            raise ValueError(
                "NoisyIntensityOutputTap takes exactly one focal_plane; "
                f"got {focal_plane_names!r}. For multi-filter noisy outputs, "
                "declare one noisy_intensity tap per filter."
            )
        self.name = name
        self.focal_plane_name = focal_plane_names[0]
        self.focal_plane_names = list(focal_plane_names)
        self.source = f"focal:{self.focal_plane_name}"

        self.int_phot_flux = None if int_phot_flux is None else float(int_phot_flux)
        self.aperture_area = None if aperture_area is None else float(aperture_area)
        self.clamp_nonnegative = bool(clamp_nonnegative)
        self._detector_config = dict(detector) if detector else {}
        self._detector: Any | None = None

    def _build_detector(self, detector_grid: Any) -> Any:
        """Construct the underlying HCIPy NoisyDetector once."""
        return hcipy.NoisyDetector(detector_grid, **self._detector_config)

    def extract(
        self,
        fp_results: Any,
        *,
        overrides: dict[str, Any] | None = None,
    ) -> NDArray[np.floating]:
        if not isinstance(fp_results, dict):
            raise TypeError(
                "NoisyIntensityOutputTap.extract expects a dict of FocalPlaneResults; "
                f"got {type(fp_results).__name__}"
            )
        if self.focal_plane_name not in fp_results:
            raise KeyError(
                f"focal plane {self.focal_plane_name!r} not in available outputs {list(fp_results)}"
            )
        result = fp_results[self.focal_plane_name]

        # Per-sample override falls back to YAML default
        overrides = overrides or {}
        effective_flux = overrides.get("int_phot_flux", self.int_phot_flux)

        # Build the detector once (lazy: needs the focal grid, which we read
        # from the first wavefront).
        focal_grid = result.wavefronts[0].grid
        if self._detector is None:
            self._detector = self._build_detector(focal_grid)

        # Convert wavelength-summed intensity to a power Field. Per HCIPy
        # convention, `wf.power == |E|² * grid.weights` — so the broadband
        # equivalent is `result.intensity * grid.weights` (intensity is the
        # raw sum of |E_λ|² without weights).
        power_field = hcipy.Field(
            np.asarray(result.intensity).ravel() * np.asarray(focal_grid.weights),
            focal_grid,
        )

        if effective_flux is not None:
            if self.aperture_area is None:
                raise RuntimeError(
                    "NoisyIntensityOutputTap requires aperture_area when "
                    "int_phot_flux is set. The loader normally injects this "
                    "from ApertureResult.area; if you're constructing the "
                    "tap by hand, pass aperture_area explicitly."
                )
            natural_total = float(power_field.sum())
            if natural_total > 0:
                power_field = power_field * ((effective_flux * self.aperture_area) / natural_total)

        # Single integrate call ("one exposure"): dark adds at natural rate,
        # read noise is applied once at read_out.
        self._detector.integrate(power_field, dt=1.0, weight=1.0)

        img = np.asarray(self._detector.read_out().shaped, dtype=np.float64)
        if self.clamp_nonnegative:
            img = np.abs(img)
        # Trailing channel axis for shape-consistency with IntensityOutputTap
        return img[..., None]


__all__ = ["NoisyIntensityOutputTap"]
