"""NoisyDetector post-processor.

Refactored in v2.0.0a9 from the v2.0.0a7 ``NoisyIntensityOutputTap``. The
math is unchanged — single-call ``Detector.integrate(power_field, dt=1)``
with the field built from ``intensity * grid.weights`` and optionally
rescaled to ``flux * area``. The move to a post-processor lets noise
compose with the new ``convolve_image`` processor (convolve first, then
apply noise to the convolved image — physically correct ordering).

Per-sample override flow: the pipeline populates
``PipelineContext.overrides`` with ``output_overrides[output_name]``
before invoking each post-processor in the chain. This processor reads
``int_phot_flux`` from the overrides dict, falling back to the YAML
default.

Loader dependencies (focal grid + aperture area) are injected via
``_bind_loader_dependencies`` at sim-build time. Eagerly constructs the
underlying ``hcipy.NoisyDetector`` so its ``flat_field`` setter's
``np.random.normal(...)`` call burns RNG state BEFORE any user-level
``np.random.seed()`` reset — matches legacy sampler-init timing.

Single-focal-plane only: the post-processor needs one focal grid, so
multi-channel outputs that want noise must split into one output per
filter.
"""

from __future__ import annotations

from typing import Any

import hcipy
import numpy as np
from numpy.typing import NDArray

from telescope_sim.abc import LoaderBindable, PipelineContext, PostProcessor
from telescope_sim.registry import register


@register("post_processor", "noisy_detector")
class NoisyDetectorPostProcessor(PostProcessor, LoaderBindable):
    """Apply HCIPy ``NoisyDetector`` noise to an intensity image.

    Parameters
    ----------
    int_phot_flux
        Photon flux in photons/m². When set, rescales the input image so
        total accumulated charge = ``int_phot_flux * aperture_area`` (the
        legacy `_addNoiseToObservation` contract). ``None`` integrates the
        input at its natural scale — useful when an upstream
        ``convolve_image`` post-processor has already imposed an absolute
        brightness.
    detector
        Sub-config forwarded to ``hcipy.NoisyDetector``: ``read_noise``,
        ``dark_current_rate``, ``flat_field``, ``include_photon_noise``,
        ``subsampling``.
    clamp_nonnegative
        If ``True``, apply ``np.abs`` to the read-out image. Mirrors the
        legacy ``_addNoiseToObservation`` workaround for Gaussian read
        noise pulling pixels negative. Disable for science-faithful
        analyses.
    """

    name = "noisy_detector"

    def __init__(
        self,
        *,
        int_phot_flux: float | None = None,
        detector: dict[str, Any] | None = None,
        clamp_nonnegative: bool = True,
    ) -> None:
        self.int_phot_flux = None if int_phot_flux is None else float(int_phot_flux)
        self.clamp_nonnegative = bool(clamp_nonnegative)
        self._detector_config = dict(detector) if detector else {}
        # Populated by the loader via _bind_loader_dependencies
        self._aperture_area: float | None = None
        self._focal_grid: Any | None = None
        self._detector: Any | None = None

    def _bind_loader_dependencies(
        self,
        *,
        aperture_result: Any,
        focal_planes: dict[str, Any],
        focal_plane_names: list[str],
    ) -> None:
        if len(focal_plane_names) != 1:
            raise ValueError(
                "noisy_detector post-processor takes exactly one focal_plane; "
                f"the output references {focal_plane_names!r}. For multi-filter "
                "noisy outputs, declare one output (with its own noisy_detector) "
                "per filter."
            )
        fp = focal_planes[focal_plane_names[0]]
        self._focal_grid = fp.lam_setup.focal_grid
        self._aperture_area = float(aperture_result.area)
        # Eager detector construction: burns the flat_field RNG draw at
        # sim-build, NOT at first sample call. This keeps seeded RNG
        # deterministic with respect to legacy (which built the detector
        # inside sampler __init__, before any user seed reset).
        self._detector = hcipy.NoisyDetector(self._focal_grid, **self._detector_config)

    def __call__(
        self,
        image: NDArray[np.floating],
        context: PipelineContext,
    ) -> NDArray[np.floating]:
        if self._detector is None or self._focal_grid is None:
            raise RuntimeError(
                "NoisyDetectorPostProcessor must be bound via "
                "_bind_loader_dependencies() (normally done by the YAML "
                "loader) before being called."
            )

        # Input image shape: (H, W, 1) from IntensityOutputTap or
        # ConvolveImagePostProcessor. Strip the trailing channel axis,
        # process, restore.
        if image.ndim != 3 or image.shape[-1] != 1:
            raise ValueError(
                "NoisyDetectorPostProcessor expects a single-channel image "
                f"of shape (H, W, 1); got {image.shape}. Use one output per "
                "filter for multi-channel noisy detectors."
            )
        psf_or_scene = np.asarray(image[..., 0], dtype=np.float64)

        # Per-sample override falls back to YAML default
        effective_flux = context.overrides.get("int_phot_flux", self.int_phot_flux)

        # Build the power Field: HCIPy convention is `power = |E|² * weights`,
        # so the broadband equivalent for an intensity image is `intensity *
        # grid.weights`.
        weights_arr = np.asarray(self._focal_grid.weights)
        power_field = hcipy.Field(
            psf_or_scene.ravel() * weights_arr,
            self._focal_grid,
        )

        if effective_flux is not None:
            natural_total = float(power_field.sum())
            if natural_total > 0:
                power_field = power_field * ((effective_flux * self._aperture_area) / natural_total)

        # Single integrate call ("one exposure"): dark adds at natural
        # rate, read noise applied once at read_out.
        self._detector.integrate(power_field, dt=1.0, weight=1.0)

        img = np.asarray(self._detector.read_out().shaped, dtype=np.float64)
        if self.clamp_nonnegative:
            img = np.abs(img)
        return img[..., None]


__all__ = ["NoisyDetectorPostProcessor"]
