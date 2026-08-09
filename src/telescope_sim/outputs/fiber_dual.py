"""FiberDualOutputTap — dual-stack focal intensity + fiber-coupled intensity.

For dual-output fiber experiments (e.g. fixture #15): runs each focal-plane
wavefront through a multi-mode fiber (HCIPy ``StepIndexFiber``) and produces
``np.stack([focal_intensity, fiber_intensity])`` summed over wavelengths.

Returns shape ``(2, H, W, 1)`` matching the legacy fiber variant — first
axis is ``[focal, mmf]``, last axis is the (single) filter channel.
"""

from __future__ import annotations

from typing import Any

import hcipy
import numpy as np
from numpy.typing import NDArray

from telescope_sim.abc import OutputTap
from telescope_sim.registry import register


@register("output_tap", "fiber_dual")
class FiberDualOutputTap(OutputTap):
    """Stack focal-plane intensity and fiber-coupled intensity along axis 0.

    Parameters
    ----------
    focal_plane_name
        Name of the (physical) focal plane to consume.
    fiber
        Sub-config for the fiber: ``{type, ...kwargs}``. Currently only
        ``"step_index"`` is supported.
    name
        Output name (key in ``sample()['images']``).
    """

    # Couples per-wavelength hcipy focal wavefronts into an hcipy fiber
    # model; there is no dlux equivalent of this path yet.
    supported_backends = frozenset({"hcipy"})

    def __init__(
        self,
        focal_plane_name: str,
        fiber: dict,
        *,
        name: str = "fiber_dual",
    ) -> None:
        self.name = name
        self.focal_plane_name = focal_plane_name
        self.source = f"focal:{focal_plane_name}"
        self.focal_plane_names = [focal_plane_name]
        self._fiber_config = dict(fiber)
        self._fiber: Any | None = None

    def _build_fiber(self) -> Any:
        cfg = dict(self._fiber_config)
        type_name = cfg.pop("type")
        if type_name == "step_index":
            max_cache = cfg.pop("max_in_cache", None)
            fiber = hcipy.StepIndexFiber(**cfg)
            if max_cache is not None:
                fiber._max_in_cache = int(max_cache)
            return fiber
        raise ValueError(f"unsupported fiber type {type_name!r}")

    def extract(
        self, fp_results: Any, *, overrides: dict[str, Any] | None = None
    ) -> NDArray[np.floating]:
        del overrides  # this tap has no per-sample state
        if self._fiber is None:
            self._fiber = self._build_fiber()
        if not isinstance(fp_results, dict):
            raise TypeError(
                "FiberDualOutputTap.extract expects a dict of FocalPlaneResults; "
                f"got {type(fp_results).__name__}"
            )
        if self.focal_plane_name not in fp_results:
            raise KeyError(
                f"focal plane {self.focal_plane_name!r} not in available outputs {list(fp_results)}"
            )
        result = fp_results[self.focal_plane_name]
        focal_total = np.asarray(result.intensity, dtype=np.float64)
        mmf_total = np.zeros_like(focal_total)
        for wf_focal in result.wavefronts:
            mmf_intensity = self._fiber(wf_focal).intensity
            mmf_total += np.asarray(mmf_intensity.shaped, dtype=np.float64)
        # Stack along axis 0 and add the trailing filter-channel axis
        stacked = np.stack([focal_total, mmf_total], axis=0)
        return stacked[..., None]


__all__ = ["FiberDualOutputTap"]
