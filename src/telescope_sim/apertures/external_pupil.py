"""ExternalPupilAperture — wraps an arbitrary Python callable as an aperture.

Two calling conventions are supported via the ``mode`` field:

- ``"field"``: the function returns an HCIPy ``Field`` directly when called
  with a ``pupil_grid`` keyword. Use for callables like ``miles_pupil.
  generate_pupil(outer=..., pupil_grid=...)`` and ``miles_synthpsf.
  generate_pupil(...)``.
- ``"callable"``: the function returns an HCIPy aperture *callable* when
  called with whatever kwargs are supplied; the pipeline then evaluates it
  on the pupil grid via ``evaluate_supersampled``. Use for callables like
  ``hcipy.make_keck_aperture()`` or ``hcipy.make_circular_aperture(D)``.

The module is imported via ``importlib``; the path can be either a
standard dotted module name (``my_package.my_module``) or an absolute
filesystem path to a ``.py`` file (``/path/to/file.py``).
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import hcipy
from telescope_sim.abc import Aperture, ApertureResult
from telescope_sim.registry import register


def _load_callable(module: str, function: str) -> Any:
    """Resolve ``module:function`` to a callable; module is dotted or a path."""
    if module.endswith(".py") or "/" in module:
        mod_path = Path(module).resolve()
        if not mod_path.is_file():
            raise FileNotFoundError(f"module file not found: {mod_path}")
        spec = importlib.util.spec_from_file_location(mod_path.stem, mod_path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[mod_path.stem] = mod
        assert spec.loader is not None
        spec.loader.exec_module(mod)
    else:
        mod = importlib.import_module(module)
    try:
        return getattr(mod, function)
    except AttributeError as e:
        raise AttributeError(f"module {module!r} has no attribute {function!r}") from e


@register("aperture", "external_pupil")
@dataclass
class ExternalPupilAperture(Aperture):
    """Aperture built by an external Python callable.

    Parameters
    ----------
    module
        Dotted module name or filesystem path to a ``.py`` file.
    function
        Name of the callable inside the module.
    mode
        ``"field"`` (default) or ``"callable"`` — see module docstring.
    kwargs
        Extra kwargs forwarded to the callable.
    area
        Effective collecting area (used for photon flux scaling). Optional;
        defaults to ``0.0`` (only matters for noise computations).
    supersample
        Used only when ``mode == "callable"``.
    """

    module: str = ""
    function: str = "generate_pupil"
    mode: str = "field"
    kwargs: dict[str, Any] = field(default_factory=dict)
    area: float = 0.0
    supersample: int = 16

    def __post_init__(self) -> None:
        if self.mode not in ("field", "callable"):
            raise ValueError(f"mode must be 'field' or 'callable'; got {self.mode!r}")
        if not self.module:
            raise ValueError("ExternalPupilAperture requires a 'module' name or path")

    def build(self, pupil_grid: Any) -> ApertureResult:
        fn = _load_callable(self.module, self.function)
        if self.mode == "field":
            aper_field = fn(pupil_grid=pupil_grid, **self.kwargs)
        else:
            aper_callable = fn(**self.kwargs)
            aper_field = hcipy.evaluate_supersampled(aper_callable, pupil_grid, self.supersample)
        return ApertureResult(
            field=aper_field,
            area=float(self.area),
            segments=None,
            segment_coords=None,
            metadata={
                "source_module": self.module,
                "source_function": self.function,
                "mode": self.mode,
            },
        )


__all__ = ["ExternalPupilAperture"]
