"""SimulateMultiApertureTelescope v1.x compatibility shim.

Maps the v1.x ``SimulateMultiApertureTelescope`` constructor's keyword
arguments to a v2 config and delegates through to a
:class:`telescope_sim.TelescopeSim`. **Best-effort** — it covers the
common path (ELF / monolithic mirror layouts, single filter, segmented
PTT) and falls back to ``NotImplementedError`` for the rarer kwargs.

Users with custom setups should migrate to a YAML config; this shim is
just to lower the barrier to a quick v2 trial. Documented as deprecated;
slated for removal in a future release.

Example
-------
The legacy two-liner::

    from telescope_sim.legacy import SimulateMultiApertureTelescope
    sim = SimulateMultiApertureTelescope(
        mirror_layout="elf",
        telescope_radius=1.25,
        sub_aperture_count=15,
        filter_central_wavelength=0.75e-6,
        focal_extent=1.3248,
        focal_res=64,
    )
    out = sim.get_observation()

works exactly as before, but ``sim`` is a v2 pipeline under the hood.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np

# Defaults pulled from the v1 argparser, kept here to avoid pulling
# argparse and joblib into v2. Add fields as you discover the shim is
# missing them in real-world calls.
_V1_DEFAULTS: dict[str, Any] = {
    "mirror_layout": "elf",
    "telescope_radius": 1.25,
    "sub_aperture_count": 15,
    "sub_aperture_radius": None,  # auto from chord
    "pupil_res": 256,
    "pupil_extent": None,  # auto
    "piston_scale": 1e-6,
    "tip_tilt_scale": 1e-6,
    "spider_width": None,
    "spider_angle": 0,
    "filter_central_wavelength": 0.75e-6,
    "filter_fractional_bandwidth": 0.05,
    "filter_num_samples": 3,
    "focal_extent": 1.3248,
    "focal_res": 64,
    "extra_processing": None,
}

_UNSUPPORTED_KWARGS: set[str] = {
    "telescope_setup_pkl",
    "atmosphere_type",
    "atmosphere_fried_parameter",
    "atmosphere_velocity",
    "atmosphere_outer_scale",
    "enable_atmosphere_scintillation",
    "slew_deg_per_sec",
    "slew_focal_plane_angle",
    "integrated_photon_flux",
    "read_noise",
    "dm_actuator_num",
    "dm_actuator_spacing",
    "directly_actuate_dm",
    "apply_optimal_actuator_corrections",
    "add_ptt_perturbations_sigma",
}


class SimulateMultiApertureTelescope:
    """Drop-in (best-effort) replacement for the v1.x high-level wrapper.

    Construction-time kwargs are mapped to a v2 config dict and used to
    build a :class:`telescope_sim.TelescopeSim`. ``get_observation()`` and
    ``sample()`` delegate to the underlying v2 pipeline.

    Deprecated — use :class:`telescope_sim.TelescopeSim` directly for new
    code. Migration paths are documented in the package docs.
    """

    def __init__(self, **kwargs: Any) -> None:
        warnings.warn(
            "telescope_sim.legacy.SimulateMultiApertureTelescope is deprecated. "
            "Migrate to telescope_sim.TelescopeSim.from_yaml() for new code.",
            DeprecationWarning,
            stacklevel=2,
        )

        unsupported = sorted(set(kwargs) & _UNSUPPORTED_KWARGS)
        if unsupported:
            raise NotImplementedError(
                f"v1.x kwargs {unsupported} aren't yet supported by the "
                "v2 shim. Open an issue or migrate to a YAML config."
            )

        # Merge with v1 defaults
        cfg = dict(_V1_DEFAULTS)
        cfg.update(kwargs)

        v2_config = _build_v2_config(cfg)

        # Defer the TelescopeSim import so importing this shim doesn't
        # drag in HCIPy unless the user actually uses it.
        from telescope_sim.config.loader import build  # noqa: PLC0415
        from telescope_sim.config.schema import SimConfig  # noqa: PLC0415

        self._sim = build(SimConfig.model_validate(v2_config))

    def get_observation(self, **sample_kwargs: Any) -> tuple:
        """Legacy ``get_observation`` shape: returns ``(X, Y[, strehls])``."""
        out = self._sim.sample(**sample_kwargs)
        image_name = next(iter(out["images"]))
        actuator_name = next(iter(out["actuations"])) if out["actuations"] else None
        x = out["images"][image_name]
        y = out["actuations"][actuator_name] if actuator_name else np.zeros(0)
        if "strehls" in out:
            return x, y, np.array(list(out["strehls"].values()))
        return x, y

    def sample(self, **kwargs: Any) -> dict:
        """Pass-through to the v2 sample dict."""
        return self._sim.sample(**kwargs)

    @property
    def sim(self):
        """The underlying v2 :class:`TelescopeSim`."""
        return self._sim


def _build_v2_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """Translate the merged v1 kwargs into a v2 SimConfig-shaped dict."""
    layout = cfg["mirror_layout"]
    if layout not in ("elf", "monolithic"):
        raise NotImplementedError(
            f"mirror_layout={layout!r} is not yet supported by the shim. Migrate to a YAML config."
        )

    n_seg = int(cfg["sub_aperture_count"])
    tel_r = float(cfg["telescope_radius"])

    if layout == "monolithic":
        seg_diam = 2.0 * tel_r
        n_seg = 1
        positions = [[0.0, 0.0]]
        aperture: dict[str, Any] = {
            "type": "segmented_circular",
            "segment_diameter": seg_diam,
            "layout": "custom",
            "positions": positions,
        }
        pupil_extent = cfg["pupil_extent"] or (1.05 * 2.0 * tel_r)
    else:
        # ELF chord-derived sub-aperture diameter (matches the canonical
        # auto-compute when sub_aperture_radius is not given)
        sub_r = cfg["sub_aperture_radius"]
        if sub_r is None:
            seg_diam = 2.0 * tel_r * float(np.sin(np.pi / n_seg))
        else:
            seg_diam = 2.0 * float(sub_r)
        aperture = {
            "type": "segmented_circular",
            "segment_diameter": seg_diam,
            "layout": "elf",
            "n_segments": n_seg,
            "ring_radius": tel_r,
            "supersample": 16,
        }
        # max segment-center extent (slightly less than 2*tel_r for odd N)
        thetas = np.linspace(0, 2 * np.pi, n_seg + 1)[:-1]
        xs = tel_r * np.cos(thetas)
        ys = tel_r * np.sin(thetas)
        max_extent = float(max(xs.max() - xs.min(), ys.max() - ys.min()))
        pupil_extent = cfg["pupil_extent"] or (max_extent + seg_diam) * 1.05

    if cfg["spider_width"]:
        aperture["spider"] = {
            "width": float(cfg["spider_width"]),
            "angle": float(cfg["spider_angle"]),
        }

    config = {
        "pupil": {"resolution": int(cfg["pupil_res"]), "extent": pupil_extent},
        "aperture": aperture,
        "correctors": {
            "segments": {
                "type": "segmented_ptt",
                "piston_scale": float(cfg["piston_scale"]),
                "tip_tilt_scale": float(cfg["tip_tilt_scale"]),
                "wavefront_role": "actuate",
                "target_strategy": "actuators",
                "target": True,
            }
        },
        "corrector_chain": ["segments"],
        "focal_planes": {
            "filter1": {
                "type": "angular",
                "central_lam": float(cfg["filter_central_wavelength"]),
                "focal_extent": float(cfg["focal_extent"]),
                "focal_res": int(cfg["focal_res"]),
                "fractional_bandwidth": float(cfg["filter_fractional_bandwidth"]),
                "num_samples": int(cfg["filter_num_samples"]),
            }
        },
        "outputs": {
            "psf": {
                "tap": {"type": "intensity", "focal_planes": ["filter1"]},
                "post_processing": [{"type": "max_intensity_norm"}],
            }
        },
        "strehl_core_rad": None,
    }
    return config


__all__ = ["SimulateMultiApertureTelescope"]
