"""Backend-selection mechanics and config-time gating for the jax backend.

The jax backend is opt-in and shares everything but propagation with the
default hcipy backend, so the interesting surface is *selection* (schema
default, YAML field, constructor override, registry overlay precedence)
and *refusal* (stages that have no jax implementation must fail at build
time with an actionable message, not silently at sample time).

Propagation-parity assertions live in ``test_jax_backend_parity.py``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import yaml
from pydantic import ValidationError

pytest.importorskip("jax", reason="jax backend requires the optional [jax] extra")

from telescope_sim import TelescopeSim  # noqa: E402
from telescope_sim.abc import Corrector  # noqa: E402

# Importing the backend package populates the "jax" registry overlay (and
# enables jax x64); the loader does the same lazily on backend="jax".
from telescope_sim.backends.jax.focal_planes import (  # noqa: E402
    JaxAngularFocalPlane,
    JaxPhysicalFocalPlane,
    _check_coronagraph,
)
from telescope_sim.config.loader import build  # noqa: E402
from telescope_sim.config.schema import SimConfig  # noqa: E402
from telescope_sim.focal_planes.angular import AngularFocalPlane  # noqa: E402
from telescope_sim.focal_planes.physical import PhysicalFocalPlane  # noqa: E402
from telescope_sim.registry import available, backend_registry, lookup, register  # noqa: E402

CIRCLE_APERTURE = {
    "type": "external_pupil",
    "module": "hcipy",
    "function": "make_circular_aperture",
    "mode": "callable",
    "kwargs": {"diameter": 1.0},
    "area": float(np.pi * 0.25),
}


def _base_config(**overrides: Any) -> dict:
    """Tiny single-wavelength config; deliberately cheap to build on both backends."""
    cfg: dict[str, Any] = {
        "pupil": {"resolution": 32, "extent": 1.05},
        "aperture": dict(CIRCLE_APERTURE),
        "correctors": {
            "dm": {
                "type": "zernike",
                "n_modes": 4,
                "zernike_diameter": 1.0,
                "starting_mode": 2,
                "actuate_scale": 1.0e-8,
            }
        },
        "corrector_chain": ["dm"],
        "focal_planes": {
            "filter1": {
                "type": "angular",
                "central_lam": 1.0e-6,
                "focal_extent": 1.0,
                "focal_res": 16,
                "num_samples": 1,
            }
        },
        "outputs": {
            "psf": {
                "tap": {"type": "intensity", "focal_planes": ["filter1"]},
                "post_processing": [],
            }
        },
    }
    cfg.update(overrides)
    return cfg


# --- Schema ------------------------------------------------------------------


def test_schema_backend_defaults_to_hcipy():
    assert SimConfig.model_validate(_base_config()).backend == "hcipy"


def test_schema_accepts_jax_backend():
    assert SimConfig.model_validate(_base_config(backend="jax")).backend == "jax"


def test_schema_rejects_unknown_backend():
    with pytest.raises(ValidationError, match="backend"):
        SimConfig.model_validate(_base_config(backend="numpy"))


# --- YAML field / constructor override ---------------------------------------


def _write_yaml(tmp_path, cfg) -> str:
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return str(path)


def _fp_types(sim: TelescopeSim) -> list[type]:
    return [type(fp) for fp in sim.focal_planes.values()]


def test_yaml_backend_field_selects_jax_focal_planes(tmp_path):
    """`backend: jax` in the YAML is honored with no code-side argument."""
    sim = TelescopeSim.from_yaml(_write_yaml(tmp_path, _base_config(backend="jax")))
    assert _fp_types(sim) == [JaxAngularFocalPlane]


def test_yaml_without_backend_field_stays_on_hcipy(tmp_path):
    sim = TelescopeSim.from_yaml(_write_yaml(tmp_path, _base_config()))
    assert _fp_types(sim) == [AngularFocalPlane]


def test_from_yaml_backend_argument_overrides_yaml_field(tmp_path):
    """The explicit argument wins in *both* directions (one YAML, either backend)."""
    jax_yaml = _write_yaml(tmp_path, _base_config(backend="jax"))
    forced_hcipy = TelescopeSim.from_yaml(jax_yaml, backend="hcipy")
    assert _fp_types(forced_hcipy) == [AngularFocalPlane]

    hcipy_yaml = str(tmp_path / "hcipy.yaml")
    with open(hcipy_yaml, "w") as f:
        yaml.safe_dump(_base_config(), f)
    forced_jax = TelescopeSim.from_yaml(hcipy_yaml, backend="jax")
    assert _fp_types(forced_jax) == [JaxAngularFocalPlane]


def test_physical_focal_plane_also_swaps(tmp_path):
    cfg = _base_config()
    cfg["focal_planes"] = {
        "filter1": {
            "type": "physical",
            "central_lam": 1.0e-6,
            "focal_extent": 2.0e-4,
            "focal_res": 16,
            "focal_length": 10.0,
            "num_samples": 1,
        }
    }
    path = _write_yaml(tmp_path, cfg)
    assert _fp_types(TelescopeSim.from_yaml(path)) == [PhysicalFocalPlane]
    assert _fp_types(TelescopeSim.from_yaml(path, backend="jax")) == [JaxPhysicalFocalPlane]


# --- Registry overlay --------------------------------------------------------


def test_overlay_shadows_agnostic_table_for_its_backend_only():
    assert lookup("focal_plane", "angular") is AngularFocalPlane
    assert lookup("focal_plane", "angular", backend="jax") is JaxAngularFocalPlane
    # No "hcipy" overlay exists — the default backend resolves the agnostic table.
    assert lookup("focal_plane", "angular", backend="hcipy") is AngularFocalPlane


def test_lookup_falls_back_to_agnostic_for_unshadowed_names():
    """Only focal planes are overlaid; every other kind falls through to the
    shared implementation, which is what makes one YAML runnable on both."""
    from telescope_sim.apertures.segmented_circular import SegmentedCircularAperture
    from telescope_sim.correctors.zernike import ZernikeCorrector

    assert lookup("corrector", "zernike", backend="jax") is ZernikeCorrector
    assert lookup("aperture", "segmented_circular", backend="jax") is SegmentedCircularAperture
    assert "physical" not in backend_registry["jax"]["corrector"]


def test_lookup_unknown_name_on_jax_reports_agnostic_availability():
    with pytest.raises(KeyError, match="focal_plane/no_such_plane is not registered"):
        lookup("focal_plane", "no_such_plane", backend="jax")


def test_lookup_unknown_backend_falls_back_to_agnostic():
    assert lookup("focal_plane", "angular", backend="not_a_backend") is AngularFocalPlane


@register("post_processor", "_overlay_only_probe", backend="_probe_backend")
class _OverlayOnlyProbe:
    """Registered for a fictitious backend only — must not leak elsewhere."""


def test_backend_registration_does_not_leak_into_agnostic_table():
    assert lookup("post_processor", "_overlay_only_probe", backend="_probe_backend") is (
        _OverlayOnlyProbe
    )
    assert "_overlay_only_probe" not in available("post_processor")
    with pytest.raises(KeyError, match="not registered"):
        lookup("post_processor", "_overlay_only_probe")


def test_register_rejects_duplicate_and_unknown_kind():
    with pytest.raises(ValueError, match="already registered"):
        register("post_processor", "_overlay_only_probe", backend="_probe_backend")(
            type("Other", (), {})
        )
    with pytest.raises(ValueError, match="Unknown registry kind"):
        register("not_a_kind", "x")


# --- Config-time refusals ----------------------------------------------------


def test_fiber_dual_tap_rejected_on_jax():
    """fiber_dual consumes per-λ hcipy focal wavefronts, which the summed-OPD
    jax propagation never materializes."""
    cfg = _base_config()
    cfg["focal_planes"] = {
        "filter1": {
            "type": "physical",
            "central_lam": 1.0e-6,
            "focal_extent": 2.0e-4,
            "focal_res": 16,
            "focal_length": 10.0,
            "num_samples": 1,
        }
    }
    cfg["outputs"] = {
        "fiber": {
            "tap": {
                "type": "fiber_dual",
                "focal_plane_name": "filter1",
                "fiber": {
                    "type": "step_index",
                    "core_radius": 2.5e-5,
                    "NA": 0.22,
                    "length": 1.0,
                },
            },
            "post_processing": [],
        }
    }
    config = SimConfig.model_validate(cfg)
    with pytest.raises(ValueError, match=r"output_tap/fiber_dual is not supported on the 'jax'"):
        build(config, backend="jax")


@pytest.mark.parametrize("coro_type", ["vortex", "vector_vortex"])
def test_vortex_coronagraphs_rejected_on_jax(coro_type):
    cfg = _base_config()
    cfg["coronagraph"] = {"type": coro_type, "charge": 2}
    config = SimConfig.model_validate(cfg)
    with pytest.raises(ValueError, match=rf"coronagraph/{coro_type} is not supported on the 'jax'"):
        build(config, backend="jax")


def test_identity_coronagraph_allowed_on_jax():
    cfg = _base_config()
    cfg["coronagraph"] = {"type": "identity"}
    sim = build(SimConfig.model_validate(cfg), backend="jax")
    assert sim.sample()["images"]["psf"].shape == (16, 16, 1)


def test_check_coronagraph_is_re_asserted_at_sample_time():
    """The propagation module double-checks even though the loader gates —
    a sim assembled via from_components() bypasses the loader entirely."""
    _check_coronagraph(None)
    _check_coronagraph(type("Fake", (), {"name": "identity"})())
    with pytest.raises(NotImplementedError, match="not supported on the 'jax' backend"):
        _check_coronagraph(type("Fake", (), {"name": "vortex"})())


# A corrector with no ``_dm`` / ``_sm``: the pipeline's ``_mirror_of`` returns
# None, so the jax backend has no OPD to sum for it. Subclasses the ABC so
# it stays invisible to the fit_surface contract guard in
# test_corrector_fit_contract.py (it inherits the base's raising default).
@register("corrector", "_no_mirror_probe")
class _NoMirrorCorrector(Corrector):
    def __init__(self, *, name="probe", **role_kwargs):
        self.name = name
        for k, v in role_kwargs.items():
            setattr(self, k, v)
        self._values = np.zeros(2)

    def apply(self, wf):
        return wf

    def set_actuators(self, values):
        self._values = np.atleast_1d(np.asarray(values, dtype=float))

    def flatten(self):
        self.set_actuators(np.zeros(2))

    @property
    def n_actuators(self):
        return 2

    @property
    def actuators(self):
        return self._values


def _no_mirror_config() -> dict:
    cfg = _base_config()
    cfg["correctors"] = {"weird": {"type": "_no_mirror_probe"}}
    cfg["corrector_chain"] = ["weird"]
    return cfg


def test_non_opd_corrector_rejected_on_jax():
    config = SimConfig.model_validate(_no_mirror_config())
    with pytest.raises(ValueError, match=r"\['weird'\] expose no mirror surface"):
        build(config, backend="jax")


def test_non_opd_corrector_is_fine_on_hcipy():
    """Control: the refusal is backend-specific, not a broken corrector."""
    sim = build(SimConfig.model_validate(_no_mirror_config()), backend="hcipy")
    assert sim.sample()["images"]["psf"].shape == (16, 16, 1)
