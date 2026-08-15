"""Smoke tests — verify the package imports and the registry is wired up."""

from __future__ import annotations

import pytest


def test_package_imports() -> None:
    import telescope_sim

    assert hasattr(telescope_sim, "TelescopeSim")
    assert hasattr(telescope_sim, "register")
    assert hasattr(telescope_sim, "__version__")


def test_registry_has_all_kinds() -> None:
    from telescope_sim.registry import registry

    expected_kinds = {
        "aperture",
        "corrector",
        "coronagraph",
        "focal_plane",
        "output_tap",
        "post_processor",
    }
    assert set(registry) == expected_kinds


def test_register_and_lookup_roundtrip() -> None:
    from telescope_sim.abc import Aperture
    from telescope_sim.registry import lookup, register

    @register("aperture", "_test_dummy")
    class _DummyAperture(Aperture):
        def build(self, pupil_grid):  # noqa: ANN001
            return None

    assert lookup("aperture", "_test_dummy") is _DummyAperture


def test_register_duplicate_raises() -> None:
    from telescope_sim.abc import Aperture
    from telescope_sim.registry import register

    @register("aperture", "_test_dupe")
    class _A(Aperture):
        def build(self, pupil_grid):  # noqa: ANN001
            return None

    with pytest.raises(ValueError, match="already registered"):

        @register("aperture", "_test_dupe")
        class _B(Aperture):  # noqa: F811
            def build(self, pupil_grid):  # noqa: ANN001
                return None


def test_from_preset_unknown_lists_available() -> None:
    """Asking for a nonexistent preset reports what's actually available."""
    from telescope_sim import TelescopeSim

    with pytest.raises(ValueError, match="unknown preset"):
        TelescopeSim.from_preset("__does_not_exist__")


def test_from_preset_elf_15seg_builds() -> None:
    """The bundled sELF preset constructs a usable pipeline."""
    from telescope_sim import TelescopeSim

    sim = TelescopeSim.from_preset("elf_15seg")
    assert "segments" in sim.correctors
    assert set(sim.focal_planes) == {"filter1"}
