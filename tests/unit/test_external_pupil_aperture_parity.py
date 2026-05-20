"""Parity tests for ``ExternalPupilAperture`` against legacy aper_func/vp.generate_pupil paths.

Two patterns appear in legacy code:

- **callable mode** — used by ``variants/coro__coro_mas_psf.py:124``:
      self.aper = hcipy.evaluate_supersampled(aper_func(), self.pupil_grid, 8)

- **field mode** — used by ``variants/vampires_*_2024-05__coro_mas_psf.py:125``:
      self.aper = vp.generate_pupil(outer=7.79/7.92, pupil_grid=self.pupil_grid)

(``vp`` here is ``test_fixtures/helpers/miles_pupil``; a Field-returning
helper that takes the grid as a kwarg.)

The v2 ``ExternalPupilAperture`` should reproduce *both* paths bit-for-bit
when configured to match. Drift points to audit:

- Does ``mode="field"`` actually call ``fn(pupil_grid=pupil_grid, **kwargs)``?
- Does ``mode="callable"`` correctly chain ``fn(**kwargs)`` then
  ``hcipy.evaluate_supersampled(..., supersample)``?
- Does the ``supersample`` config override the default (16 in v2 vs 8 in the
  legacy callable path)?
- Are dotted module names AND filesystem paths both loadable?
"""

from __future__ import annotations

import textwrap

import hcipy
import numpy as np
import pytest

# Trigger registration
import telescope_sim.apertures.external_pupil  # noqa: F401


@pytest.fixture(scope="module")
def pupil_grid():
    return hcipy.make_pupil_grid(64, 1.0)


# --- mode="callable" parity (the coro_mas_psf lineage) ---------------------


def test_external_pupil_callable_mode_matches_legacy_evaluate_supersampled(pupil_grid):
    """v2 callable mode matches `evaluate_supersampled(fn(**kwargs), grid, supersample)`."""
    from telescope_sim.apertures.external_pupil import ExternalPupilAperture

    # Legacy direct path (coro__coro_mas_psf.py:124 with supersample=8)
    legacy_callable = hcipy.aperture.make_obstructed_circular_aperture(
        pupil_diameter=0.85, central_obscuration_ratio=0.3
    )
    legacy_field = hcipy.evaluate_supersampled(legacy_callable, pupil_grid, 8)

    # v2 wrapper
    aper = ExternalPupilAperture(
        module="hcipy.aperture",
        function="make_obstructed_circular_aperture",
        mode="callable",
        kwargs={"pupil_diameter": 0.85, "central_obscuration_ratio": 0.3},
        supersample=8,
    )
    result = aper.build(pupil_grid)

    np.testing.assert_allclose(np.asarray(result.field), np.asarray(legacy_field), rtol=0, atol=0)


def test_external_pupil_supersample_default_is_16(pupil_grid):
    """The documented default supersample is 16 (NOT 8 — that's the legacy coro choice)."""
    from telescope_sim.apertures.external_pupil import ExternalPupilAperture

    aper = ExternalPupilAperture(
        module="hcipy",
        function="make_circular_aperture",
        mode="callable",
        kwargs={"diameter": 0.8},
    )
    assert aper.supersample == 16

    # And the field built at default 16 differs from one built at supersample=8
    aper_ss16 = aper
    aper_ss8 = ExternalPupilAperture(
        module="hcipy",
        function="make_circular_aperture",
        mode="callable",
        kwargs={"diameter": 0.8},
        supersample=8,
    )
    f16 = np.asarray(aper_ss16.build(pupil_grid).field)
    f8 = np.asarray(aper_ss8.build(pupil_grid).field)
    # Should differ on the boundary pixels (where supersampling matters)
    assert np.linalg.norm(f16 - f8) > 0


# --- mode="field" parity (the vampires / miles_pupil lineage) --------------


def test_external_pupil_field_mode_passes_pupil_grid_kwarg(pupil_grid, tmp_path):
    """v2 field mode calls fn(pupil_grid=pupil_grid, **kwargs) and returns the Field.

    Builds a stand-in module that mimics ``miles_pupil.generate_pupil(outer=..., pupil_grid=...)``
    and verifies (a) v2 calls it with the pupil_grid kwarg and (b) the returned
    Field passes through unchanged.
    """
    from telescope_sim.apertures.external_pupil import ExternalPupilAperture

    # Standalone module on disk — also exercises the filesystem-path loading branch
    mod_path = tmp_path / "_test_pupil_fn.py"
    mod_path.write_text(
        textwrap.dedent("""
            import hcipy
            import numpy as np

            CALL_LOG = []

            def generate_pupil(outer, pupil_grid, **extra):
                CALL_LOG.append(
                    {"outer": outer, "pupil_grid_size": pupil_grid.size, "extra": extra}
                )
                aper = hcipy.make_circular_aperture(outer)
                return hcipy.evaluate_supersampled(aper, pupil_grid, 16)
        """)
    )

    aper = ExternalPupilAperture(
        module=str(mod_path),
        function="generate_pupil",
        mode="field",
        kwargs={"outer": 0.8},
    )
    result = aper.build(pupil_grid)

    # Same module is now in sys.modules; verify CALL_LOG to confirm the call
    # signature was right
    import sys

    mod = sys.modules[mod_path.stem]
    assert len(mod.CALL_LOG) == 1
    call = mod.CALL_LOG[0]
    assert call["outer"] == 0.8
    assert call["pupil_grid_size"] == pupil_grid.size

    # And the field is exactly what `generate_pupil(outer=0.8, pupil_grid=pupil_grid)` returns
    expected = hcipy.evaluate_supersampled(hcipy.make_circular_aperture(0.8), pupil_grid, 16)
    np.testing.assert_allclose(np.asarray(result.field), np.asarray(expected), rtol=0, atol=0)


def test_external_pupil_dotted_and_path_module_both_load(pupil_grid, tmp_path):
    """Dotted module names (`hcipy.aperture`) and filesystem paths both resolve."""
    from telescope_sim.apertures.external_pupil import ExternalPupilAperture

    # Dotted module
    a = ExternalPupilAperture(
        module="hcipy",
        function="make_circular_aperture",
        mode="callable",
        kwargs={"diameter": 0.5},
    ).build(pupil_grid)

    # Filesystem path
    mod_path = tmp_path / "_test_circular.py"
    mod_path.write_text(
        textwrap.dedent("""
            import hcipy

            def my_circular(diameter):
                return hcipy.make_circular_aperture(diameter)
        """)
    )
    b = ExternalPupilAperture(
        module=str(mod_path),
        function="my_circular",
        mode="callable",
        kwargs={"diameter": 0.5},
    ).build(pupil_grid)

    np.testing.assert_allclose(np.asarray(a.field), np.asarray(b.field), rtol=0, atol=0)


# --- validation ------------------------------------------------------------


def test_external_pupil_rejects_unknown_mode():
    from telescope_sim.apertures.external_pupil import ExternalPupilAperture

    with pytest.raises(ValueError, match="mode"):
        ExternalPupilAperture(module="hcipy", function="make_circular_aperture", mode="bogus")


def test_external_pupil_requires_module():
    from telescope_sim.apertures.external_pupil import ExternalPupilAperture

    with pytest.raises(ValueError, match="module"):
        ExternalPupilAperture(module="", function="f", mode="callable")


def test_external_pupil_missing_path_raises(tmp_path):
    from telescope_sim.apertures.external_pupil import ExternalPupilAperture

    bogus = tmp_path / "does_not_exist.py"
    aper = ExternalPupilAperture(module=str(bogus), function="anything", mode="callable", kwargs={})
    with pytest.raises(FileNotFoundError):
        aper.build(hcipy.make_pupil_grid(8, 1.0))


def test_external_pupil_missing_function_raises(pupil_grid, tmp_path):
    from telescope_sim.apertures.external_pupil import ExternalPupilAperture

    mod_path = tmp_path / "_empty.py"
    mod_path.write_text("x = 1\n")
    aper = ExternalPupilAperture(module=str(mod_path), function="nonexistent", mode="callable")
    with pytest.raises(AttributeError, match="nonexistent"):
        aper.build(pupil_grid)


def test_external_pupil_aperture_result_carries_metadata(pupil_grid):
    """ApertureResult stores source provenance for debugging downstream issues."""
    from telescope_sim.apertures.external_pupil import ExternalPupilAperture

    aper = ExternalPupilAperture(
        module="hcipy",
        function="make_circular_aperture",
        mode="callable",
        kwargs={"diameter": 0.7},
        area=12.5,
    )
    result = aper.build(pupil_grid)
    assert result.metadata["source_module"] == "hcipy"
    assert result.metadata["source_function"] == "make_circular_aperture"
    assert result.metadata["mode"] == "callable"
    assert result.area == 12.5
    assert result.segments is None
    assert result.segment_coords is None
