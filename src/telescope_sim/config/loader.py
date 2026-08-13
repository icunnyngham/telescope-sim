"""YAML loader — parses a config file, validates it, and builds the pipeline.

Resolves stage types via :mod:`telescope_sim.registry`, instantiates the
concrete classes, and assembles a :class:`telescope_sim.pipeline.TelescopeSim`.

Triggering imports of the standard implementations as a side effect of
importing this module ensures the registry is populated before any
config is resolved.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import hcipy
import yaml

# Side-effect imports: populate the registry with all stock implementations
import telescope_sim.apertures.external_pupil  # noqa: F401
import telescope_sim.apertures.segmented_circular  # noqa: F401
import telescope_sim.coronagraphs  # noqa: F401
import telescope_sim.correctors.actuator_grid  # noqa: F401
import telescope_sim.correctors.segmented_ptt  # noqa: F401
import telescope_sim.correctors.zernike  # noqa: F401
import telescope_sim.focal_planes.angular  # noqa: F401
import telescope_sim.focal_planes.physical  # noqa: F401
import telescope_sim.outputs.fiber_dual  # noqa: F401
import telescope_sim.outputs.intensity  # noqa: F401
import telescope_sim.post  # noqa: F401  (post-processor package imports its modules)
from telescope_sim.abc import (
    Aperture,
    Coronagraph,
    Corrector,
    FocalPlane,
    OutputTap,
    PostProcessor,
)
from telescope_sim.config.schema import SimConfig, StageConfig
from telescope_sim.pipeline import (
    TelescopeSim,
    _mirror_of,
    _OutputSpec,
    _PipelineComponents,
)
from telescope_sim.registry import lookup
from telescope_sim.strehl import StrehlEstimator, build_strehl_estimator


def _lookup_for_backend(kind: str, name: str, backend: str) -> type:
    """Backend-aware registry lookup + supported-backends gate.

    Backend-specific registrations shadow agnostic ones. An agnostic
    implementation may declare a ``supported_backends`` class attribute
    (iterable of backend names) to opt out of backends it cannot serve;
    resolving it for an unsupported backend is a config error.
    """
    cls = lookup(kind, name, backend=backend)
    supported = getattr(cls, "supported_backends", None)
    if supported is not None and backend not in supported:
        raise ValueError(
            f"{kind}/{name} is not supported on the {backend!r} backend "
            f"(supports: {sorted(supported)})."
        )
    return cls


def _instantiate(kind: str, cfg: StageConfig | dict | str, *, backend: str, **extras: Any) -> Any:
    """Resolve and construct a registered implementation from a stage config."""
    if isinstance(cfg, str):
        cls = _lookup_for_backend(kind, cfg, backend)
        return cls(**extras)
    payload = cfg.model_dump() if isinstance(cfg, StageConfig) else dict(cfg)
    type_name = payload.pop("type")
    cls = _lookup_for_backend(kind, type_name, backend)
    return cls(**payload, **extras)


def load_yaml(path: str | Path) -> SimConfig:
    """Load and validate a YAML configuration file."""
    with open(path) as f:
        data = yaml.safe_load(f)
    return SimConfig.model_validate(data)


def build(config: SimConfig, *, backend: str | None = None) -> TelescopeSim:  # noqa: PLR0912,PLR0915  (config-driven assembly: per-stage branches)
    """Instantiate a TelescopeSim from a validated config.

    ``backend`` overrides the config's ``backend`` field when given
    (useful for running one YAML against both compute backends).
    """
    backend = backend or config.backend
    if config.precision != "float64" and backend != "jax":
        raise ValueError(
            f"precision={config.precision!r} requires backend='jax'; the "
            "hcipy backend computes in float64 only."
        )
    if backend == "jax":
        # Populate the jax backend-registry overlay (and fail with an
        # actionable message when the optional dependency is missing).
        try:
            import telescope_sim.backends.jax  # noqa: F401, PLC0415
        except ImportError as exc:
            raise ImportError(
                "backend='jax' requires the optional JAX dependency; "
                "install with: pip install 'telescope-sim[jax]'"
            ) from exc

    # 1) Pupil grid
    pupil_grid = hcipy.make_pupil_grid(config.pupil.resolution, config.pupil.extent)

    # 2) Aperture
    aperture_impl: Aperture = _instantiate("aperture", config.aperture, backend=backend)
    aperture_result = aperture_impl.build(pupil_grid)

    # 3) Correctors
    correctors_by_name: dict[str, Corrector] = {}
    for name, corr_cfg in config.correctors.items():
        # Strip the role-related fields from the payload before passing the
        # remainder to the implementation constructor.
        payload = corr_cfg.model_dump()
        role_kwargs = {
            "wavefront_role": payload.pop("wavefront_role"),
            "target_strategy": payload.pop("target_strategy"),
            "fit_source": payload.pop("fit_source"),
            "target": payload.pop("target"),
        }
        type_name = payload.pop("type")
        cls = _lookup_for_backend("corrector", type_name, backend)

        # Some correctors need the aperture's segments/segment_coords at
        # construction time; supply them by introspection here. Others just
        # need the pupil grid (via _bind_pupil_grid). A more general
        # dependency-injection scheme can come later if needed.
        if type_name == "segmented_ptt":
            corr = cls(
                segments=aperture_result.segments,
                segment_coords=aperture_result.segment_coords,
                name=name,
                **role_kwargs,
                **payload,
            )
        else:
            corr = cls(name=name, **role_kwargs, **payload)
        # Correctors that need a pupil-grid-aware setup register it here.
        if hasattr(corr, "_bind_pupil_grid"):
            corr._bind_pupil_grid(pupil_grid, aperture_result.field)
        correctors_by_name[name] = corr

    # Honor the explicit chain order if given; otherwise iterate config order.
    chain_names = config.corrector_chain or list(correctors_by_name)
    missing = [n for n in chain_names if n not in correctors_by_name]
    if missing:
        raise ValueError(
            f"corrector_chain references unknown correctors: {missing}; "
            f"defined: {list(correctors_by_name)}"
        )
    corrector_chain = [correctors_by_name[n] for n in chain_names]

    # The jax backend composes correctors by summing their mirror-surface
    # OPD contributions (thin phase screens commute), so every corrector in
    # the chain must expose a DM-like surface. Correctors applying arbitrary
    # wavefront transforms are hcipy-only.
    if backend == "jax":
        non_opd = [c.name for c in corrector_chain if _mirror_of(c) is None]
        if non_opd:
            raise ValueError(
                f"correctors {non_opd} expose no mirror surface and cannot "
                "run on the 'jax' backend (which composes correctors as "
                "summed pupil-plane OPD)."
            )

    # 4) Coronagraph (optional). Built before the focal planes so that the
    #    reference-PSF generation can deliberately bypass it.
    coronagraph: Coronagraph | None = None
    if config.coronagraph is not None:
        coro_payload = config.coronagraph.model_dump()
        coro_type = coro_payload.pop("type")
        coro_cls = _lookup_for_backend("coronagraph", coro_type, backend)
        coronagraph = coro_cls(**coro_payload)
        if hasattr(coronagraph, "_bind_pupil_grid"):
            coronagraph._bind_pupil_grid(pupil_grid)

    # 5) Focal planes — built fresh; correctors are at-rest (zero) here
    #    because they were just constructed and nothing has called
    #    set_actuators() yet. Reference PSFs are computed without the
    #    coronagraph (matches legacy convention).
    focal_planes: dict[str, FocalPlane] = {}
    strehl_estimators: dict[str, StrehlEstimator] = {}
    for fp_name, fp_cfg in config.focal_planes.items():
        payload = fp_cfg.model_dump()
        type_name = payload.pop("type")
        cls = _lookup_for_backend("focal_plane", type_name, backend)
        fp = cls(name=fp_name, **payload)
        if backend == "jax":
            fp._precision = config.precision
        fp.build(pupil_grid, aperture_result.field)
        for c in corrector_chain:
            c.flatten()
        fp.compute_reference_psf(corrector_chain)
        focal_planes[fp_name] = fp
        # Build the Strehl estimator now that the at-rest reference PSF
        # exists. argmax/mask/weighted sums are cached here so the
        # per-sample path stays cheap (mirrors legacy lam_setup cache).
        if fp.reference_psf is not None:
            strehl_estimators[fp_name] = build_strehl_estimator(
                method=config.strehl_method,
                reference_psf=fp.reference_psf,
                focal_grid=fp.lam_setup.focal_grid,
                core_radius_rad=config.strehl_core_rad,
            )

    # 5) Outputs (tap + ordered post-processors)
    outputs: list[_OutputSpec] = []
    for out_name, out_cfg in config.outputs.items():
        # Tap
        tap_payload = out_cfg.tap.model_dump()
        tap_type = tap_payload.pop("type")
        tap_cls = _lookup_for_backend("output_tap", tap_type, backend)
        # The intensity / fiber_dual taps take focal_plane_names from the
        # YAML's `focal_planes` field.
        if "focal_planes" in tap_payload:
            tap_payload["focal_plane_names"] = tap_payload.pop("focal_planes")
        tap: OutputTap = tap_cls(name=out_name, **tap_payload)

        # Post-processors
        pps: list[PostProcessor] = []
        for pp_cfg in out_cfg.post_processing:
            pps.append(_instantiate("post_processor", pp_cfg, backend=backend))

        # Track which focal planes this output references (for ctx-building)
        fp_names = list(
            getattr(tap, "focal_plane_names", [])
            or ([tap.source] if isinstance(tap.source, str) else [])
        )

        # Loader-bound dependencies: walk post-processors and inject
        # focal-plane / aperture info into any that implement the
        # ``_bind_loader_dependencies`` hook. Used by ``noisy_detector``
        # (needs focal_grid + aperture_area; eagerly builds the HCIPy
        # NoisyDetector so flat_field RNG-burn happens BEFORE any
        # user-level np.random.seed reset) and ``convolve_image`` (needs
        # the focal plane's reference_psf_sum for kernel normalization).
        for pp in pps:
            if hasattr(pp, "_bind_loader_dependencies"):
                pp._bind_loader_dependencies(
                    aperture_result=aperture_result,
                    focal_planes=focal_planes,
                    focal_plane_names=fp_names,
                )

        outputs.append(
            _OutputSpec(name=out_name, tap=tap, post_processors=pps, focal_plane_names=fp_names)
        )

    components = _PipelineComponents(
        pupil_grid=pupil_grid,
        aperture=aperture_result,
        correctors=corrector_chain,
        focal_planes=focal_planes,
        outputs=outputs,
        strehl_estimators=strehl_estimators,
        coronagraph=coronagraph,
        backend=backend,
        precision=config.precision,
    )
    return TelescopeSim.from_components(components)


def build_from_yaml(path: str | Path, *, backend: str | None = None) -> TelescopeSim:
    """Convenience: load a YAML file and build the pipeline in one step."""
    return build(load_yaml(path), backend=backend)


def build_from_preset(name: str, *, backend: str | None = None) -> TelescopeSim:
    """Resolve a preset name to a packaged YAML and build the pipeline."""
    preset_path = Path(__file__).resolve().parent.parent / "presets" / f"{name}.yaml"
    if not preset_path.is_file():
        # List available presets for the error message
        presets_dir = preset_path.parent
        available = sorted(p.stem for p in presets_dir.glob("*.yaml"))
        raise ValueError(f"unknown preset {name!r}; available: {available or '(none)'}")
    return build_from_yaml(preset_path, backend=backend)


__all__ = ["load_yaml", "build", "build_from_yaml", "build_from_preset"]
