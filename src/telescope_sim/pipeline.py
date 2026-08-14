"""Pipeline orchestrator — holds the optical chain and runs `sample()`.

This module exposes the top-level :class:`TelescopeSim` class, the entry
point for constructing a simulation from a YAML config, a preset, or a
validated pydantic config object.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import hcipy
import numpy as np
from numpy.typing import ArrayLike, NDArray

from telescope_sim.abc import (
    ApertureResult,
    Coronagraph,
    Corrector,
    FocalPlane,
    OutputTap,
    PipelineContext,
    PostProcessor,
)
from telescope_sim.focal_planes.physical import FocalPlaneResult
from telescope_sim.strehl import StrehlEstimator


def _mirror_of(corrector: Any) -> Any | None:
    """Return the HCIPy DM-like object backing a corrector, or None.

    Looks for ``_dm`` (``hcipy.DeformableMirror``, used by Zernike +
    custom basis correctors) then ``_sm``
    (``hcipy.SegmentedDeformableMirror``, used by ``SegmentedPTTCorrector``).
    Both expose ``.surface`` in meters of surface displacement. Returns
    ``None`` for correctors that don't back onto a DM, so the pipeline
    skips them in OPD bookkeeping.
    """
    return getattr(corrector, "_dm", None) or getattr(corrector, "_sm", None)


@dataclass
class _OutputSpec:
    """Per-output configuration assembled from the YAML schema."""

    name: str
    tap: OutputTap
    post_processors: list[PostProcessor] = field(default_factory=list)
    focal_plane_names: list[str] = field(default_factory=list)


@dataclass
class _PipelineComponents:
    """Resolved objects of a built pipeline. Held by :class:`TelescopeSim`."""

    pupil_grid: Any
    aperture: ApertureResult
    correctors: list[Corrector]
    focal_planes: dict[str, FocalPlane]
    outputs: list[_OutputSpec]
    strehl_estimators: dict[str, StrehlEstimator] = field(default_factory=dict)
    coronagraph: Coronagraph | None = None
    backend: str = "hcipy"
    precision: str = "float64"


def _stack_sample_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Stack per-sample ``sample()`` result dicts along a new batch axis."""
    out: dict[str, Any] = {
        "images": {k: np.stack([r["images"][k] for r in results]) for k in results[0]["images"]},
        "actuations": {
            k: np.stack([r["actuations"][k] for r in results]) for k in results[0]["actuations"]
        },
    }
    if "strehls" in results[0]:
        out["strehls"] = {
            k: np.array([r["strehls"][k] for r in results]) for k in results[0]["strehls"]
        }
    return out


class TelescopeSim:
    """Composable telescope-PSF simulator.

    Construct via one of the classmethods (not via ``__init__`` directly):

    - :meth:`from_preset` — load a packaged preset by name
    - :meth:`from_yaml`   — load a user-supplied YAML config
    - :meth:`from_components` — instantiate from already-resolved components
      (primarily for the pipeline's own internal use and for advanced users
      bypassing the YAML/pydantic layer)
    """

    def __init__(self, components: _PipelineComponents) -> None:
        self._c = components
        # Lazily-built jax-backend artifacts (see forward_fn / sample_batch).
        self._forward: Any = None
        self._batched_forward: Any = None
        self._batch_post_programs: list[Any] | None = None
        self._batch_post_fns: dict[Any, Any] = {}

    # --- Construction entry points -----------------------------------------

    @classmethod
    def from_components(cls, components: _PipelineComponents) -> TelescopeSim:
        """Instantiate from already-resolved pipeline components."""
        return cls(components)

    @classmethod
    def from_yaml(cls, path: str | Path, *, backend: str | None = None) -> TelescopeSim:
        """Load a configuration YAML and build the pipeline.

        ``backend`` overrides the YAML's ``backend`` field (``"hcipy"`` /
        ``"jax"``) so one config can be run against either compute backend.
        """
        # Deferred to avoid the loader → pipeline import cycle.
        from telescope_sim.config.loader import build_from_yaml  # noqa: PLC0415

        return build_from_yaml(path, backend=backend)

    @classmethod
    def from_preset(cls, name: str, *, backend: str | None = None) -> TelescopeSim:
        """Load a packaged preset by name."""
        # Deferred to avoid the loader → pipeline import cycle.
        from telescope_sim.config.loader import build_from_preset  # noqa: PLC0415

        return build_from_preset(name, backend=backend)

    # --- Convenience accessors ---------------------------------------------

    @property
    def correctors(self) -> dict[str, Corrector]:
        return {c.name: c for c in self._c.correctors}

    @property
    def focal_planes(self) -> dict[str, FocalPlane]:
        return self._c.focal_planes

    @property
    def aperture(self) -> ApertureResult:
        return self._c.aperture

    @property
    def coronagraph(self) -> Coronagraph | None:
        return self._c.coronagraph

    # --- Main entry point --------------------------------------------------

    def sample(  # noqa: PLR0912,PLR0915  (orchestration: branches + statements reflect chain stages)
        self,
        actuations: Mapping[str, ArrayLike] | None = None,
        *,
        atmos: Any = None,
        output_overrides: Mapping[str, Mapping[str, Any]] | None = None,
        meas_strehl: bool = False,
        meas_pupil_opd: bool = False,
    ) -> dict[str, Any]:
        """Run the optical chain and return a dict of outputs.

        Parameters
        ----------
        actuations
            Per-corrector actuator state. Keys are corrector names (matching
            those declared in the config); each value is whatever shape that
            corrector's ``set_actuators`` accepts.
        atmos
            Per-sample external atmosphere. Any callable taking a
            ``hcipy.Wavefront`` and returning a modified ``hcipy.Wavefront``
            (typically a ``hcipy.InfiniteAtmosphericLayer`` or
            ``hcipy.MultiLayerAtmosphere``, but any wf→wf callable works).
            Applied at the front of the chain, before any corrector. The
            caller owns time evolution — v2 holds no atmosphere state.

            If the object also exposes ``.phase_for(lam)`` returning
            HCIPy-convention phase = 2π·OPD/lam, the atmosphere's OPD is
            seeded into the per-corrector cumulative-OPD stream, so fit-role
            correctors with ``fit_source="cumulative_phase_pre_self"`` will
            naturally fit to (and cancel) the atmosphere. Without
            ``.phase_for`` the wavefront is still modified, but fit-role
            correctors only see corrector-chain OPD.

            The reference PSF is never atmospheric — it's cached once at
            sim-build time with ``atmos=None`` implicit.
        output_overrides
            Per-sample tap-config overrides, keyed by output name. For example,
            ``{"psf": {"int_phot_flux": 5.0e7}}`` to vary the photon flux on a
            noisy-intensity tap from sample to sample. Taps that have no
            per-sample state ignore the override.
        meas_strehl
            If True, includes a ``strehls`` entry in the returned dict.
        meas_pupil_opd
            If True, includes a ``pupil_opd`` entry in the returned dict:
            the cumulative pupil-plane OPD seen at the back of the chain,
            as an ``hcipy.Field`` on the simulator's pupil grid in meters.
            Sums atmosphere (when ``atmos.phase_for`` is available) +
            every DM-backed corrector's surface × 2. Companion to
            ``sim.aperture.field`` for masked display via
            ``hcipy.imshow_field(out["pupil_opd"], mask=sim.aperture.field)``.

        Returns
        -------
        dict
            ``images``       — dict of output_name → numpy array
            ``actuations``   — dict of corrector_name → numpy array
                              (only for correctors with ``target=True``)
            ``strehls``      — present iff ``meas_strehl`` is True
            ``pupil_opd``    — present iff ``meas_pupil_opd`` is True;
                              ``hcipy.Field`` of cumulative pupil-plane OPD
                              in meters
        """
        actuations = dict(actuations or {})
        output_overrides = dict(output_overrides or {})

        # 1-2) Apply actuator state, resolve fit-role correctors, and
        #      accumulate the cumulative-OPD stream (shared with the batch
        #      paths via _apply_chain_state).
        cum_opd_pre, running_opd = self._apply_chain_state(actuations, atmos)

        # 3) Propagate each focal plane and collect FocalPlaneResult objects
        #    (each holds both summed intensity and per-wavelength wavefronts
        #    so downstream taps can pick what they need). Atmosphere applies
        #    at the front of the per-λ loop, before any corrector.
        fp_results: dict[str, Any] = {}
        for name, fp in self._c.focal_planes.items():
            fp_results[name] = fp._propagate_chain(
                self._c.correctors, coronagraph=self._c.coronagraph, atmos=atmos
            )

        # 4-6) Taps + post-processing, actuation echo, Strehl — shared with
        #      sample_batch() via _finalize_outputs.
        result = self._finalize_outputs(
            fp_results,
            output_overrides=output_overrides,
            meas_strehl=meas_strehl,
            cum_opd_pre=cum_opd_pre,
        )

        # 7) Pupil-plane OPD readback. The cumulative-OPD stream computed
        #    in step 2 already includes the atmosphere seed (when it
        #    exposes phase_for) and every DM-backed corrector's surface×2,
        #    so we just wrap it as an hcipy.Field for masked display.
        if meas_pupil_opd:
            result["pupil_opd"] = hcipy.Field(running_opd.copy(), self._c.pupil_grid)

        return result

    def _apply_chain_state(  # noqa: PLR0912  (chain orchestration: branches reflect roles/sources)
        self,
        actuations: Mapping[str, ArrayLike],
        atmos: Any = None,
        *,
        need_opd_stream: bool = True,
    ) -> tuple[list[NDArray[np.float64]] | None, NDArray[np.float64] | None]:
        """Steps 1-2 of ``sample()``: actuator state + cumulative OPD.

        Applies caller actuations to actuate/impose correctors, walks the
        chain accumulating per-corrector cumulative pupil-plane OPD
        (= 2 × surface) "just before this corrector", and resolves
        fit-role correctors against it as it goes (convention:
        ``fit_surface`` returns *matching* actuator values; the apply
        site negates). Atmosphere seeds the running OPD when it exposes
        ``.phase_for``; otherwise the OPD stream stays at zero and
        fit-role correctors won't see the atmosphere.

        Returns ``(cum_opd_pre, running_opd)``. With
        ``need_opd_stream=False`` and no fit-role correctors in the
        chain, the surface walk is skipped entirely and ``(None, None)``
        is returned — a fast path for batch loops that only need the
        actuator state applied.
        """
        # 1) Apply actuator state to "actuate" and "impose" correctors.
        for c in self._c.correctors:
            if c.wavefront_role in {"actuate", "impose"}:
                values = actuations.get(c.name)
                if values is None:
                    c.flatten()
                else:
                    c.set_actuators(values)
            # "fit" correctors are resolved below

        if not need_opd_stream and not any(c.wavefront_role == "fit" for c in self._c.correctors):
            return None, None

        # 2) Chain walk with fit resolution and cumulative-OPD snapshots
        #    (used by step 5's residual-fit target strategies).
        correctors_by_name = {c.name: c for c in self._c.correctors}
        running_opd = np.zeros(self._c.pupil_grid.size, dtype=np.float64)
        if atmos is not None and hasattr(atmos, "phase_for"):
            running_opd = running_opd + np.asarray(atmos.phase_for(1.0)) / (2.0 * np.pi)
        cum_opd_pre: list[NDArray[np.float64]] = []
        seen_names: set[str] = set()

        for c in self._c.correctors:
            cum_opd_pre.append(running_opd.copy())

            if c.wavefront_role == "fit":
                fs = c.fit_source
                if fs is None or fs == "cumulative_phase_pre_self":
                    phase_in = cum_opd_pre[-1]
                elif fs in correctors_by_name and fs in seen_names:
                    other = correctors_by_name[fs]
                    other_mirror = _mirror_of(other)
                    if other_mirror is None:
                        raise ValueError(
                            f"corrector {c.name!r} fit_source={fs!r} "
                            "refers to a corrector with no DM surface to "
                            "fit to."
                        )
                    phase_in = 2.0 * np.asarray(other_mirror.surface)
                elif fs in correctors_by_name:
                    raise ValueError(
                        f"corrector {c.name!r} has wavefront_role='fit' with "
                        f"fit_source={fs!r}, but that corrector appears "
                        "later in the chain. fit_source must reference an "
                        "earlier corrector or use "
                        "'cumulative_phase_pre_self'."
                    )
                else:
                    raise ValueError(
                        f"corrector {c.name!r} has unknown fit_source={fs!r}. "
                        "Use 'cumulative_phase_pre_self' or the name of an "
                        "earlier corrector in the chain."
                    )
                fit_values = np.asarray(c.fit_surface(phase_in))
                c.set_actuators(-fit_values)

            # After this corrector's state is finalized, add its OPD
            # contribution to the running cumulative. Any corrector
            # exposing an HCIPy DM-like object via ``_dm`` (regular DM)
            # or ``_sm`` (segmented DM) contributes here.
            mirror = _mirror_of(c)
            if mirror is not None:
                running_opd = running_opd + 2.0 * np.asarray(mirror.surface)

            seen_names.add(c.name)

        return cum_opd_pre, running_opd

    def _finalize_outputs(
        self,
        fp_results: dict[str, Any],
        *,
        output_overrides: Mapping[str, Mapping[str, Any]],
        meas_strehl: bool,
        cum_opd_pre: list[NDArray[np.float64]] | None,
    ) -> dict[str, Any]:
        """Steps 4-6 of ``sample()``: taps + post, actuation echo, Strehl.

        Factored out so ``sample_batch()`` can run them per batch element
        against device-batched propagation results with semantics identical
        to ``sample()``. ``cum_opd_pre`` (per-corrector cumulative-OPD
        snapshots from step 2) may be None when no target corrector uses a
        residual-fit strategy.
        """
        # 4) Run output taps + per-output post-processors.
        images: dict[str, NDArray] = {}
        for out_spec in self._c.outputs:
            arr = out_spec.tap.extract(fp_results, overrides=output_overrides.get(out_spec.name))

            # Build context for post-processors
            ref_peaks = [
                self._c.focal_planes[n].reference_peak_intensity for n in out_spec.focal_plane_names
            ]
            ref_sums = [
                self._c.focal_planes[n].reference_psf_sum for n in out_spec.focal_plane_names
            ]
            ctx = PipelineContext(
                output_name=out_spec.name,
                focal_plane_name=",".join(out_spec.focal_plane_names),
                reference_peak_intensity=ref_peaks[0] if ref_peaks else None,
                reference_psf_sum=ref_sums[0] if ref_sums else None,
                overrides=dict(output_overrides.get(out_spec.name) or {}),
                extras={
                    "reference_peak_intensities": np.array(ref_peaks, dtype=np.float64),
                    "reference_psf_sums": np.array(ref_sums, dtype=np.float64),
                },
            )
            for pp in out_spec.post_processors:
                arr = pp(arr, ctx)
            images[out_spec.name] = np.asarray(arr)

        # 5) Build the actuation echo / Y output. Residual-fit strategies
        #    use the per-corrector cumulative-OPD snapshot computed in
        #    step 2. Convention: fit_surface returns *matching* values
        #    (the wavefront state in the corrector's basis), so the Y
        #    formulas below are unnegated — Y reports "what is at the
        #    pupil"; the ML model trainer applies -Y downstream to drive
        #    corrections. Matches legacy v1 ``out_actuate = caller +
        #    matching_fit(atmos)``.
        actuator_echo = self._actuation_echo(cum_opd_pre)

        result: dict[str, Any] = {
            "images": images,
            "actuations": actuator_echo,
        }

        # 6) Strehl. Estimators were built once at construction with the
        #    reference-PSF argmax / core mask / weighted sums cached, so
        #    this loop is O(1) (peak) or O(core_pixels) (matched_filter)
        #    per focal plane — same as the legacy `_strehl` call site.
        if meas_strehl:
            strehls: dict[str, float] = {}
            for name in self._c.focal_planes:
                est = self._c.strehl_estimators.get(name)
                if est is None:
                    continue
                strehls[name] = est.compute(fp_results[name].intensity)
            result["strehls"] = strehls

        return result

    def _actuation_echo(self, cum_opd_pre: list[NDArray[np.float64]] | None) -> dict[str, NDArray]:
        """Step 5 of ``sample()``: the actuation echo / Y output.

        Reads the correctors' current actuator state, so callers must have
        applied the sample's actuations first. ``cum_opd_pre`` is required
        only when a target corrector uses a residual-fit strategy.
        """
        actuator_echo: dict[str, NDArray] = {}
        for i, c in enumerate(self._c.correctors):
            if not c.target:
                continue
            if c.target_strategy == "none":
                continue
            if c.target_strategy == "actuators":
                actuator_echo[c.name] = np.asarray(c.actuators)
                continue
            if cum_opd_pre is None:
                raise RuntimeError(
                    f"corrector {c.name!r} uses target_strategy="
                    f"{c.target_strategy!r} but no cumulative-OPD snapshots "
                    "were provided."
                )
            if c.target_strategy == "actuators_plus_residual_fit":
                residual = np.asarray(c.fit_surface(cum_opd_pre[i]))
                actuator_echo[c.name] = np.asarray(c.actuators) + residual
            elif c.target_strategy == "residual_fit_only":
                actuator_echo[c.name] = np.asarray(c.fit_surface(cum_opd_pre[i]))
        return actuator_echo

    # --- Pure forward function + batch sampling (jax backend) ---------------

    def forward_fn(self) -> Any:
        """Pure jittable forward model (jax backend only), built once.

        Returns a :class:`telescope_sim.backends.jax.forward.TelescopeForward`:
        a pure function from caller-facing actuation values to raw summed
        focal-plane intensities, exposing the ``actuations → opd`` and
        ``opd → intensity`` stages separately (the latter is the hook for
        external OPD such as atmosphere screens), plus ``actuation_echo``
        for in-graph training targets. Fit-role correctors are folded into
        the graph by composed-fit probing at build time; note their
        composed response covers *actuation-driven* OPD only — external
        OPD added at the intensity stage bypasses them (use
        :meth:`sample` for atmosphere-reactive fitting). Safe to
        ``jax.jit`` / ``vmap`` / ``grad`` and compose into custom sampling
        strategies; :meth:`sample_batch` is the packaged reference
        composition.

        Raises ``NotImplementedError`` on the hcipy backend.
        """
        if self._c.backend != "jax":
            raise NotImplementedError(
                "forward_fn() requires backend='jax'; the hcipy backend has "
                "no pure-function propagation path."
            )
        if self._forward is None:
            # Deferred so the base package works without the [jax] extra.
            from telescope_sim.backends.jax.forward import build_forward  # noqa: PLC0415

            self._forward = build_forward(self._c)
        return self._forward

    def sample_batch(
        self,
        actuations: Mapping[str, ArrayLike] | None = None,
        *,
        output_overrides: Mapping[str, Mapping[str, Any]] | None = None,
        meas_strehl: bool = False,
        batch_size: int | None = None,
        key: Any = None,
    ) -> dict[str, Any]:
        """Reference iid batch sampler: stacked ``sample()`` semantics.

        Every array in ``actuations`` carries a leading batch dimension on
        top of what :meth:`sample` accepts for that corrector; the returned
        dict has the shape of a :meth:`sample` result with a leading batch
        axis on every image, actuation echo, and Strehl value.

        On the jax backend, propagation for the whole batch runs as one
        jitted, vmapped device dispatch over :meth:`forward_fn`; output
        taps, post-processors (including detector noise, which keeps its
        host-side numpy RNG), actuation echoes, and Strehl run per sample
        through the same code path ``sample()`` uses. On the hcipy backend
        the same semantics come from a plain Python loop over
        :meth:`sample`.

        This is deliberately a *reference composition* — curriculum,
        temporal-sequence, or RL samplers should compose
        :meth:`forward_fn` directly in user code instead of extending this
        method. Note JAX re-jits per batch shape: prefer a fixed batch
        size (or pad) inside loops.

        ``batch_size`` is only required when ``actuations`` is empty (an
        at-rest batch); otherwise it must agree with the arrays' leading
        dimension. Fit-role corrector chains are supported (their state is
        folded into the forward graph; on the host-post path they are
        additionally resolved per sample for the echo readback).

        ``key`` (jax backend only; an int seed or a JAX PRNG key) opts
        into **on-device post-processing**: each output's tap + post
        chain — including detector noise on JAX PRNG streams — runs
        inside the batched device dispatch, so training data never
        round-trips through host-side numpy post. Noisy outputs are then
        reproducible per key *within* the jax backend but deliberately do
        NOT bit-match the host path's numpy draws (the flat-field fixed
        pattern is shared; the random draws are not). Requires every
        output's chain to have an in-graph equivalent (the ``intensity``
        tap, ``noisy_detector``, ``convolve_image``, the norms,
        ``channels_first``) — anything else raises, and dropping ``key=``
        restores host-side post. In key-mode, ``output_overrides`` for
        ``int_phot_flux`` / ``convolve_image`` accept either one value or
        an array with a leading batch dimension, and actuation echoes and
        Strehl are computed on-device from the forward model's composed
        maps and cached estimator constants (fp-level, not bit, parity
        with the host path).
        """
        acts, n_batch = self._validate_batch_actuations(actuations, batch_size)

        if key is not None:
            if self._c.backend != "jax":
                raise NotImplementedError(
                    "sample_batch(key=...) — on-device post-processing — requires backend='jax'."
                )
            return self._sample_batch_on_device(
                acts,
                n_batch,
                output_overrides=dict(output_overrides or {}),
                meas_strehl=meas_strehl,
                key=key,
            )

        if self._c.backend != "jax":
            return _stack_sample_results(
                [
                    self.sample(
                        {k: v[b] for k, v in acts.items()},
                        output_overrides=output_overrides,
                        meas_strehl=meas_strehl,
                    )
                    for b in range(n_batch)
                ]
            )

        return self._sample_batch_host_post(
            acts, n_batch, output_overrides=output_overrides, meas_strehl=meas_strehl
        )

    def _sample_batch_host_post(
        self,
        acts: dict[str, NDArray[np.float64]],
        n_batch: int,
        *,
        output_overrides: Mapping[str, Mapping[str, Any]] | None,
        meas_strehl: bool,
    ) -> dict[str, Any]:
        """Default jax batch path: device propagation, host-side steps 4-6."""
        forward = self.forward_fn()
        if acts:
            if self._batched_forward is None:
                import jax  # noqa: PLC0415

                self._batched_forward = jax.jit(jax.vmap(forward))
            intensities = {
                name: np.asarray(img) for name, img in self._batched_forward(acts).items()
            }
        else:
            # At-rest batch: one propagation, replicated host-side.
            intensities = {
                name: np.broadcast_to(np.asarray(img), (n_batch, *np.shape(img)))
                for name, img in forward({}).items()
            }

        # Per-sample host bookkeeping mirrors sample() steps 1-2: the OPD
        # stream is walked only when fit-role correctors need resolving or
        # a residual-fit echo consumes the snapshots.
        needs_opd = any(
            c.target and c.target_strategy in ("actuators_plus_residual_fit", "residual_fit_only")
            for c in self._c.correctors
        )
        results = []
        for b in range(n_batch):
            cum_opd_pre, _ = self._apply_chain_state(
                {k: v[b] for k, v in acts.items()}, need_opd_stream=needs_opd
            )
            fp_results = {
                name: FocalPlaneResult(intensity=intensities[name][b], wavefronts=[])
                for name in self._c.focal_planes
            }
            results.append(
                self._finalize_outputs(
                    fp_results,
                    output_overrides=dict(output_overrides or {}),
                    meas_strehl=meas_strehl,
                    cum_opd_pre=cum_opd_pre,
                )
            )
        return _stack_sample_results(results)

    def _sample_batch_on_device(
        self,
        acts: dict[str, NDArray[np.float64]],
        n_batch: int,
        *,
        output_overrides: dict[str, Mapping[str, Any]],
        meas_strehl: bool,
        key: Any,
    ) -> dict[str, Any]:
        """Key-mode ``sample_batch``: propagation, tap/post, echoes, and Strehl on device.

        Actuation echoes come from the forward model's precomputed
        composed-fit maps and Strehl from its in-graph estimator
        translations (both fp-level, not bit, parity with the host path;
        custom Strehl estimator objects fall back to a host loop). See
        :meth:`sample_batch` for the key/override semantics and the
        documented determinism fork.
        """
        import jax  # noqa: PLC0415
        import jax.numpy as jnp  # noqa: PLC0415

        from telescope_sim.backends.jax.post import compile_output_program  # noqa: PLC0415

        forward = self.forward_fn()
        dtype = jnp.float32 if self._c.precision == "float32" else jnp.float64
        if self._batch_post_programs is None:
            self._batch_post_programs = [
                compile_output_program(spec, self._c.focal_planes, dtype)
                for spec in self._c.outputs
            ]
        unknown = sorted(set(output_overrides) - {spec.name for spec in self._c.outputs})
        if unknown:
            raise ValueError(f"output_overrides reference unknown output(s): {unknown}")

        if isinstance(key, int):
            key = jax.random.PRNGKey(key)
        sample_keys = jax.random.split(key, n_batch)

        if acts:
            if self._batched_forward is None:
                self._batched_forward = jax.jit(jax.vmap(forward))
            intensities = self._batched_forward(acts)
        else:
            single = forward({})
            intensities = {
                name: jnp.broadcast_to(img, (n_batch, *img.shape)) for name, img in single.items()
            }

        images: dict[str, NDArray] = {}
        for idx, (spec, prog) in enumerate(
            zip(self._c.outputs, self._batch_post_programs, strict=True)
        ):
            override_args = self._coerce_device_overrides(
                prog, spec.name, dict(output_overrides.get(spec.name) or {}), n_batch, dtype
            )
            # Per-output key stream: fold the output index into each
            # per-sample key so outputs never share draws.
            out_keys = jax.vmap(lambda k, i=idx: jax.random.fold_in(k, i))(sample_keys)
            fn = self._batch_post_fns.get(idx)
            if fn is None:
                fn = self._batch_post_fns[idx] = jax.jit(jax.vmap(prog))
            images[spec.name] = np.asarray(fn(intensities, out_keys, override_args))

        result: dict[str, Any] = {
            "images": images,
            "actuations": self._device_echoes(forward, acts, n_batch),
        }

        if meas_strehl:
            result["strehls"] = self._device_strehls(forward, intensities, n_batch)
        return result

    def _device_strehls(
        self, forward: Any, intensities: Mapping[str, Any], n_batch: int
    ) -> dict[str, NDArray]:
        """Batched Strehl for the device path: in-graph estimators where
        available, host loop for custom estimator objects (the stock
        peak / matched_filter methods both translate in-graph)."""
        strehls: dict[str, NDArray] = {}
        device_names = set(forward.strehl_names)
        if device_names:
            strehl_fn = self._batch_post_fns.get("strehl")
            if strehl_fn is None:
                import jax  # noqa: PLC0415

                strehl_fn = self._batch_post_fns["strehl"] = jax.jit(
                    jax.vmap(forward.strehls_from_intensities)
                )
            strehls.update({k: np.asarray(v) for k, v in strehl_fn(intensities).items()})
        for name in self._c.focal_planes:
            est = self._c.strehl_estimators.get(name)
            if est is None or name in device_names:
                continue
            fp_intensities = np.asarray(intensities[name])
            strehls[name] = np.array([est.compute(fp_intensities[b]) for b in range(n_batch)])
        return strehls

    def _device_echoes(
        self, forward: Any, acts: dict[str, NDArray[np.float64]], n_batch: int
    ) -> dict[str, NDArray]:
        """Batched in-graph actuation echoes (see ``TelescopeForward.actuation_echo``)."""
        if not forward.echo_names:
            return {}
        if not acts:
            # At-rest batch: one echo, replicated.
            return {
                k: np.broadcast_to(np.asarray(v), (n_batch, *np.shape(v)))
                for k, v in forward.actuation_echo({}).items()
            }
        echo_fn = self._batch_post_fns.get("echo")
        if echo_fn is None:
            import jax  # noqa: PLC0415

            echo_fn = self._batch_post_fns["echo"] = jax.jit(jax.vmap(forward.actuation_echo))
        return {k: np.asarray(v) for k, v in echo_fn(acts).items()}

    def _coerce_device_overrides(
        self,
        prog: Any,
        out_name: str,
        overrides: dict[str, Any],
        n_batch: int,
        dtype: Any,
    ) -> dict[str, Any]:
        """Turn key-mode override values into leading-batch-dim traced args."""
        import jax.numpy as jnp  # noqa: PLC0415

        from telescope_sim.backends.jax.post import OVERRIDE_SAMPLE_NDIM  # noqa: PLC0415

        args: dict[str, Any] = {}
        for name, value in overrides.items():
            if name not in prog.override_params:
                raise ValueError(
                    f"output {out_name!r}: override {name!r} cannot be applied "
                    "on-device; drop key= to use host-side post-processing."
                )
            arr = jnp.asarray(value, dtype=dtype)
            sample_ndim = OVERRIDE_SAMPLE_NDIM[name]
            if arr.ndim == sample_ndim:
                arr = jnp.broadcast_to(arr, (n_batch, *arr.shape))
            elif not (arr.ndim == sample_ndim + 1 and arr.shape[0] == n_batch):
                raise ValueError(
                    f"output {out_name!r}: override {name!r} must be one value "
                    f"or carry a leading batch dimension of {n_batch}; got "
                    f"shape {np.shape(value)}"
                )
            args[name] = arr
        return args

    def _validate_batch_actuations(
        self,
        actuations: Mapping[str, ArrayLike] | None,
        batch_size: int | None,
    ) -> tuple[dict[str, NDArray[np.float64]], int]:
        """Coerce/validate ``sample_batch`` inputs; returns (arrays, batch size)."""
        acts = {k: np.asarray(v, dtype=np.float64) for k, v in dict(actuations or {}).items()}
        by_name = {c.name: c for c in self._c.correctors}
        unknown = sorted(set(acts) - set(by_name))
        if unknown:
            raise ValueError(
                f"unknown corrector(s) in actuations: {unknown}; defined: {list(by_name)}"
            )
        for name, values in acts.items():
            n_act = int(by_name[name].n_actuators)
            if values.ndim < 2 or int(np.prod(values.shape[1:])) != n_act:
                raise ValueError(
                    f"actuations[{name!r}]: expected a leading batch dimension "
                    f"over per-sample actuations ({n_act} values each), got "
                    f"shape {values.shape}"
                )
        sizes = {v.shape[0] for v in acts.values()}
        if len(sizes) > 1:
            raise ValueError(f"inconsistent batch sizes across actuations: {sorted(sizes)}")
        n_batch = sizes.pop() if sizes else batch_size
        if n_batch is None:
            raise ValueError("empty actuations requires an explicit batch_size")
        if batch_size is not None and batch_size != n_batch:
            raise ValueError(
                f"batch_size={batch_size} disagrees with the actuations' "
                f"leading dimension {n_batch}"
            )
        if int(n_batch) < 1:
            raise ValueError("batch size must be >= 1")
        return acts, int(n_batch)


__all__ = ["TelescopeSim"]
