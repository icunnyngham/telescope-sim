"""Pure JAX forward model: actuations → pupil OPD → focal intensities.

:meth:`telescope_sim.pipeline.TelescopeSim.forward_fn` (jax backend only)
returns a :class:`TelescopeForward` — a pure function from caller-facing
actuation values to raw summed focal-plane intensities, built once from
the sim's components and safe to ``jax.jit`` / ``vmap`` / ``grad`` / scan
over. It is the stable primitive for batch sampling, gradient-based
optimization, and (planned) differentiable-model export; higher-level
sampling strategies compose it in user code
(:meth:`~telescope_sim.pipeline.TelescopeSim.sample_batch` is the
packaged reference composition).

The two stages are exposed separately:

- :meth:`TelescopeForward.opd_from_actuations` — actuations to total
  pupil-plane OPD (meters). Every supported corrector's surface is
  linear in its dm actuators (``surface = T @ dm_actuators`` with ``T``
  the influence-function transformation matrix), and the caller→dm map
  is probed numerically at build time through ``set_actuators``, so each
  actuate/impose corrector collapses to one precomputed
  ``(n_pix, n_actuators)`` matrix. Fit-role correctors are folded in by
  **composed-fit probing**: their (linear) ``fit_surface`` is probed
  through each upstream contribution map at build time, making their
  state and OPD contribution linear in the input actuations too — the
  chain walk mirrors ``sample()`` step 2 exactly, including chained fits
  and named ``fit_source`` references. Correctors with a nonlinear
  ``set_actuators`` are rejected at build time.
- :meth:`TelescopeForward.intensity_from_opd` — OPD to per-focal-plane
  summed intensity, through the same jitted MFT kernels ``sample()``
  uses. This stage is the external-OPD hook — but note that OPD added
  here **bypasses fit-role correctors**: the composed-fit maps respond
  to input actuations only. For atmosphere that fit-role correctors
  should see (and cancel), use ``sample(atmos=...)``.
- :meth:`TelescopeForward.actuation_echo` — the target correctors'
  echo / Y output (state + residual-fit strategies) as precomputed
  linear maps of the same inputs, for fully on-device training targets.

Out of scope, by design: output taps and post-processors (see
``backends/jax/post.py`` for their in-graph batch programs).
``forward_fn`` returns raw physics only.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import jax.numpy as jnp
import numpy as np
from numpy.typing import ArrayLike, NDArray

from telescope_sim.backends.jax.focal_planes import _check_coronagraph
from telescope_sim.pipeline import _mirror_of


def _probe_affine_actuation(corrector: Any) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Extract ``dm_actuators = A @ caller_values + offset`` by probing.

    Pokes unit caller vectors through ``set_actuators`` and reads the
    backing mirror's actuator vector, then verifies the extracted affine
    map against an independent dense probe vector so a nonlinear
    corrector fails loudly instead of silently mismatching ``sample()``.
    The corrector's actuator state is restored on exit.
    """
    mirror = _mirror_of(corrector)
    n = int(corrector.n_actuators)
    saved = np.array(mirror.actuators, dtype=np.float64, copy=True)
    try:
        corrector.set_actuators(np.zeros(n))
        offset = np.array(mirror.actuators, dtype=np.float64, copy=True)
        matrix = np.empty((offset.size, n), dtype=np.float64)
        probe = np.zeros(n)
        for i in range(n):
            probe[i] = 1.0
            corrector.set_actuators(probe)
            matrix[:, i] = np.asarray(mirror.actuators, dtype=np.float64) - offset
            probe[i] = 0.0
        check = np.cos(np.linspace(0.0, 3.0, n))
        corrector.set_actuators(check)
        got = np.asarray(mirror.actuators, dtype=np.float64)
        want = matrix @ check + offset
        scale = max(float(np.abs(want).max()), float(np.abs(got).max()), np.finfo(float).tiny)
        if float(np.abs(got - want).max()) > 1e-9 * scale:
            raise ValueError(
                f"corrector {corrector.name!r}: set_actuators() is not linear in "
                "the caller values, so it cannot be expressed in the pure "
                "forward function. Use sample() for this chain."
            )
    finally:
        mirror.actuators = saved
    return matrix, offset


class TelescopeForward:
    """Pure ``actuations → intensities`` function for one built sim.

    Instances are returned by ``TelescopeSim.forward_fn()`` — see the
    module docstring for the contract. All methods are pure functions of
    their arguments over internally-held constant arrays: jit, vmap,
    grad, and scan compose freely. Actuation dicts are ordinary JAX
    pytrees.

    Attributes
    ----------
    corrector_names
        Actuatable corrector names, in chain order.
    n_actuators
        Per-corrector flat actuation length (caller-facing, matching
        ``set_actuators``).
    focal_plane_names
        Names of the focal planes ``intensity_from_opd`` returns.
    """

    def __init__(
        self,
        *,
        maps: dict[str, jnp.ndarray],
        opd_offset: jnp.ndarray,
        planes: dict[str, tuple[Any, jnp.ndarray]],
        n_actuators: dict[str, int],
        dtype: Any,
        echoes: dict[str, tuple[dict[str, jnp.ndarray], jnp.ndarray, tuple[int, ...]]]
        | None = None,
    ) -> None:
        self._maps = maps
        self._opd_offset = opd_offset
        self._planes = planes
        self._n_actuators = dict(n_actuators)
        self._dtype = dtype
        self._echoes = dict(echoes or {})

    @property
    def corrector_names(self) -> tuple[str, ...]:
        return tuple(self._maps)

    @property
    def n_actuators(self) -> dict[str, int]:
        return dict(self._n_actuators)

    @property
    def focal_plane_names(self) -> tuple[str, ...]:
        return tuple(self._planes)

    @property
    def echo_names(self) -> tuple[str, ...]:
        """Target correctors whose actuation echo is computable in-graph."""
        return tuple(self._echoes)

    def opd_from_actuations(self, actuations: Mapping[str, ArrayLike] | None = None) -> jnp.ndarray:
        """Total pupil-plane OPD (meters, flat) for caller actuation values.

        Keys are corrector names; values are anything ``set_actuators``
        accepts for that corrector (any shape raveling to its flat
        actuation length). Missing correctors are flat — the same
        convention as ``sample()``. Unknown keys and wrong sizes raise at
        trace time.
        """
        acts = self._validated(actuations)
        opd = self._opd_offset
        for name, matrix in self._maps.items():
            if name in acts:
                opd = opd + matrix @ acts[name]
        return opd

    def intensity_from_opd(self, opd: ArrayLike) -> dict[str, jnp.ndarray]:
        """Per-focal-plane summed intensity for a pupil-plane OPD (meters).

        The external-OPD hook: atmosphere screens, batch-generated
        turbulence, or any extra pupil OPD is added to
        ``opd_from_actuations(...)`` before this call. Returns raw
        intensities — no taps, no post-processing.
        """
        opd = jnp.asarray(opd, dtype=self._dtype)
        out: dict[str, jnp.ndarray] = {}
        for name, (mft, amplitude) in self._planes.items():
            out[name] = mft._summed_intensity(amplitude, opd.reshape(mft.pupil_shape))
        return out

    def __call__(self, actuations: Mapping[str, ArrayLike] | None = None) -> dict[str, jnp.ndarray]:
        """``intensity_from_opd(opd_from_actuations(actuations))``."""
        return self.intensity_from_opd(self.opd_from_actuations(actuations))

    def actuation_echo(
        self, actuations: Mapping[str, ArrayLike] | None = None
    ) -> dict[str, jnp.ndarray]:
        """In-graph actuation echo / Y output for the target correctors.

        Pure and jit/vmap-compatible, mirroring ``sample()``'s echo
        strategies: every target corrector's echo (its caller-facing
        state — including fit-role state resolved by the composed-fit
        maps — plus any residual-fit term) is a precomputed linear map of
        the input actuations. Returns canonical per-corrector echo shapes
        (the same shapes ``sample()['actuations']`` carries). Matches the
        host echo at floating-point (not bit) level.
        """
        acts = self._validated(actuations)
        out: dict[str, jnp.ndarray] = {}
        for name, (maps, const, shape) in self._echoes.items():
            value = const
            for input_name, matrix in maps.items():
                if input_name in acts:
                    value = value + matrix @ acts[input_name]
            out[name] = value.reshape(shape)
        return out

    def _validated(self, actuations: Mapping[str, ArrayLike] | None) -> dict[str, jnp.ndarray]:
        """Coerce an actuations mapping to validated flat traced vectors."""
        acts = dict(actuations or {})
        unknown = sorted(set(acts) - set(self._maps))
        if unknown:
            raise ValueError(
                f"unknown corrector(s) in actuations: {unknown}; defined: {list(self._maps)}"
            )
        flat: dict[str, jnp.ndarray] = {}
        for name, value in acts.items():
            vec = jnp.asarray(value, dtype=self._dtype).reshape(-1)
            if vec.shape[0] != self._n_actuators[name]:
                raise ValueError(
                    f"actuations[{name!r}]: expected {self._n_actuators[name]} "
                    f"values per sample, got shape {np.shape(value)}"
                )
            flat[name] = vec
        return flat


def _probe_fit_columns(corrector: Any, phase_matrix: np.ndarray) -> np.ndarray:
    """Compose a corrector's ``fit_surface`` with a linear phase map.

    ``phase_matrix`` maps some input vector to pupil-plane OPD
    ``(n_pix, n_in)``; the result maps that input to the corrector's
    flat caller-facing fit values ``(n_flat, n_in)`` — one host
    least-squares call per column. ``fit_surface`` is linear in the
    phase (pinned by the fit-contract suite), which is what makes this
    composition exact.
    """
    columns = [
        np.asarray(corrector.fit_surface(phase_matrix[:, i]), dtype=np.float64).ravel()
        for i in range(phase_matrix.shape[1])
    ]
    return np.column_stack(columns) if columns else np.zeros((int(corrector.n_actuators), 0))


def _fit_of_offset(corrector: Any, offset: np.ndarray) -> np.ndarray:
    if offset.any():
        return np.asarray(corrector.fit_surface(offset), dtype=np.float64).ravel()
    return np.zeros(int(corrector.n_actuators), dtype=np.float64)


def _merge_maps(left: dict[str, np.ndarray], right: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    out = dict(left)
    for k, v in right.items():
        out[k] = out[k] + v if k in out else v
    return out


def build_forward(components: Any) -> TelescopeForward:  # noqa: PLR0912,PLR0915  (single chain walk; branches mirror sample() step 2)
    """Build a :class:`TelescopeForward` from resolved pipeline components.

    Walks the corrector chain in order, mirroring ``sample()`` step 2's
    cumulative-OPD bookkeeping symbolically: every corrector's OPD
    contribution is a linear(+constant) map from the *input* (actuate /
    impose) actuations. Fit-role correctors are resolved by
    **composed-fit probing** — their host-side ``fit_surface`` is probed
    through each upstream contribution map (one least-squares call per
    upstream actuator), so their state, OPD contribution, and any
    residual-fit echoes all collapse to precomputed matrices with no
    per-kind code.
    """
    if components.backend != "jax":
        raise NotImplementedError(
            "forward_fn() requires backend='jax'; the hcipy backend has no "
            "pure-function propagation path."
        )
    _check_coronagraph(components.coronagraph)

    dtype = jnp.float32 if components.precision == "float32" else jnp.float64
    n_pix = int(components.pupil_grid.size)
    chain_names = {c.name for c in components.correctors}

    # Per-corrector primitives: dm_actuators = A·caller + b (probed), and
    # OPD contribution = 2·T·dm = M·caller + off.
    probed: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for c in components.correctors:
        mirror = _mirror_of(c)
        if mirror is None:
            raise ValueError(
                f"corrector {c.name!r} exposes no mirror surface; it cannot "
                "be expressed in the pure forward function."
            )
        matrix, offset = _probe_affine_actuation(c)
        transform = mirror.influence_functions.transformation_matrix
        m_opd = 2.0 * np.asarray(transform @ matrix)
        off = (
            2.0 * np.asarray(transform @ offset).ravel()
            if offset.any()
            else np.zeros(n_pix, dtype=np.float64)
        )
        probed[c.name] = (m_opd, off)

    # Chain walk. All maps are {input_name: (rows, n_act_input)} + const:
    #   total_*    — running cumulative OPD (rows = n_pix)
    #   own_*      — one corrector's finalized OPD contribution
    #   state_*    — one corrector's caller-facing actuator values (rows = n_flat)
    total_maps: dict[str, np.ndarray] = {}
    total_off = np.zeros(n_pix, dtype=np.float64)
    own_maps: dict[str, tuple[dict[str, np.ndarray], np.ndarray]] = {}
    state_maps: dict[str, tuple[dict[str, np.ndarray], np.ndarray]] = {}
    snapshots: list[tuple[dict[str, np.ndarray], np.ndarray]] = []
    n_actuators: dict[str, int] = {}

    for c in components.correctors:
        snapshots.append((dict(total_maps), total_off))
        m_opd, off_c = probed[c.name]
        n_act = int(c.n_actuators)
        if c.wavefront_role == "fit":
            fs = c.fit_source
            if fs is None or fs == "cumulative_phase_pre_self":
                src_maps, src_off = snapshots[-1]
            elif fs in own_maps:
                src_maps, src_off = own_maps[fs]
            elif fs in chain_names:
                raise ValueError(
                    f"corrector {c.name!r} has wavefront_role='fit' with "
                    f"fit_source={fs!r}, but that corrector appears later in "
                    "the chain. fit_source must reference an earlier "
                    "corrector or use 'cumulative_phase_pre_self'."
                )
            else:
                raise ValueError(
                    f"corrector {c.name!r} has unknown fit_source={fs!r}. "
                    "Use 'cumulative_phase_pre_self' or the name of an "
                    "earlier corrector in the chain."
                )
            # Matching fit values as a linear map of the inputs; the
            # pipeline negates at the apply site (set_actuators(-fit)).
            fit_maps = {k: _probe_fit_columns(c, m) for k, m in src_maps.items()}
            fit_off = _fit_of_offset(c, src_off)
            state_maps[c.name] = ({k: -v for k, v in fit_maps.items()}, -fit_off)
            own = (
                {k: -(m_opd @ v) for k, v in fit_maps.items()},
                off_c - m_opd @ fit_off,
            )
        else:
            n_actuators[c.name] = n_act
            state_maps[c.name] = ({c.name: np.eye(n_act)}, np.zeros(n_act))
            own = ({c.name: m_opd}, off_c)
        own_maps[c.name] = own
        total_maps = _merge_maps(total_maps, own[0])
        total_off = total_off + own[1]

    # Echo programs: caller-facing target values as linear maps of the
    # inputs, mirroring sample() step 5's three strategies.
    echoes: dict[str, tuple[dict[str, jnp.ndarray], jnp.ndarray, tuple[int, ...]]] = {}
    for i, c in enumerate(components.correctors):
        if not c.target or c.target_strategy == "none":
            continue
        maps, const = state_maps[c.name]
        if c.target_strategy != "actuators":
            pre_maps, pre_off = snapshots[i]
            residual_maps = {k: _probe_fit_columns(c, m) for k, m in pre_maps.items()}
            residual_off = _fit_of_offset(c, pre_off)
            if c.target_strategy == "actuators_plus_residual_fit":
                maps = _merge_maps(maps, residual_maps)
                const = const + residual_off
            else:  # residual_fit_only
                maps, const = residual_maps, residual_off
        shape = tuple(int(s) for s in np.shape(np.asarray(c.actuators)))
        echoes[c.name] = (
            {k: jnp.asarray(v, dtype=dtype) for k, v in maps.items()},
            jnp.asarray(const, dtype=dtype),
            shape,
        )

    planes: dict[str, tuple[Any, jnp.ndarray]] = {}
    for name, fp in components.focal_planes.items():
        mft = fp._mft
        amplitude = jnp.asarray(
            np.asarray(fp._amplitude, dtype=np.float64).reshape(mft.pupil_shape), dtype=dtype
        )
        planes[name] = (mft, amplitude)

    return TelescopeForward(
        maps={k: jnp.asarray(v, dtype=dtype) for k, v in total_maps.items()},
        opd_offset=jnp.asarray(total_off, dtype=dtype),
        planes=planes,
        n_actuators=n_actuators,
        dtype=dtype,
        echoes=echoes,
    )


__all__ = ["TelescopeForward", "build_forward"]
