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
  corrector collapses to one precomputed ``(n_pix, n_actuators)``
  matrix. Correctors with a nonlinear ``set_actuators`` are rejected at
  build time.
- :meth:`TelescopeForward.intensity_from_opd` — OPD to per-focal-plane
  summed intensity, through the same jitted MFT kernels ``sample()``
  uses. This stage is the external-OPD hook: add an atmosphere screen's
  OPD (meters) to the actuation OPD before calling.

Out of scope, by design: output taps, post-processors (detector noise),
actuation echoes, and fit-role correctors (their least-squares
resolution runs host-side in ``sample()``; an in-graph fit stage is a
planned extension). ``forward_fn`` returns raw physics only.
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
    ) -> None:
        self._maps = maps
        self._opd_offset = opd_offset
        self._planes = planes
        self._n_actuators = dict(n_actuators)
        self._dtype = dtype

    @property
    def corrector_names(self) -> tuple[str, ...]:
        return tuple(self._maps)

    @property
    def n_actuators(self) -> dict[str, int]:
        return dict(self._n_actuators)

    @property
    def focal_plane_names(self) -> tuple[str, ...]:
        return tuple(self._planes)

    def opd_from_actuations(self, actuations: Mapping[str, ArrayLike] | None = None) -> jnp.ndarray:
        """Total pupil-plane OPD (meters, flat) for caller actuation values.

        Keys are corrector names; values are anything ``set_actuators``
        accepts for that corrector (any shape raveling to its flat
        actuation length). Missing correctors are flat — the same
        convention as ``sample()``. Unknown keys and wrong sizes raise at
        trace time.
        """
        acts = dict(actuations or {})
        unknown = sorted(set(acts) - set(self._maps))
        if unknown:
            raise ValueError(
                f"unknown corrector(s) in actuations: {unknown}; defined: {list(self._maps)}"
            )
        opd = self._opd_offset
        for name, matrix in self._maps.items():
            if name not in acts:
                continue
            values = jnp.asarray(acts[name], dtype=self._dtype).reshape(-1)
            if values.shape[0] != self._n_actuators[name]:
                raise ValueError(
                    f"actuations[{name!r}]: expected {self._n_actuators[name]} "
                    f"values per sample, got shape {np.shape(acts[name])}"
                )
            opd = opd + matrix @ values
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


def build_forward(components: Any) -> TelescopeForward:
    """Build a :class:`TelescopeForward` from resolved pipeline components."""
    if components.backend != "jax":
        raise NotImplementedError(
            "forward_fn() requires backend='jax'; the hcipy backend has no "
            "pure-function propagation path."
        )
    _check_coronagraph(components.coronagraph)
    fit_role = [c.name for c in components.correctors if c.wavefront_role == "fit"]
    if fit_role:
        raise NotImplementedError(
            f"correctors {fit_role} have wavefront_role='fit', which is "
            "resolved by host-side least squares; the pure forward function "
            "covers only the actuate/impose path. Use sample() instead."
        )

    dtype = jnp.float32 if components.precision == "float32" else jnp.float64
    maps: dict[str, jnp.ndarray] = {}
    n_actuators: dict[str, int] = {}
    opd_offset = np.zeros(components.pupil_grid.size, dtype=np.float64)
    for c in components.correctors:
        mirror = _mirror_of(c)
        if mirror is None:
            raise ValueError(
                f"corrector {c.name!r} exposes no mirror surface; it cannot "
                "be expressed in the pure forward function."
            )
        matrix, offset = _probe_affine_actuation(c)
        transform = mirror.influence_functions.transformation_matrix
        # OPD = 2 × surface = 2 · T · (A · caller + offset); the constant
        # part folds into a single static OPD term.
        maps[c.name] = jnp.asarray(2.0 * np.asarray(transform @ matrix), dtype=dtype)
        if offset.any():
            opd_offset += 2.0 * np.asarray(transform @ offset).ravel()
        n_actuators[c.name] = int(c.n_actuators)

    planes: dict[str, tuple[Any, jnp.ndarray]] = {}
    for name, fp in components.focal_planes.items():
        mft = fp._mft
        amplitude = jnp.asarray(
            np.asarray(fp._amplitude, dtype=np.float64).reshape(mft.pupil_shape), dtype=dtype
        )
        planes[name] = (mft, amplitude)

    return TelescopeForward(
        maps=maps,
        opd_offset=jnp.asarray(opd_offset, dtype=dtype),
        planes=planes,
        n_actuators=n_actuators,
        dtype=dtype,
    )


__all__ = ["TelescopeForward", "build_forward"]
