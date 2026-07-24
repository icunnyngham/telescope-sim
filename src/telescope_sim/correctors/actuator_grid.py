"""Actuator-grid deformable mirror corrector.

Wraps HCIPy's ``DeformableMirror`` over an N×N grid of influence
functions (``make_gaussian_influence_functions`` or
``make_xinetics_influence_functions``), driven by raw per-actuator
commands. DM misalignment relative to the pupil — rotation and
mirrored actuator indexing — is baked in at construction so a
simulated bench can reproduce the command-to-surface mapping of a
physically mounted DM.

Caller-facing actuator state has shape ``(N²,)`` (a shaped ``(N, N)``
command array is accepted equivalently). The corrector multiplies
caller values by ``actuate_scale`` before handing them to the
underlying HCIPy DM, so callers command in their own units (e.g. the
bench's raw DM units) and ``actuate_scale`` carries the
units→meters-of-surface calibration.

Command-array and misalignment conventions
------------------------------------------

- A shaped ``(N, N)`` command indexes the actuator lattice as
  ``cmd[iy, ix]``: axis 0 walks the y coordinate ascending, axis 1
  walks x ascending. Flattening is row-major, matching HCIPy's
  actuator-position ordering (x varies fastest). The lattice is
  centered on the optical axis with spacing ``actuator_pitch``.
- ``flip_x`` / ``flip_y`` mirror the *command indexing* (``fliplr`` /
  ``flipud`` of the shaped command) before it reaches the DM — the
  simulated analogue of a mirrored cable/mapping between the command
  array and the physical actuators. The ``actuators`` readback
  un-applies the flips, so callers always round-trip their own values.
- ``rotation_deg`` rotates the *influence-function geometry*: positive
  values rotate the DM counterclockwise relative to the pupil (x
  right, y up) as seen in a plotted surface. Composition order is
  flip-then-rotate: the command array is mirrored first, then the
  rotated DM renders it.
"""

from __future__ import annotations

from typing import Any

import hcipy
import numpy as np
import scipy.linalg
import scipy.sparse
from numpy.typing import ArrayLike, NDArray

from telescope_sim.abc import Corrector
from telescope_sim.abc.corrector import TargetStrategy, WavefrontRole
from telescope_sim.registry import register

_INFLUENCE_FACTORIES = {
    "gaussian": hcipy.make_gaussian_influence_functions,
    "xinetics": hcipy.make_xinetics_influence_functions,
}


@register("corrector", "actuator_grid")
class ActuatorGridCorrector(Corrector):
    """N×N actuator-grid deformable mirror with influence functions.

    Parameters
    ----------
    num_actuators
        Actuators per side; the DM has ``num_actuators²`` total.
    actuator_pitch
        Actuator spacing in pupil-plane projected units (meters for a
        metric pupil grid).
    influence
        Influence-function model: ``"gaussian"`` (Gaussian pokes with
        nearest-neighbour ``crosstalk``) or ``"xinetics"`` (HCIPy's
        measured Xinetics actuator shape).
    crosstalk
        Influence-function value at a nearest-neighbour actuator.
        Only meaningful for ``influence="gaussian"``; ignored for
        ``"xinetics"`` (whose shape is fixed by measurement).
    rotation_deg
        DM orientation relative to the pupil. Positive values rotate
        the DM counterclockwise (x right, y up) as seen in a plotted
        surface.
    flip_x, flip_y
        Mirror the command-array indexing along x (``fliplr``) / y
        (``flipud``) before commands reach the DM. Applied before the
        rotation (flip-then-rotate).
    actuate_scale
        Multiplicative factor applied to caller actuator values before
        they are written to the underlying HCIPy DM (caller units →
        meters of surface).
    """

    def __init__(
        self,
        num_actuators: int,
        actuator_pitch: float,
        *,
        influence: str = "gaussian",
        crosstalk: float = 0.15,
        rotation_deg: float = 0.0,
        flip_x: bool = False,
        flip_y: bool = False,
        actuate_scale: float = 1.0,
        name: str = "actuator_dm",
        wavefront_role: WavefrontRole = "actuate",
        target_strategy: TargetStrategy = "none",
        fit_source: str | None = None,
        target: bool = False,
    ) -> None:
        if influence not in _INFLUENCE_FACTORIES:
            raise ValueError(
                f"unknown influence {influence!r}; expected one of {sorted(_INFLUENCE_FACTORIES)}"
            )
        self.name = name
        self.num_actuators = int(num_actuators)
        self.actuator_pitch = float(actuator_pitch)
        self.influence = influence
        self.crosstalk = float(crosstalk)
        self.rotation_deg = float(rotation_deg)
        self.flip_x = bool(flip_x)
        self.flip_y = bool(flip_y)
        self.actuate_scale = float(actuate_scale)

        self.wavefront_role = wavefront_role
        self.target_strategy = target_strategy
        self.fit_source = fit_source
        self.target = target

        # Populated by :meth:`_bind_pupil_grid` (called by the pipeline loader)
        self._dm: Any | None = None
        self._aperture_mask: NDArray[np.bool_] | None = None
        # Lazy fit-solver state, built on first fit_surface() call
        self._fit_matrix: Any | None = None
        self._fit_cho: Any | None = None

    def _bind_pupil_grid(self, pupil_grid: Any, aperture_field: Any) -> None:
        """Build the influence functions + HCIPy DM on a given pupil grid.

        Both HCIPy factories place actuators on an internal unrotated
        lattice and evaluate the pokes on ``pupil_grid.rotated(-z_tilt)``,
        so passing ``z_tilt`` renders the DM rotated by ``+z_tilt``
        (counterclockwise) relative to the pupil.
        """
        factory_kwargs: dict[str, Any] = {"z_tilt": np.deg2rad(self.rotation_deg)}
        if self.influence == "gaussian":
            factory_kwargs["crosstalk"] = self.crosstalk
        basis = _INFLUENCE_FACTORIES[self.influence](
            pupil_grid,
            self.num_actuators,
            self.actuator_pitch,
            **factory_kwargs,
        )
        self._dm = hcipy.DeformableMirror(basis)
        # Aperture mask for fit_surface: fit only over transmitting pixels
        # (matches the ZernikeCorrector convention).
        self._aperture_mask = np.asarray(aperture_field, dtype=float).ravel() > 0
        self._fit_matrix = None
        self._fit_cho = None

    def _shape_command(self, values: ArrayLike) -> NDArray[np.floating]:
        """Coerce a flat ``(N²,)`` or shaped ``(N, N)`` command to ``(N, N)``."""
        n = self.num_actuators
        arr = np.asarray(values, dtype=float)
        if arr.ndim == 2 and arr.shape != (n, n):
            raise ValueError(f"expected ({n}, {n}) shaped actuators, got {arr.shape}")
        if arr.size != n * n:
            raise ValueError(f"expected {n * n} actuators, got {arr.size}")
        return arr.reshape(n, n)

    # --- Corrector interface ----------------------------------------------

    def apply(self, wf: Any) -> Any:
        if self._dm is None:
            raise RuntimeError(
                "ActuatorGridCorrector must be bound to a pupil grid via "
                "_bind_pupil_grid() before apply()."
            )
        return self._dm(wf)

    def set_actuators(self, values: ArrayLike) -> None:
        if self._dm is None:
            raise RuntimeError("set_actuators() before _bind_pupil_grid()")
        shaped = self._shape_command(values)
        # Mirror the command indexing (cable/mapping flip) before the
        # rotated influence geometry renders it — flip-then-rotate.
        if self.flip_y:
            shaped = np.flipud(shaped)
        if self.flip_x:
            shaped = np.fliplr(shaped)
        self._dm.actuators = shaped.reshape(-1) * self.actuate_scale

    def flatten(self) -> None:
        if self._dm is not None:
            self._dm.actuators = np.zeros(self.n_actuators)

    def _build_fit_solver(self) -> None:
        """Cache the aperture-masked influence matrix + normal-equations factor.

        The influence basis is sparse (HCIPy truncates pokes at a cutoff
        radius), so the masked matrix stays sparse and the dense work is
        confined to the (N², N²) Gram matrix. On sparse pupils many grid
        actuators have little or no support inside the aperture, making
        the Gram matrix singular; a tiny Tikhonov term (1e-9 x mean
        diagonal) keeps the factorization stable and pins unilluminated
        actuators near zero instead of letting them blow up.
        """
        matrix = self._dm.influence_functions.transformation_matrix
        masked = matrix[np.flatnonzero(self._aperture_mask)]
        gram = masked.T @ masked
        if scipy.sparse.issparse(gram):
            gram = gram.toarray()
        gram = np.asarray(gram, dtype=float)
        n = gram.shape[0]
        gram[np.diag_indices(n)] += 1e-9 * np.trace(gram) / n
        self._fit_matrix = masked
        self._fit_cho = scipy.linalg.cho_factor(gram)

    def fit_surface(self, phase: NDArray[np.floating]) -> NDArray[np.floating]:
        """Least-squares-project a pupil-plane OPD onto the influence basis.

        Input is OPD in meters (path-length); output is caller-facing
        actuator commands that *reproduce* that OPD as this DM's surface
        contribution (matching, not cancellation — see
        :meth:`Corrector.fit_surface` for the convention). The fit is an
        aperture-masked regularized least squares onto the (rotated)
        influence functions; command flips are un-applied on the way out,
        so ``set_actuators(fit_surface(opd))`` reproduces the fit
        regardless of the configured misalignment.

        The ``/ 2.0`` is the surface→OPD round-trip factor: setting
        actuator value ``v`` produces surface ``v * actuate_scale`` but
        contributes ``2 * v * actuate_scale`` to the OPD.

        The aperture-masked mean of the input is subtracted before the
        fit. A uniform OPD offset is unobservable in the PSF, and unlike
        a global phase the DM's *approximation* of one is not: the
        influence basis only holds a flat to within a print-through
        ripple, so fitting the mean would leak ``mean x ripple`` into the
        corrected wavefront and pollute residual-fit ML targets with an
        arbitrary common-mode term. Piston is therefore never commanded.
        """
        if self._dm is None or self._aperture_mask is None:
            raise RuntimeError("fit_surface() before _bind_pupil_grid()")
        if self._fit_cho is None:
            self._build_fit_solver()
        phase = np.asarray(phase, dtype=float).ravel()
        rhs = phase[self._aperture_mask]
        rhs = rhs - rhs.mean()
        amps = scipy.linalg.cho_solve(self._fit_cho, np.asarray(self._fit_matrix.T @ rhs).ravel())
        shaped = amps.reshape(self.num_actuators, self.num_actuators)
        # Un-apply the command flips (as in the `actuators` readback) so the
        # returned values are in the caller's command indexing.
        if self.flip_x:
            shaped = np.fliplr(shaped)
        if self.flip_y:
            shaped = np.flipud(shaped)
        return shaped.reshape(-1) / (2.0 * self.actuate_scale)

    @property
    def n_actuators(self) -> int:
        return self.num_actuators**2

    @property
    def actuators(self) -> NDArray:
        if self._dm is None:
            return np.zeros(self.n_actuators)
        shaped = (np.asarray(self._dm.actuators) / self.actuate_scale).reshape(
            self.num_actuators, self.num_actuators
        )
        # Un-apply the command flips so callers read back what they set.
        if self.flip_x:
            shaped = np.fliplr(shaped)
        if self.flip_y:
            shaped = np.flipud(shaped)
        return shaped.reshape(-1)


__all__ = ["ActuatorGridCorrector"]
