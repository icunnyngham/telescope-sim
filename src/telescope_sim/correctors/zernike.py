"""Zernike-basis deformable mirror corrector.

Wraps HCIPy's ``DeformableMirror`` over a Zernike mode basis with the
per-mode normalization the VAMPIRES-family variants use (each mode is
divided by its peak absolute value so caller-facing actuator amplitudes
are in units of ``max-mode-amplitude × actuate_scale``).

Caller-facing actuator state has shape ``(n_modes,)``. The corrector
multiplies caller values by ``actuate_scale`` before handing them to the
underlying HCIPy DM.
"""

from __future__ import annotations

from typing import Any

import hcipy
import numpy as np
from numpy.typing import ArrayLike, NDArray

from telescope_sim.abc import Corrector
from telescope_sim.abc.corrector import TargetStrategy, WavefrontRole
from telescope_sim.registry import register


@register("corrector", "zernike")
class ZernikeCorrector(Corrector):
    """Zernike-mode deformable mirror.

    Parameters
    ----------
    n_modes
        Number of Zernike modes (Noll indexing; starts at ``starting_mode``).
    zernike_diameter
        Diameter (pupil-plane units) over which the Zernike basis is defined.
    starting_mode
        First Noll index in the basis (default 2, skipping piston).
    actuate_scale
        Multiplicative factor applied to caller actuator values before they
        are written to the underlying HCIPy DM.
    """

    def __init__(
        self,
        n_modes: int,
        zernike_diameter: float,
        *,
        starting_mode: int = 2,
        actuate_scale: float = 1.0,
        name: str = "zernike_dm",
        wavefront_role: WavefrontRole = "actuate",
        target_strategy: TargetStrategy = "none",
        fit_source: str | None = None,
        target: bool = False,
    ) -> None:
        self.name = name
        self.n_modes = int(n_modes)
        self.zernike_diameter = float(zernike_diameter)
        self.starting_mode = int(starting_mode)
        self.actuate_scale = float(actuate_scale)

        self.wavefront_role = wavefront_role
        self.target_strategy = target_strategy
        self.fit_source = fit_source
        self.target = target

        # Populated by :meth:`_bind_pupil_grid` (called by the pipeline loader)
        self._dm: Any | None = None
        self._basis: Any | None = None
        self._basis_matrix: NDArray[np.floating] | None = None
        self._aperture_mask: NDArray[np.bool_] | None = None

    def _bind_pupil_grid(self, pupil_grid: Any, aperture_field: Any) -> None:
        """Build the Zernike basis + HCIPy DM on a given pupil grid."""
        basis = hcipy.make_zernike_basis(
            self.n_modes,
            self.zernike_diameter,
            pupil_grid,
            starting_mode=self.starting_mode,
        )
        # Peak-normalize each mode to match the canonical VAMPIRES treatment
        basis = hcipy.ModeBasis([b / np.max(np.abs(b)) for b in basis])
        self._basis = basis
        self._dm = hcipy.DeformableMirror(basis)

        # Stack into a (n_pix, n_modes) matrix for the lstsq fit.
        self._basis_matrix = np.column_stack([np.asarray(m, dtype=float).ravel() for m in basis])
        # Aperture mask: only fit phase over transmitting pixels (matches
        # legacy `_aprox_via_dm` which slices via `self.aper_sel`).
        self._aperture_mask = np.asarray(aperture_field, dtype=float).ravel() > 0

    # --- Corrector interface ----------------------------------------------

    def apply(self, wf: Any) -> Any:
        if self._dm is None:
            raise RuntimeError(
                "ZernikeCorrector must be bound to a pupil grid via "
                "_bind_pupil_grid() before apply()."
            )
        return self._dm(wf)

    def set_actuators(self, values: ArrayLike) -> None:
        if self._dm is None:
            raise RuntimeError("set_actuators() before _bind_pupil_grid()")
        arr = np.asarray(values, dtype=float).reshape(-1)
        if arr.size != self.n_modes:
            raise ValueError(f"expected {self.n_modes} actuators, got {arr.size}")
        self._dm.actuators = arr * self.actuate_scale

    def flatten(self) -> None:
        if self._dm is not None:
            self._dm.actuators = np.zeros(self.n_modes)

    def fit_surface(self, phase: NDArray[np.floating]) -> NDArray[np.floating]:
        """Project a pupil-plane OPD onto the Zernike basis.

        Input is OPD in meters (path-length); output is caller-facing
        actuator amplitudes that *reproduce* that OPD as this DM's
        surface contribution (matching, not cancellation — see
        :meth:`Corrector.fit_surface` for the convention). Modes are
        peak-normalized; this is the standard diagonal projection,
        which is exact when the basis is orthogonal over the aperture
        footprint (e.g. a clean circular aperture).

        The ``/ 2.0`` is the surface→OPD round-trip factor: setting
        actuator value ``v`` produces surface ``v * actuate_scale``
        but contributes ``2 * v * actuate_scale`` to the OPD.

        Defensive: the aperture-masked mean of the input is subtracted
        before the lstsq. Uniform OPD offsets are unobservable in the
        PSF, but a non-zero-mean input (atmosphere phase screen,
        fit-source corrector surface, etc.) would otherwise be
        absorbed into non-piston modes, distorting them and polluting
        any ML target derived from the fit.
        """
        if self._basis_matrix is None or self._aperture_mask is None:
            raise RuntimeError("fit_surface() before _bind_pupil_grid()")
        phase = np.asarray(phase, dtype=float).ravel()
        # Subtract aperture-masked mean before lstsq (see docstring).
        phase = phase - phase[self._aperture_mask].mean()
        # Solve B[mask] @ amps = phase[mask] in the least-squares sense.
        # Matches legacy `_aprox_via_dm`: lstsq on aperture-masked pixels.
        # This is exact when the basis is linearly independent over the
        # aperture support (the usual case), even if individual modes
        # aren't strictly orthogonal on the discrete grid.
        B = self._basis_matrix[self._aperture_mask]
        rhs = phase[self._aperture_mask]
        amps, _, _, _ = np.linalg.lstsq(B, rhs, rcond=None)
        return amps / (2.0 * self.actuate_scale)

    @property
    def n_actuators(self) -> int:
        return self.n_modes

    @property
    def actuators(self) -> NDArray:
        if self._dm is None:
            return np.zeros(self.n_modes)
        return np.asarray(self._dm.actuators) / self.actuate_scale


__all__ = ["ZernikeCorrector"]
