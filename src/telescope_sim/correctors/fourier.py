"""Fourier-mode deformable mirror corrector.

A low-order modal DM over a Fourier basis: cosine and sine modes at a
discrete set of low spatial frequencies, sorted by energy. A natural
action space for ML-controlled DMs whose commands should be smooth, and a
natural residual-fit target when a chain composes segmented or
atmospheric disturbances — a smooth-mode DM reproduces the smooth part of
an upstream disturbance and leaves the discontinuities (for example the
spider-edge steps of a per-petal piston pattern) as fitting error.

Construction is driven by ``n_axis``: an ``n_axis × n_axis`` slice of the
pupil grid's FFT frequency grid is taken (``hcipy.make_fft_grid(...,
fov=n_axis / pupil_dim)``) and handed to ``hcipy.make_fourier_basis``,
which emits a cosine and a sine mode per frequency point with ``±p``
pairs de-duplicated and a single piston for DC. The modes are energy
sorted and peak-normalized, then wrapped in an ``hcipy.DeformableMirror``.

Mode count for an ``n_axis × n_axis`` frequency grid on an even-parity
pupil grid (``n_axis`` even, no shift). The FFT grid is asymmetric (the
``+(n_axis/2)·δ`` endpoint is missing), so ``2·n_axis − 1`` "lonely"
frequencies — those touching ``−(n_axis/2)·δ`` on either axis — have no
``±p`` partner to de-duplicate against:

- 1 piston (origin),
- ``((n_axis − 1)² − 1) / 2`` symmetric ``±p`` pairs, 2 modes each,
- ``2·n_axis − 1`` lonely frequencies, 2 modes each,

which sums to ``n_axis² + 2·n_axis − 1``. For the default ``n_axis = 6``:
``1 + 12·2 + 11·2 = 47`` modes; ``n_axis = 4`` gives 23, ``n_axis = 8``
gives 79. ``remove_piston=True``
(the default) drops the DC mode — a uniform pupil-plane offset is
unobservable in a Fraunhofer PSF — so the default basis has **46**
actuators. ``n_modes`` optionally truncates to the lowest-energy ``N``
modes after piston removal; the DM is built from the truncated basis, so
``n_actuators`` is exactly what was asked for.

Caller-facing actuator state has shape ``(n_actuators,)``. The corrector
multiplies caller values by ``actuate_scale`` before writing the DM, so a
value of 1 is a peak surface displacement of ``actuate_scale`` meters on
that mode (the same convention as ``zernike``).
"""

from __future__ import annotations

from typing import Any

import hcipy
import numpy as np
from numpy.typing import ArrayLike, NDArray

from telescope_sim.abc import Corrector
from telescope_sim.abc.corrector import TargetStrategy, WavefrontRole
from telescope_sim.registry import register


@register("corrector", "fourier")
class FourierCorrector(Corrector):
    """Fourier-basis deformable mirror.

    Parameters
    ----------
    n_axis
        Frequency-grid axis count. Default 6 yields 47 raw modes on an
        even-parity pupil grid (46 with the piston removed); see the
        module docstring for the count formula.
    n_modes
        Optional truncation to the lowest-energy ``n_modes`` modes
        (after piston removal when ``remove_piston`` is set). ``None``
        keeps every emitted mode.
    remove_piston
        Drop the DC piston mode (default ``True``). A uniform pupil-plane
        offset is unobservable in the PSF; keeping it spends an actuator
        on nothing and clutters fit-derived targets. Set ``False`` to
        match a third-party DM model that exposes piston.
    actuate_scale
        Multiplier from caller actuator values to meters of peak surface
        displacement per mode.
    """

    def __init__(
        self,
        *,
        n_axis: int = 6,
        n_modes: int | None = None,
        remove_piston: bool = True,
        actuate_scale: float = 1.0e-6,
        name: str = "fourier_dm",
        wavefront_role: WavefrontRole = "actuate",
        target_strategy: TargetStrategy = "none",
        fit_source: str | None = None,
        target: bool = False,
    ) -> None:
        self.name = name
        self.n_axis = int(n_axis)
        self.n_modes = None if n_modes is None else int(n_modes)
        self.remove_piston = bool(remove_piston)
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
        self._mode_freqs: NDArray[np.floating] | None = None
        self.n_modes_effective: int = 0

    # --- Loader-driven setup ----------------------------------------------

    def _bind_pupil_grid(self, pupil_grid: Any, aperture_field: Any) -> None:
        """Build the Fourier basis + HCIPy DM on a given pupil grid."""
        fov = self.n_axis / max(pupil_grid.dims)
        fourier_grid = hcipy.make_fft_grid(pupil_grid, q=1, fov=fov)
        basis_full = hcipy.make_fourier_basis(pupil_grid, fourier_grid, sort_by_energy=True)
        modes_list = list(basis_full)
        raw_freqs = self._mode_frequencies_for(fourier_grid)

        if self.remove_piston:
            # Mode 0 of the energy-sorted basis is DC (constant 1).
            modes_list = modes_list[1:]
            raw_freqs = raw_freqs[1:]

        if self.n_modes is not None:
            if len(modes_list) < self.n_modes:
                raise ValueError(
                    f"n_axis={self.n_axis} yields {len(modes_list)} Fourier modes "
                    f"(remove_piston={self.remove_piston}), fewer than the requested "
                    f"n_modes={self.n_modes}; increase n_axis."
                )
            modes_list = modes_list[: self.n_modes]
            raw_freqs = raw_freqs[: self.n_modes]

        # Peak-normalize so caller amplitudes map to a predictable peak
        # stroke (same convention as ZernikeCorrector).
        modes = [m / np.max(np.abs(m)) for m in modes_list]
        self._basis = hcipy.ModeBasis(modes)
        self._dm = hcipy.DeformableMirror(self._basis)
        self.n_modes_effective = len(modes)
        self._mode_freqs = np.asarray(raw_freqs, dtype=float)

        # (n_pix, n_modes) basis matrix + aperture mask for the lstsq fit.
        self._basis_matrix = np.column_stack([np.asarray(m, dtype=float).ravel() for m in modes])
        self._aperture_mask = np.asarray(aperture_field, dtype=float).ravel() > 0

    @staticmethod
    def _mode_frequencies_for(fourier_grid: Any) -> NDArray[np.floating]:
        """Spatial-frequency magnitude per emitted mode, in basis order.

        Reproduces the energy sort and the one-mode-for-DC / two-modes-
        otherwise emission of ``make_fourier_basis`` so the array aligns
        with the basis index.
        """
        pts = np.asarray(fourier_grid.points)
        energies = (pts * pts).sum(axis=1)
        freqs: list[float] = []
        for i in np.argsort(energies, kind="stable"):
            e = float(energies[i])
            if e <= 1e-30:
                freqs.append(0.0)
            else:
                freqs.extend([float(np.sqrt(e))] * 2)
        return np.asarray(freqs, dtype=float)

    # --- Corrector interface ----------------------------------------------

    def apply(self, wf: Any) -> Any:
        if self._dm is None:
            raise RuntimeError(
                "FourierCorrector must be bound to a pupil grid via "
                "_bind_pupil_grid() before apply()."
            )
        return self._dm(wf)

    def set_actuators(self, values: ArrayLike) -> None:
        if self._dm is None:
            raise RuntimeError("set_actuators() before _bind_pupil_grid()")
        arr = np.asarray(values, dtype=float).reshape(-1)
        if arr.size != self.n_modes_effective:
            raise ValueError(f"expected {self.n_modes_effective} actuators, got {arr.size}")
        self._dm.actuators = arr * self.actuate_scale

    def flatten(self) -> None:
        if self._dm is not None:
            self._dm.actuators = np.zeros(self.n_modes_effective)

    def fit_surface(self, phase: NDArray[np.floating]) -> NDArray[np.floating]:
        """Least-squares projection of a pupil-plane OPD onto the Fourier basis.

        Input is OPD in meters (path length); output is caller-facing
        actuator amplitudes that *reproduce* that OPD as this DM's surface
        contribution (matching, not cancellation — see
        :meth:`Corrector.fit_surface`), with the same ``/(2·actuate_scale)``
        surface→OPD round-trip factor as ``zernike``. The aperture-masked
        mean is subtracted first: with ``remove_piston`` there is no mode
        to absorb a constant offset, which would otherwise distort the
        fit of the low-frequency modes.
        """
        if self._basis_matrix is None or self._aperture_mask is None:
            raise RuntimeError("fit_surface() before _bind_pupil_grid()")
        phase = np.asarray(phase, dtype=float).ravel()
        phase = phase - phase[self._aperture_mask].mean()
        B = self._basis_matrix[self._aperture_mask]
        rhs = phase[self._aperture_mask]
        amps, _, _, _ = np.linalg.lstsq(B, rhs, rcond=None)
        return amps / (2.0 * self.actuate_scale)

    @property
    def n_actuators(self) -> int:
        return self.n_modes_effective

    @property
    def actuators(self) -> NDArray:
        if self._dm is None:
            return np.zeros(self.n_modes_effective)
        return np.asarray(self._dm.actuators) / self.actuate_scale

    @property
    def basis(self) -> Any | None:
        """The peak-normalized ``hcipy.ModeBasis`` driving the DM."""
        return self._basis

    @property
    def mode_frequencies(self) -> NDArray[np.floating] | None:
        """Spatial-frequency magnitude (radians per pupil-plane unit) per actuator."""
        return self._mode_freqs


__all__ = ["FourierCorrector"]
