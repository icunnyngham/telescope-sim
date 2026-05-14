"""Corrector ABC — pupil-plane element with actuator state and role-based behavior.

Two orthogonal axes control a corrector's role in the pipeline:

- ``wavefront_role``: how actuator state gets set during ``sample()``
    * ``"actuate"`` — set by the ML model (caller's ``actuations`` dict)
    * ``"impose"``  — set by the caller via a separate channel
                      (e.g. environment, atmosphere, fixed aberrations)
    * ``"fit"``     — lstsq-fit to another corrector's surface or a named
                      wavefront. Requires ``fit_source``.

- ``target_strategy``: what this corrector contributes to the Y output dict
    * ``"none"``                          — not in Y
    * ``"actuators"``                     — echo back current actuator state
    * ``"actuators_plus_residual_fit"``   — ``actuators + fit_surface(
                                            cumulative_phase_pre_self)``
    * ``"residual_fit_only"``             — ``fit_surface(
                                            cumulative_phase_pre_self)`` only

``fit_source`` (used when ``wavefront_role == "fit"``) can be:

- a corrector name (string)            — fit to that corrector's surface
- a named wavefront                    — fit to that wavefront's phase
- the special value
  ``"cumulative_phase_pre_self"``      — fit to cumulative phase just before
                                         this corrector in the chain

Common patterns this expresses:

- **Segment-impose / DM-fit**: ``segments`` (``actuate``, target=actuators)
  plus a downstream ``xinetics`` corrector (``fit``, ``fit_source=segments``,
  target=none) — the segmented mirror's commanded surface is approximated by
  the DM in the beam, but Y reports the segment commands.
- **Atmosphere + residual fit**: an ``impose`` corrector for the atmosphere
  followed by an ``actuate`` corrector with
  ``target_strategy=actuators_plus_residual_fit`` — Y reports
  ``model_actuators + fit(atmosphere)``, i.e. the correction the model would
  have needed to fully cancel the disturbance.
- **Stacked imposes + fitted DM**: any number of ``impose`` correctors
  followed by a single ``actuate`` corrector with
  ``target_strategy=actuators_plus_residual_fit`` for ML training against
  cumulative residual phase.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

WavefrontRole = Literal["actuate", "impose", "fit"]
TargetStrategy = Literal[
    "none",
    "actuators",
    "actuators_plus_residual_fit",
    "residual_fit_only",
]


class Corrector(ABC):
    """ABC for pluggable pupil-plane correctors (DMs, segmented mirrors, etc.).

    Concrete implementations must provide :meth:`apply` and :meth:`set_actuators`.
    Implementations that participate in fit-roles or residual-fit targets must
    additionally provide :meth:`fit_surface`.
    """

    name: str
    wavefront_role: WavefrontRole
    target_strategy: TargetStrategy = "none"
    fit_source: str | None = None
    target: bool = False

    @abstractmethod
    def apply(self, wf: Any) -> Any:
        """Apply the corrector's current state to a wavefront in-chain."""
        ...

    @abstractmethod
    def set_actuators(self, values: ArrayLike) -> None:
        """Set the corrector's actuator state."""
        ...

    def flatten(self) -> None:
        """Zero all actuators. Default impl calls ``set_actuators(0)``-shaped."""
        self.set_actuators(np.zeros(self.n_actuators))

    def fit_surface(self, phase: NDArray[np.floating]) -> NDArray[np.floating]:
        """Fit the corrector's actuator basis to a pupil-plane OPD array.

        Required when participating in ``role=fit`` or
        ``target_strategy=actuators_plus_residual_fit`` / ``residual_fit_only``.
        Default implementation raises :class:`NotImplementedError` to make
        misconfiguration visible.

        Convention
        ----------
        - **Input** ``phase``: pupil-plane optical path difference (OPD) in
          meters — i.e. path-length, ``OPD = 2 * surface`` for a single DM
          reflection. The pipeline accumulates this as it walks the chain.
        - **Output**: *matching* caller-facing actuator values — i.e. values
          that, when set via :meth:`set_actuators`, would *reproduce* the
          input OPD as this corrector's surface contribution (modulo the
          round-trip factor; surface = OPD / 2).
        - **Sign**: matching, not cancellation. To cancel the input OPD,
          the caller negates. The pipeline does this automatically at the
          apply site for ``wavefront_role="fit"``.
        - **Y echoes** for ``target_strategy="actuators_plus_residual_fit"``
          and ``"residual_fit_only"`` consume ``fit_surface`` *unnegated*:
          Y reports the wavefront state in this corrector's basis (matches
          legacy v1 ``out_actuate = caller + matching_fit(atmos)``). An ML
          model trained to predict Y is then driven through ``-Y`` by the
          downstream controller to cancel.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support fit_surface(). "
            "Override this method to enable use as a fit-role or residual-fit "
            "corrector."
        )

    @property
    @abstractmethod
    def n_actuators(self) -> int: ...

    @property
    @abstractmethod
    def actuators(self) -> NDArray: ...


__all__ = ["Corrector", "WavefrontRole", "TargetStrategy"]
