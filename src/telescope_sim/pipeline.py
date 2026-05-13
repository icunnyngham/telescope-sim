"""Pipeline orchestrator — holds the optical chain and runs `sample()`.

This module exposes the top-level :class:`TelescopeSim` class, the entry point
for constructing a simulation from a YAML config, a preset, or a validated
pydantic config object. Pipeline construction and the optical-chain execution
land in subsequent development phases; this stub gives users a clear error and
a pointer to the docs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


class TelescopeSim:
    """Composable telescope-PSF simulator.

    Construct via one of the classmethods (not via ``__init__`` directly):

    - :meth:`from_preset` — load a packaged preset by name
    - :meth:`from_yaml`   — load a user-supplied YAML config
    - :meth:`from_config` — supply an already-validated pydantic config

    Calls currently raise :class:`NotImplementedError` while implementation is
    in progress.
    """

    def __init__(self, config: Any) -> None:
        self._config = config
        self.atmosphere: Any = None
        raise NotImplementedError(
            "TelescopeSim pipeline construction is not yet implemented."
        )

    @classmethod
    def from_preset(cls, name: str) -> TelescopeSim:
        """Load a packaged preset by name (e.g. ``"elf_7seg"``)."""
        raise NotImplementedError("Presets are not yet implemented.")

    @classmethod
    def from_yaml(cls, path: str | Path) -> TelescopeSim:
        """Load configuration from a YAML file."""
        raise NotImplementedError("YAML loader is not yet implemented.")

    @classmethod
    def from_config(cls, config: Any) -> TelescopeSim:
        """Instantiate from an already-validated pydantic config model."""
        raise NotImplementedError("Config-based instantiation is not yet implemented.")

    def sample(
        self,
        actuations: dict[str, Any] | None = None,
        *,
        t_exp: float = 1.0,
        int_phot_flux: float | None = None,
        meas_strehl: bool = False,
        extras: frozenset[str] | None = None,
    ) -> dict[str, Any]:
        """Run the optical chain and return a dict of outputs.

        Returns
        -------
        dict
            Keys: ``images`` (dict[str, ndarray]), ``actuations`` (dict[str, ndarray]),
            and conditionally ``strehls``, ``phase``, etc.
        """
        raise NotImplementedError("sample() is not yet implemented.")


__all__ = ["TelescopeSim"]
