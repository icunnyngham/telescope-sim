"""SimulateMultiApertureTelescope v1.x compatibility shim.

Maps the legacy constructor's keyword arguments to a preset or config and
delegates ``sample()`` calls through to the current pipeline. Best-effort only;
unusual configurations may need migration to a custom YAML.
"""

from __future__ import annotations

from typing import Any


class SimulateMultiApertureTelescope:
    """Drop-in (best-effort) for the v1.x high-level wrapper.

    Deprecated. Migrate to :class:`telescope_sim.TelescopeSim` constructed
    from a YAML config or preset.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError(
            "Legacy compatibility shim is not yet implemented. "
            "Use telescope_sim.TelescopeSim directly."
        )


__all__ = ["SimulateMultiApertureTelescope"]
