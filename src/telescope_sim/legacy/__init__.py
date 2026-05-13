"""Legacy compatibility shim.

A best-effort drop-in replacement for the v1.x ``SimulateMultiApertureTelescope``
high-level wrapper, implemented atop the current pipeline. Provided to ease
migration of existing user code; documented as deprecated.
"""

from telescope_sim.legacy.shim import SimulateMultiApertureTelescope

__all__ = ["SimulateMultiApertureTelescope"]
