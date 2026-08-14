"""Coronagraph implementations.

Concrete classes register themselves at import time via
``@register("coronagraph", "<name>")``. Built-ins include ``identity``
(no-op passthrough), ``vortex``, ``vector_vortex``, and ``lyot``.
``perfect`` will follow when a variant that exercises it comes online.
"""

# Side-effect imports: register stock implementations
from telescope_sim.coronagraphs import lyot, standard  # noqa: F401
