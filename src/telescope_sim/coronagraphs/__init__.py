"""Coronagraph implementations.

Concrete classes register themselves at import time via
``@register("coronagraph", "<name>")``. Built-ins include ``identity``
(no-op passthrough), ``vortex``, and ``vector_vortex``. ``lyot`` and
``perfect`` will follow as variants that exercise them come online.
"""

# Side-effect import: register stock implementations
from telescope_sim.coronagraphs import standard  # noqa: F401
