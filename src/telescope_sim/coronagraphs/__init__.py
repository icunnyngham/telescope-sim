"""Coronagraph implementations.

Concrete classes register themselves at import time via
``@register("coronagraph", "<name>")``. Planned built-ins include ``identity``
(no-op passthrough), ``vortex``, ``vector_vortex``, ``lyot``, and ``perfect``.
"""
