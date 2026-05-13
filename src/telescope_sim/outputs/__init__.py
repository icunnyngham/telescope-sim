"""OutputTap implementations.

Concrete classes register themselves at import time via
``@register("output_tap", "<name>")``. Planned built-ins include ``intensity``
(standard focal-plane PSF), ``fiber_coupled`` (couples the focal-plane
wavefront into a fiber mode field), and ``phase`` (pupil-plane phase output).
"""
