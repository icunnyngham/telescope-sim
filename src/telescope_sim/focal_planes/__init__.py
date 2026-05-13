"""FocalPlane implementations.

Concrete classes register themselves at import time via
``@register("focal_plane", "<name>")``. Planned built-ins include ``angular``
(arcsec-based focal grid, the default for science PSFs) and ``physical``
(metric focal grid with explicit ``focal_length`` for fiber coupling and
similar physical-units use cases).
"""
