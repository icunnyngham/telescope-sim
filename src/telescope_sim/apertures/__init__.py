"""Aperture implementations.

Concrete classes register themselves at import time via
``@register("aperture", "<name>")``. Planned built-ins include
``segmented_circular``, ``segmented_hexagonal``, ``monolithic``, and
``external_pupil`` (which wraps an arbitrary user-supplied callable).
"""
