"""Aperture implementations.

Concrete classes register themselves at import time via
``@register("aperture", "<name>")``. Built-ins: ``segmented_circular``
and ``external_pupil`` (which wraps an arbitrary user-supplied callable
or field). Any aperture can additionally carry a ``segmentation:`` block
(:mod:`telescope_sim.apertures.segmentation`) that partitions the built
pupil into its spider-bounded segments for segment-wise correctors.
"""
