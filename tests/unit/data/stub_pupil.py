"""Synthetic spider pupils for the segmentation tests.

``generate_pupil`` returns an obstructed circular aperture with four
cardinal spider vanes, optionally with one extra short vane that splits
the first quadrant into two unequal pieces (the "fragment" case).
"""

from __future__ import annotations

import hcipy
import numpy as np


def generate_pupil(
    pupil_grid, *, stub: bool = False, spider_width: float = 0.04, supersample: int = 8
):
    aperture = hcipy.make_obstructed_circular_aperture(
        pupil_diameter=1.0,
        central_obscuration_ratio=0.2,
        num_spiders=4,
        spider_width=spider_width,
    )
    field = hcipy.evaluate_supersampled(aperture, pupil_grid, supersample)
    if stub:
        # One extra vane from the centre out through the first quadrant
        # at ~22° — cuts that quadrant into a small and a large piece.
        stub_vane = hcipy.make_spider((0.0, 0.0), (1.0, 0.4), spider_width)

        def stub_float(grid):  # make_spider yields an integer field; supersampling needs float
            return hcipy.Field(np.asarray(stub_vane(grid), dtype=float), grid)

        field = field * hcipy.evaluate_supersampled(stub_float, pupil_grid, supersample)
    return hcipy.Field(np.asarray(field, dtype=float), pupil_grid)
