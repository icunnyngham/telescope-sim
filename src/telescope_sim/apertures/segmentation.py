"""Pupil segmentation by connected-components labeling.

Partitions an already-built aperture transmission map into disjoint
segments — the regions a spider cuts a monolithic pupil into (its
"petals") — so that segment-wise correctors such as ``segmented_ptt`` can
drive a pupil whose aperture kind does not define segments itself (for
example an ``external_pupil`` mask). Requested from YAML through the
aperture's optional ``segmentation:`` block::

    aperture:
      type: external_pupil
      module: my_pupil.py
      segmentation:
        method: connected_components
        n_segments: 4          # 4 for a 4-vane spider, 6 for an ELT-style pupil
        merge_fragments: true  # absorb extra regions into the nearest kept one
        threshold: 0.0         # a pixel is lit where transmission > threshold

The loader calls :func:`partition_connected_components` after the aperture
build and stores the result on the
:class:`~telescope_sim.abc.ApertureResult` (``segments``,
``segment_coords``, and a ``metadata["segmentation"]`` record), exactly
the fields a segmented aperture kind would have produced. Downstream
nothing changes: the pupil-plane transmission is untouched, and
``segmented_ptt`` builds its piston/tip/tilt mirror on the segment masks
on either compute backend.

The typical use is the **low wind effect** / **island effect**: under low
wind, radiatively cooled spider arms imprint a differential piston, tip
and tilt on each spider-bounded region of the pupil. The standard
description in the literature is the first three Zernike modes over each
region — 12 modes on a 4-vane pupil, of rank 11 once the unobservable
mean piston is removed — and, on segmented giant telescopes, a single
"petal piston" per region. A ``segmented_ptt`` corrector on a partitioned
pupil models the first directly (``wavefront_role: impose`` for a
disturbance, ``actuate`` for a controllable petal element) and the second
with ``piston_only: true``.

Algorithm
---------
1. Label the connected regions of ``transmission > threshold`` with
   :func:`scipy.ndimage.label` (4-connectivity, so a one-pixel diagonal
   spider still separates regions).
2. Keep the ``n_segments`` largest regions by pixel count. If more
   regions exist — a thin obstruction splitting one petal, an isolated
   speck — each extra region is merged into the kept region whose
   centroid is nearest (``merge_fragments=True``) or the partition is
   refused (``merge_fragments=False``). Fewer regions than requested is
   always an error: the spider does not partition the pupil as assumed.
3. Order the segments by ascending polar angle of their (post-merge)
   centroids, measured from the pupil-grid origin, so the actuator index
   is stable for a given pupil. The map from index to region is
   data-dependent; read :attr:`SegmentationResult.segment_coords` rather
   than assuming a compass convention.
4. Emit one binary ``hcipy.Field`` mask per segment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import hcipy
import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import label


@dataclass
class SegmentationResult:
    """Outputs of :func:`partition_connected_components`.

    Attributes
    ----------
    segments
        ``hcipy.ModeBasis`` of binary segment masks (1.0 inside, 0.0
        outside) on the pupil grid, in index order.
    segment_coords
        ``(n_segments, 2)`` centroids (pupil-plane coordinates) in the
        same order.
    merge_info
        One record per merged fragment: the fragment's raw label, size
        in pixels, centroid, and the index of the segment it joined.
    n_raw_regions
        Number of connected regions found before merging.
    """

    segments: Any
    segment_coords: NDArray[np.floating]
    merge_info: list[dict[str, Any]] = field(default_factory=list)
    n_raw_regions: int = 0


def partition_connected_components(
    pupil_grid: Any,
    aperture_field: Any,
    *,
    n_segments: int,
    merge_fragments: bool = True,
    threshold: float = 0.0,
) -> SegmentationResult:
    """Partition a pupil transmission map into ``n_segments`` connected regions.

    Parameters
    ----------
    pupil_grid
        The HCIPy pupil grid the field lives on (``make_pupil_grid``).
    aperture_field
        Pupil-plane transmission (any array raveling to the grid size).
    n_segments
        Number of segments to keep — the number of regions the spider
        is expected to cut the pupil into.
    merge_fragments
        Absorb extra regions into the nearest kept one (by centroid
        distance). When ``False``, extra regions raise ``ValueError``.
    threshold
        A pixel belongs to the pupil where ``transmission > threshold``.
        ``0.0`` counts every partially lit pixel; raise it (e.g. ``0.5``)
        when an anti-aliased thin spider would otherwise bridge regions.

    Raises
    ------
    ValueError
        Fewer than ``n_segments`` regions are found, or more are found
        and ``merge_fragments`` is ``False``.
    """
    n_segments = int(n_segments)
    if n_segments < 1:
        raise ValueError(f"n_segments must be >= 1, got {n_segments}")

    shape = tuple(int(s) for s in pupil_grid.shape)  # (ny, nx)
    lit_2d = np.asarray(aperture_field, dtype=float).reshape(shape) > float(threshold)
    labels_2d, n_raw = label(lit_2d)
    n_raw = int(n_raw)
    if n_raw < n_segments:
        raise ValueError(
            f"segmentation found {n_raw} connected region(s) in the aperture but "
            f"n_segments={n_segments} was requested; the spider does not partition "
            "the pupil into that many pieces (check the aperture, or the threshold)."
        )

    x_2d = np.asarray(pupil_grid.x).reshape(shape)
    y_2d = np.asarray(pupil_grid.y).reshape(shape)

    sizes = np.zeros(n_raw, dtype=int)
    centroids = np.zeros((n_raw, 2))
    for k in range(n_raw):
        m = labels_2d == k + 1
        sizes[k] = int(m.sum())
        centroids[k] = (x_2d[m].mean(), y_2d[m].mean())

    keep = np.argsort(sizes, kind="stable")[::-1][:n_segments]  # raw 0-based indices
    if n_raw > n_segments and not merge_fragments:
        extras = sorted(set(range(n_raw)) - set(int(k) for k in keep))
        raise ValueError(
            f"segmentation found {n_raw} connected regions but n_segments={n_segments}; "
            f"extra region(s) with sizes {[int(sizes[e]) for e in extras]} px would need "
            "merging (set merge_fragments: true) or a cleaner aperture."
        )

    # Map every raw region onto one of the kept regions.
    target_of = {int(k): int(k) for k in keep}
    merge_info: list[dict[str, Any]] = []
    for k in range(n_raw):
        if k in target_of:
            continue
        dists = np.linalg.norm(centroids[keep] - centroids[k], axis=1)
        nearest = int(keep[int(np.argmin(dists))])
        target_of[k] = nearest
        merge_info.append(
            {
                "fragment_label": k + 1,
                "fragment_size_px": int(sizes[k]),
                "fragment_centroid": (float(centroids[k, 0]), float(centroids[k, 1])),
                "merged_into_label": nearest + 1,
            }
        )

    merged_masks = {int(k): np.zeros(shape, dtype=bool) for k in keep}
    for k, tgt in target_of.items():
        merged_masks[tgt] |= labels_2d == k + 1

    merged_centroids = {
        k: (float(x_2d[m].mean()), float(y_2d[m].mean())) for k, m in merged_masks.items()
    }
    order = sorted(
        merged_masks,
        key=lambda k: float(np.arctan2(merged_centroids[k][1], merged_centroids[k][0])),
    )
    index_of_raw = {k: i for i, k in enumerate(order)}
    for rec in merge_info:
        rec["segment_index"] = index_of_raw[rec.pop("merged_into_label") - 1]

    fields = [hcipy.Field(merged_masks[k].ravel().astype(float), pupil_grid) for k in order]
    coords = np.array([merged_centroids[k] for k in order], dtype=float)
    return SegmentationResult(
        segments=hcipy.ModeBasis(fields, pupil_grid),
        segment_coords=coords,
        merge_info=merge_info,
        n_raw_regions=n_raw,
    )


__all__ = ["SegmentationResult", "partition_connected_components"]
