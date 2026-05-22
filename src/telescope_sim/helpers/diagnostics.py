"""Diagnostic plotting helpers for tutorials and interactive use.

These are convenience wrappers — nothing here is on the simulation hot path.
Functions here adopt a single visual convention: every PSF panel is paired
with the cumulative pupil-plane OPD that produced it, both with colorbars.
"""

from __future__ import annotations

from typing import Any

import hcipy
import numpy as np


def plot_opd_and_psfs(
    sim: Any,
    out: dict[str, Any],
    *,
    log_dynamic_range: float = 1e-8,
    suptitle: str | None = None,
    figsize: tuple[float, float] | None = None,
) -> Any:
    """Plot cumulative pupil OPD (left) and per-filter PSFs (right) side-by-side.

    Adopts the recommended diagnostic convention: every PSF is shown next to
    the wavefront state that produced it. The OPD panel uses
    :func:`hcipy.imshow_field` with the aperture as mask; PSFs use
    :class:`matplotlib.colors.LogNorm` so the colorbar labels carry the
    actual intensity values. Every panel gets a colorbar.

    Parameters
    ----------
    sim
        :class:`~telescope_sim.pipeline.TelescopeSim` instance. Used for
        ``sim.aperture.field`` (mask) and ``sim.focal_planes`` (filter
        labels).
    out
        Dict returned by ``sim.sample(meas_pupil_opd=True)``. Must contain
        ``images["psf"]`` and ``pupil_opd``. If ``strehls`` is also present
        (i.e. ``meas_strehl=True``), Strehl values annotate the per-filter
        titles.
    log_dynamic_range
        Ratio of ``vmin`` to ``vmax`` for the PSF colormap. Default
        ``1e-8`` (eight decades of dynamic range).
    suptitle
        Optional figure suptitle.
    figsize
        Override the auto-computed figure size. Default is
        ``(4 * (1 + n_filters), 4)``.

    Returns
    -------
    matplotlib.figure.Figure
        The constructed figure (not closed); caller can ``plt.show()`` or
        further customize.
    """
    # Import matplotlib lazily so the package doesn't pull it as a hard dep.
    import matplotlib.pyplot as plt  # noqa: PLC0415
    from matplotlib.colors import LogNorm  # noqa: PLC0415

    psf = out["images"]["psf"]
    n_filt = psf.shape[-1] if psf.ndim > 2 else 1
    filt_names = list(sim.focal_planes)

    if figsize is None:
        figsize = (4 * (1 + n_filt), 4)
    fig, axes = plt.subplots(1, 1 + n_filt, figsize=figsize)
    if 1 + n_filt == 1:
        axes = [axes]  # subplots returns a bare Axes when n=1

    # OPD panel — wavelength-independent, in nm.
    opd_nm = hcipy.Field(1e9 * np.asarray(out["pupil_opd"]), out["pupil_opd"].grid)
    plt.sca(axes[0])
    im0 = hcipy.imshow_field(opd_nm, mask=sim.aperture.field, cmap="RdBu_r")
    axes[0].set_title("cumulative pupil OPD")
    axes[0].set_axis_off()
    fig.colorbar(im0, ax=axes[0], shrink=0.7, label="OPD [nm]")

    # PSF panels — log intensity via LogNorm so colorbar shows raw values.
    for i in range(n_filt):
        ax = axes[i + 1]
        img = psf[..., i] if psf.ndim > 2 else psf
        vmax = float(np.nanmax(img))
        vmin = vmax * log_dynamic_range if vmax > 0 else 1.0
        im = ax.imshow(img, norm=LogNorm(vmin=vmin, vmax=vmax), cmap="inferno")
        title = filt_names[i] if i < len(filt_names) else f"channel {i}"
        if "strehls" in out and title in out["strehls"]:
            title = f"{title} — Strehl {out['strehls'][title]:.3f}"
        ax.set_title(title)
        ax.set_axis_off()
        fig.colorbar(im, ax=ax, shrink=0.7, label="intensity")

    if suptitle:
        fig.suptitle(suptitle)
    plt.tight_layout()
    return fig


__all__ = ["plot_opd_and_psfs"]
