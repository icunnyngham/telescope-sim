"""Render the 2024 and 2026 SCExAO/VAMPIRES F750 configurations side by side.

    python render.py [--png pupil_2024_vs_2026.png]

Builds both configs in this folder with telescope-sim, prints the values a
second machine should reproduce, and (with --png) saves the pupil amplitudes
and at-rest PSFs.
"""
import os
import sys
from pathlib import Path

import numpy as np
import yaml
from telescope_sim import TelescopeSim

# the configs' pupil `module` paths are relative to this folder
os.chdir(Path(__file__).resolve().parent)

CONFIGS = {
    "2024": "vampires_f750_2024.yaml",
    "2026 (provisional)": "vampires_f750_bench_2026.yaml",
}

rendered = {}
for label, path in CONFIGS.items():
    cfg = yaml.safe_load(open(path))
    sim = TelescopeSim.from_yaml(path)
    fp = sim.focal_planes["filter1"]
    n = cfg["pupil"]["resolution"]
    px = cfg["pupil"]["extent"] / n
    amp = np.asarray(sim.aperture.field, float)
    ref = np.asarray(fp.reference_psf)
    tilt = np.zeros(35)
    tilt[0] = 0.1
    out = sim.sample({"zernike_dm": tilt}, meas_strehl=True)
    image = out["images"]["psf"][..., 0]
    print(f"== {label}: {path}")
    print("  sum(amplitude), sum(amplitude^2) x pixel area [m^2]:", amp.sum() * px**2, (amp**2).sum() * px**2)
    print("  lit pixels, maximum amplitude:", int((amp > 0).sum()), amp.max())
    print("  wavelengths [nm]:", fp.lam_setup.filter_lams * 1e9)
    print("  pixel scale [mas]:", cfg["focal_planes"]["filter1"]["focal_extent"] / fp.focal_res * 1e3)
    print("  reference PSF peak, sum:", ref.max(), ref.sum())
    print("  0.1 of x tilt: peak pixel (iy, ix), peak Strehl:",
          tuple(int(i) for i in np.unravel_index(image.argmax(), image.shape)), out["strehls"]["filter1"])
    rendered[label] = (amp.reshape(n, n), ref, cfg["pupil"]["extent"] / 2)

if "--png" in sys.argv:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    fig, axes = plt.subplots(2, 2, figsize=(8.4, 8.0))
    for col, (label, (amp, ref, half)) in enumerate(rendered.items()):
        ax = axes[0, col]
        im = ax.imshow(amp, origin="lower", extent=[-half, half, -half, half], cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"{label}: pupil amplitude", fontsize=10, loc="left")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        fig.colorbar(im, ax=ax, shrink=0.8)
        ax = axes[1, col]
        im = ax.imshow(ref / ref.max(), origin="lower", cmap="inferno", norm=LogNorm(vmin=1e-5, vmax=1))
        ax.set_title(f"{label}: at-rest PSF / peak", fontsize=10, loc="left")
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    png = sys.argv[sys.argv.index("--png") + 1]
    fig.savefig(png, dpi=110)
    print("wrote", png)
