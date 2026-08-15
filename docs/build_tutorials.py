"""Build and execute the tutorial notebooks under ``docs/tutorials/``.

Each tutorial is defined in this file as a sequence of (cell_type,
source) pairs. The script builds them as nbformat NotebookNodes,
executes via :mod:`nbclient`, and writes them to disk with outputs.

Run from the repo root::

    conda activate telescope-sim-dev
    python docs/build_tutorials.py

Or only specific tutorials::

    python docs/build_tutorials.py 01_canonical_mini_elf
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "docs" / "tutorials"


def _md(text: str) -> tuple[str, str]:
    return ("markdown", text)


def _code(text: str) -> tuple[str, str]:
    return ("code", text)


# --- Tutorial definitions --------------------------------------------------

TUTORIALS: dict[str, list[tuple[str, str]]] = {
    "01_canonical_mini_elf": [
        _md(
            "# 1. The sELF array\n\n"
            "The simplest pipeline: the sELF (small-ELF) design — 15 circular "
            "0.5 m sub-apertures on a 1.5 m ring (3.5 m outer diameter), no "
            "atmosphere, no coronagraph. Loaded from the bundled `elf_15seg` "
            "preset."
        ),
        _md(
            "The `elf_15seg` preset is a small YAML config bundled with the package. "
            "It declares the pupil grid, aperture, one PTT corrector, the **sELF "
            "focal-plane wavefront-sensing band** (0.90–1.05 µm — 15.4% fractional "
            "bandwidth at 0.975 µm — sampled at 11 wavelengths), and a single "
            "intensity output. Let's read it directly so the shape of a real "
            "config is visible."
        ),
        _code(
            "from importlib.resources import files\n"
            "\n"
            'config_text = files("telescope_sim.presets").joinpath("elf_15seg.yaml").read_text()\n'
            "print(config_text)\n"
        ),
        _code(
            "import numpy as np\n"
            "import matplotlib.pyplot as plt\n"
            "from matplotlib.colors import LogNorm\n"
            "from telescope_sim import TelescopeSim\n"
            "from telescope_sim.helpers.diagnostics import plot_opd_and_psfs\n"
            "\n"
            'sim = TelescopeSim.from_preset("elf_15seg")\n'
            'print("correctors:", list(sim.correctors))\n'
            'print("focal planes:", list(sim.focal_planes))\n'
        ),
        _md(
            "By convention in these tutorials, every PSF is shown next to the "
            "**cumulative pupil OPD** that produced it — the wavefront state at "
            "the back of the corrector chain (and including any external "
            "atmosphere). Setting `meas_pupil_opd=True` on `sample()` returns "
            "`out['pupil_opd']` as an `hcipy.Field`; the helper "
            "`plot_opd_and_psfs` in `telescope_sim.helpers.diagnostics` handles "
            "the layout (OPD via `hcipy.imshow_field(mask=sim.aperture.field)`, "
            "PSFs via `imshow(norm=LogNorm(...))`, colorbars everywhere)."
        ),
        _md(
            "Sample at rest (no actuators applied). The pupil OPD is ~0 inside "
            "the aperture, and the PSF shows the segmented ring's signature: a "
            "broad envelope set by the 0.5 m segment size, modulated by the "
            "sharp interference structure of the 15-element ring — the array "
            "resolves like a 3.5 m telescope while collecting like fifteen "
            "0.5 m ones."
        ),
        _code(
            "out = sim.sample(meas_strehl=True, meas_pupil_opd=True)\n"
            "print('psf shape:', out['images']['psf'].shape)\n"
            "print('strehls:', out['strehls'])\n"
            "print('pupil_opd RMS (nm):', 1e9 * out['pupil_opd'].std())\n"
            "plot_opd_and_psfs(sim, out, suptitle='At-rest PSFs')\n"
            "plt.show()\n"
        ),
        _md(
            "Apply random per-segment piston/tip/tilt errors and resample. The "
            "OPD panel now shows the per-segment phase errors directly, and the "
            "PSF degrades accordingly — Strehl falls as "
            "`exp(-(2π σ_OPD / λ)²)`, so the same OPD error would cost even "
            "more at wavelengths shorter than this band."
        ),
        _code(
            "rng = np.random.default_rng(0)\n"
            "ptt = rng.normal(scale=0.1, size=(15, 3))\n"
            "out = sim.sample(\n"
            '    actuations={"segments": ptt}, meas_strehl=True, meas_pupil_opd=True,\n'
            ")\n"
            "print('strehls with errors:', out['strehls'])\n"
            "print('pupil_opd RMS (nm):', 1e9 * out['pupil_opd'].std())\n"
            "plot_opd_and_psfs(sim, out, suptitle='Random PTT — both filters')\n"
            "plt.show()\n"
        ),
        _md(
            "## Detector noise\n\n"
            "The `noisy_detector` post-processor wraps HCIPy's `NoisyDetector` "
            "(read noise + dark current + flat-field + optional photon shot noise). "
            "It's a single-focal-plane processor — its underlying detector needs "
            "exactly one focal grid, which the preset's single band satisfies. "
            "The demo restates the preset's sELF aperture + PTT corrector inline "
            "and stacks `noisy_detector` under the existing `intensity` tap.\n\n"
            "Per-sample photon flux is plumbed through "
            '`sim.sample(output_overrides={"psf": {"int_phot_flux": ...}})`: '
            "the same noisy sim covers a wide flux range without rebuilding."
        ),
        _code(
            "from telescope_sim.config.schema import SimConfig\n"
            "from telescope_sim.config.loader import build\n"
            "\n"
            "# The sELF preset setup restated inline, with noisy_detector stacked\n"
            "# under the intensity tap.\n"
            "noisy_cfg = {\n"
            "    'pupil': {'resolution': 256, 'extent': 3.675},\n"
            "    'aperture': {\n"
            "        'type': 'segmented_circular',\n"
            "        'segment_diameter': 0.5,\n"
            "        'layout': 'elf', 'n_segments': 15, 'ring_radius': 1.5,\n"
            "        'supersample': 16,\n"
            "    },\n"
            "    'correctors': {\n"
            "        'segments': {\n"
            "            'type': 'segmented_ptt',\n"
            "            'piston_scale': 1.0e-6, 'tip_tilt_scale': 1.0e-6,\n"
            "            'wavefront_role': 'actuate',\n"
            "            'target_strategy': 'actuators', 'target': True,\n"
            "        },\n"
            "    },\n"
            "    'corrector_chain': ['segments'],\n"
            "    'focal_planes': {\n"
            "        'filter1': {\n"
            "            'type': 'angular', 'central_lam': 0.975e-6,\n"
            "            'focal_extent': 3.2, 'focal_res': 640,\n"
            "            'fractional_bandwidth': 0.1538, 'num_samples': 11,\n"
            "        },\n"
            "    },\n"
            "    'outputs': {\n"
            "        'psf': {\n"
            "            'tap': {'type': 'intensity', 'focal_planes': ['filter1']},\n"
            "            'post_processing': [\n"
            "                {'type': 'noisy_detector',\n"
            "                 'int_phot_flux': 1.0e7,\n"
            "                 'detector': {\n"
            "                     'read_noise': 5.0,\n"
            "                     'dark_current_rate': 0.0,\n"
            "                     'flat_field': 0.0,\n"
            "                     'include_photon_noise': True,\n"
            "                 }},\n"
            "            ],\n"
            "        },\n"
            "    },\n"
            "}\n"
            "noisy_sim = build(SimConfig.model_validate(noisy_cfg))\n"
            "np.random.seed(0)\n"
            "out_clean = noisy_sim.sample()  # default flux = 1e7 photons/m^2\n"
            "out_low = noisy_sim.sample(output_overrides={'psf': {'int_phot_flux': 1.0e5}})\n"
            "out_high = noisy_sim.sample(output_overrides={'psf': {'int_phot_flux': 1.0e9}})\n"
            "\n"
            "fig, axes = plt.subplots(1, 3, figsize=(13, 4))\n"
            "for ax, out_i, title in zip(\n"
            "    axes,\n"
            "    [out_low, out_clean, out_high],\n"
            "    ['flux = 1e5', 'flux = 1e7 (YAML default)', 'flux = 1e9'],\n"
            "):\n"
            "    img = out_i['images']['psf'][..., 0]\n"
            "    vmax = float(img.max())\n"
            "    floor = max(vmax * 1e-4, 1e-3)\n"
            "    # Clip zero/negative counts to the LogNorm floor (they are 'bad'\n"
            "    # under LogNorm and would render as blank specks).\n"
            "    im = ax.imshow(\n"
            "        np.maximum(img, floor), norm=LogNorm(vmin=floor, vmax=vmax),\n"
            "        cmap='inferno',\n"
            "    )\n"
            "    ax.set_title(title)\n"
            "    ax.set_axis_off()\n"
            "    fig.colorbar(im, ax=ax, shrink=0.7, label='counts')\n"
            "fig.suptitle('noisy_detector PSFs at varying photon flux')\n"
            "plt.tight_layout()\n"
            "plt.show()\n"
        ),
        _md(
            "## Extended-source imaging via `convolve_image`\n\n"
            "The `convolve_image` post-processor convolves a caller-supplied scene "
            "with the focal-plane PSF (normalized by the at-rest reference PSF sum, "
            "so the kernel integrates to ~1). Stacked **before** `noisy_detector`, "
            "the chain is `intensity → convolve_image → noisy_detector` — the "
            "physically correct order for noisy extended-source imaging.\n\n"
            "Below the convolve scene is injected per-sample via `output_overrides`, "
            "so the same sim can imaging different scenes without rebuilding. To "
            "compare convolution alone vs. convolution + noise we build two sims "
            "(post-processing chains are fixed at sim-build time)."
        ),
        _code(
            "# Build a synthetic extended scene (three off-axis point sources +\n"
            "# a smooth background) at the same resolution as the focal grid.\n"
            "scene = np.zeros((640, 640), dtype=np.float64)\n"
            "ys, xs = np.indices(scene.shape)\n"
            "for cy, cx, amp, sig in [(225, 275, 3.0, 5.0),\n"
            "                          (350, 350, 1.5, 5.0),\n"
            "                          (300, 450, 0.8, 5.0)]:\n"
            "    scene += amp * np.exp(-((ys - cy) ** 2 + (xs - cx) ** 2) / (2 * sig ** 2))\n"
            "scene += 0.05  # uniform background\n"
            "\n"
            "# Sim A: convolve only. The 'image' field is omitted, so the scene\n"
            "# must come in per-sample via output_overrides.\n"
            "clean_cfg = {**noisy_cfg}\n"
            "clean_cfg['outputs'] = {\n"
            "    'psf': {\n"
            "        'tap': {'type': 'intensity', 'focal_planes': ['filter1']},\n"
            "        'post_processing': [\n"
            "            {'type': 'convolve_image'},\n"
            "        ],\n"
            "    },\n"
            "}\n"
            "clean_sim = build(SimConfig.model_validate(clean_cfg))\n"
            "out_clean_scene = clean_sim.sample(\n"
            "    output_overrides={'psf': {'convolve_image': scene}}\n"
            ")\n"
            "\n"
            "# Sim B: convolve + noise. Same per-sample scene injection.\n"
            "convolve_cfg = {**noisy_cfg}\n"
            "convolve_cfg['outputs'] = {\n"
            "    'psf': {\n"
            "        'tap': {'type': 'intensity', 'focal_planes': ['filter1']},\n"
            "        'post_processing': [\n"
            "            {'type': 'convolve_image'},\n"
            "            {'type': 'noisy_detector',\n"
            "             'int_phot_flux': 5.0e7,\n"
            "             'detector': {\n"
            "                 'read_noise': 5.0, 'dark_current_rate': 0.0,\n"
            "                 'flat_field': 0.0, 'include_photon_noise': True,\n"
            "             }},\n"
            "        ],\n"
            "    },\n"
            "}\n"
            "scene_sim = build(SimConfig.model_validate(convolve_cfg))\n"
            "np.random.seed(0)\n"
            "out_noisy_scene = scene_sim.sample(\n"
            "    output_overrides={'psf': {'convolve_image': scene}}\n"
            ")\n"
            "\n"
            "fig, axes = plt.subplots(1, 3, figsize=(13, 4))\n"
            "panels = [\n"
            "    (scene, 'input scene', 'scene intensity'),\n"
            "    (out_clean_scene['images']['psf'][..., 0],\n"
            "     'intensity → convolve_image', 'intensity'),\n"
            "    (out_noisy_scene['images']['psf'][..., 0],\n"
            "     'intensity → convolve → noisy_detector', 'counts'),\n"
            "]\n"
            "for ax, (img, title, cb_label) in zip(axes, panels):\n"
            "    im = ax.imshow(img, cmap='inferno')\n"
            "    ax.set_title(title)\n"
            "    ax.set_axis_off()\n"
            "    fig.colorbar(im, ax=ax, shrink=0.7, label=cb_label)\n"
            "plt.tight_layout()\n"
            "plt.show()\n"
        ),
    ],
    "02_coronagraph_vortex": [
        _md(
            "# 2. Coronagraphs: vortex and classical Lyot\n\n"
            "Drop a coronagraph into the chain. The reference PSF used for "
            "Strehl normalization is always generated with the coronagraph "
            "bypassed (matches the legacy convention). First a `vortex` "
            "coronagraph, loading the fixture config that reproduces the "
            "original VortexCoronagraph(2) setup against a Keck aperture "
            "(HCIPy built-in); then a classical `lyot` train on a real "
            "VAMPIRES instrument mode, on both compute backends."
        ),
        _code(
            "import hcipy\n"
            "import numpy as np\n"
            "import matplotlib.pyplot as plt\n"
            "from telescope_sim import TelescopeSim\n"
            "\n"
            'sim = TelescopeSim.from_yaml("fixtures/configs/07_coro_original.yaml")\n'
            "out = sim.sample(meas_pupil_opd=True)\n"
            'psf = out["images"]["psf"][..., 0]\n'
            'print("psf shape:", psf.shape, " peak/min:", psf.max(), psf.min())\n'
            "print('pupil_opd RMS (nm):', 1e9 * out['pupil_opd'].std())\n"
            "\n"
            "# Pupil OPD (left) is ~0 at rest — Keck aperture transmits perfectly\n"
            "# in-band. Even so, the coronagraph nulls the on-axis PSF (right).\n"
            "fig, axes = plt.subplots(1, 2, figsize=(10, 4))\n"
            "opd_nm = hcipy.Field(\n"
            "    1e9 * np.asarray(out['pupil_opd']), out['pupil_opd'].grid\n"
            ")\n"
            "plt.sca(axes[0])\n"
            "im0 = hcipy.imshow_field(\n"
            "    opd_nm, mask=sim.aperture.field, cmap='RdBu_r'\n"
            ")\n"
            "axes[0].set_title('cumulative pupil OPD')\n"
            "axes[0].set_axis_off()\n"
            "fig.colorbar(im0, ax=axes[0], shrink=0.7, label='OPD [nm]')\n"
            "im1 = axes[1].imshow(psf, cmap='inferno')\n"
            "axes[1].set_title('Vortex coronagraph @ rest (Keck, charge=2)')\n"
            "axes[1].set_axis_off()\n"
            "fig.colorbar(im1, ax=axes[1], shrink=0.7, label='post-coro intensity')\n"
            "plt.tight_layout()\n"
            "plt.show()\n"
        ),
        _md(
            "## Classical Lyot coronagraph\n\n"
            "The `lyot` kind is a classical Lyot train: a hard-edged occulting "
            "spot in an intermediate focal plane plus an undersized/oversized "
            "Lyot stop back in the pupil. This config expresses a real "
            "instrument mode — VAMPIRES' CLC-3 spot (69 µm → 0.127 arcsec on "
            "sky) with the SCExAO pupil and its matching parametric Lyot stop. "
            "We clear the config's `per_sample_norm` so the coronagraphic image "
            "stays in the same physical units as the reference PSF and the "
            "suppression is directly visible."
        ),
        _code(
            "import yaml\n"
            "from matplotlib.colors import LogNorm\n"
            "from telescope_sim.config.loader import build\n"
            "from telescope_sim.config.schema import SimConfig\n"
            "\n"
            'with open("fixtures/configs/12_vampires_lyot.yaml") as f:\n'
            "    raw = yaml.safe_load(f)\n"
            'raw["outputs"]["psf"]["post_processing"] = []  # keep physical units\n'
            "config = SimConfig.model_validate(raw)\n"
            "\n"
            "sim = build(config)\n"
            "coro = sim.sample()['images']['psf'][..., 0]\n"
            "ref = sim.focal_planes['filter1'].reference_psf\n"
            "peak = ref.max()\n"
            "print(f'core suppression: {coro.max() / peak:.2e}')\n"
            "print(f'energy transmitted: {coro.sum() / ref.sum():.2%}')\n"
            "\n"
            "fig, axes = plt.subplots(1, 3, figsize=(14, 4))\n"
            "stop_field = sim.coronagraph.lyot_field\n"
            "plt.sca(axes[0])\n"
            "hcipy.imshow_field(sim.aperture.field + 0.5 * stop_field, cmap='gray')\n"
            "axes[0].set_title('SCExAO pupil + Lyot stop (overlay)')\n"
            "axes[0].set_axis_off()\n"
            "norm = LogNorm(vmin=peak * 1e-8, vmax=peak)\n"
            "for ax, img, title in (\n"
            "    (axes[1], ref, 'reference PSF (no coronagraph)'),\n"
            "    (axes[2], coro, 'CLC-3 + Lyot stop'),\n"
            "):\n"
            "    im = ax.imshow(img, cmap='inferno', norm=norm)\n"
            "    ax.set_title(title)\n"
            "    ax.set_axis_off()\n"
            "    fig.colorbar(im, ax=ax, shrink=0.7)\n"
            "plt.tight_layout()\n"
            "plt.show()\n"
        ),
        _md(
            "### Dual-backend: the coronagraph train is in the JAX graph\n\n"
            "Every coronagraph kind — `lyot` shown here, and the vortex kinds "
            "too — runs on both compute backends: the same config on "
            "`backend='jax'` folds the coronagraph train into the propagation "
            "kernels, so `sample()`, `forward_fn`, and `sample_batch` all "
            "include it — batched, jitted, and differentiable. The two "
            "backends share the exact same geometry arrays and agree to "
            "float64 round-off."
        ),
        _code(
            "import jax\n"
            "import jax.numpy as jnp\n"
            "\n"
            "sim_jax = build(config, backend='jax')\n"
            "coro_jax = sim_jax.sample()['images']['psf'][..., 0]\n"
            "print(f'max |hcipy - jax| (peak-normalized): '\n"
            "      f'{abs(coro - coro_jax).max() / peak:.2e}')\n"
            "\n"
            "# Differentiable through the coronagraph: gradient of the residual\n"
            "# core flux with respect to the Zernike actuations.\n"
            "fwd = sim_jax.forward_fn()\n"
            "c = config.focal_planes['filter1'].focal_res // 2\n"
            "\n"
            "def core_flux(acts):\n"
            "    img = fwd(acts)['filter1']\n"
            "    return img[c - 6 : c + 6, c - 6 : c + 6].sum()\n"
            "\n"
            "grad = jax.grad(core_flux)({'zernike_dm': jnp.zeros(35)})\n"
            "print('d(core flux)/d(zernike) shape:', grad['zernike_dm'].shape)\n"
            "print('largest-sensitivity modes:', jnp.argsort(-abs(grad['zernike_dm']))[:5])\n"
        ),
    ],
    "03_vampires_zernike": [
        _md(
            "# 3. Custom pupil + Zernike DM (VAMPIRES)\n\n"
            "Wraps an external pupil-generation function (`miles_pupil`) via the "
            "`external_pupil` aperture, and drives a Zernike-mode deformable "
            "mirror. The fixture config reproduces the 2023-10 VAMPIRES base "
            "setup against the captured digest."
        ),
        _code(
            "import numpy as np\n"
            "import matplotlib.pyplot as plt\n"
            "from matplotlib.colors import LogNorm\n"
            "from telescope_sim import TelescopeSim\n"
            "from telescope_sim.helpers.diagnostics import plot_opd_and_psfs\n"
            "\n"
            'sim = TelescopeSim.from_yaml("fixtures/configs/09_vampires_base.yaml")\n'
            "out = sim.sample(meas_strehl=True, meas_pupil_opd=True)\n"
            "\n"
            "plot_opd_and_psfs(sim, out, suptitle='VAMPIRES @ rest')\n"
            "plt.show()\n"
        ),
        _md(
            "Push a few Zernike modes and see the focal plane respond — the OPD "
            "panel makes the mode shape directly visible alongside the resulting PSF."
        ),
        _code(
            "amps = np.zeros(10)\n"
            "amps[3] = 0.3   # one of the low-order modes\n"
            "out = sim.sample(\n"
            '    actuations={"zernike_dm": amps},\n'
            "    meas_strehl=True, meas_pupil_opd=True,\n"
            ")\n"
            "plot_opd_and_psfs(sim, out, suptitle='One Zernike mode pushed')\n"
            "plt.show()\n"
        ),
        _md(
            "## Atmosphere as a per-sample input\n\n"
            "Atmospheres are **not** correctors — they flow into the pipeline as the "
            "`atmos=` kwarg on `sample()`. Any wf→wf callable works; HCIPy's\n"
            "`InfiniteAtmosphericLayer` is the typical case. The caller owns time "
            "evolution (call `atmos.evolve_until(t)` between samples); `telescope-sim` "
            "stays stateless.\n\n"
            "If the atmosphere also exposes `phase_for(lam)`, its OPD seeds the "
            "cumulative-OPD stream that downstream **fit-role** correctors consume — "
            'so a DM declared with `wavefront_role="fit"` and '
            '`fit_source="cumulative_phase_pre_self"` will auto-fit (and cancel) the '
            "atmosphere. No bespoke atmosphere-corrector required."
        ),
        _code(
            "import hcipy\n"
            "\n"
            "# Single-layer Kolmogorov turbulence on the existing pupil grid. The\n"
            "# Fried parameter r0 is set generously (1.5m at 760nm — well above the\n"
            "# typical seeing limit) so the 10-mode Zernike basis can do most of\n"
            "# the work; the point of this demo is the wiring, not state-of-the-art\n"
            "# AO performance.\n"
            "pupil_grid = sim.aperture.field.grid\n"
            "wavelength = 0.760e-6\n"
            "fried = 1.5\n"
            "Cn2 = hcipy.Cn_squared_from_fried_parameter(fried, wavelength)\n"
            "atmos = hcipy.InfiniteAtmosphericLayer(\n"
            "    pupil_grid, Cn2, L0=20.0, velocity=10.0, height=0.0,\n"
            ")\n"
            "atmos.evolve_until(0.0)\n"
            "print('atmos exposes phase_for? ', hasattr(atmos, 'phase_for'))\n"
            "\n"
            "out_clean = sim.sample(meas_strehl=True, meas_pupil_opd=True)\n"
            "out_atmos = sim.sample(atmos=atmos, meas_strehl=True, meas_pupil_opd=True)\n"
            'print(f\'Strehl no atmos:  {out_clean["strehls"]["filter1"]:.3f}\')\n'
            'print(f\'Strehl atmos on: {out_atmos["strehls"]["filter1"]:.3f}\')\n'
            "print(f'atmos OPD RMS:   {1e9 * out_atmos[\"pupil_opd\"].std():.1f} nm')\n"
            "\n"
            "plot_opd_and_psfs(sim, out_clean, suptitle='no atmos')\n"
            "plot_opd_and_psfs(sim, out_atmos, suptitle='atmos applied')\n"
            "plt.show()\n"
        ),
        _md(
            "Now build a sim variant with the Zernike DM in **fit-role**, fitting "
            "`cumulative_phase_pre_self`. With the same atmosphere injected, the DM "
            "auto-cancels it — no actuator commands required from the caller. "
            '`target_strategy="actuators_plus_residual_fit"` makes `out["actuations"]` '
            "report the actuator values needed; for ML training, this is the natural "
            "supervision signal."
        ),
        _code(
            "import yaml\n"
            "from pathlib import Path\n"
            "from telescope_sim.config.schema import SimConfig\n"
            "from telescope_sim.config.loader import build\n"
            "\n"
            "fit_cfg = yaml.safe_load(\n"
            "    Path('fixtures/configs/09_vampires_base.yaml').read_text()\n"
            ")\n"
            "# Flip the existing zernike_dm into a fit-role corrector that fits\n"
            "# whatever cumulative OPD it sees upstream (here: just atmos).\n"
            "fit_cfg['correctors']['zernike_dm'].update({\n"
            "    'n_modes': 30,                # more horsepower than the 10-mode default\n"
            "    'starting_mode': 2,\n"
            "    'wavefront_role': 'fit',\n"
            "    'target_strategy': 'actuators_plus_residual_fit',\n"
            "    'fit_source': 'cumulative_phase_pre_self',\n"
            "    'target': True,\n"
            "})\n"
            "fit_sim = build(SimConfig.model_validate(fit_cfg))\n"
            "\n"
            "atmos.evolve_until(0.0)  # reset the layer to the same frozen phase\n"
            "out_corrected = fit_sim.sample(\n"
            "    atmos=atmos, meas_strehl=True, meas_pupil_opd=True,\n"
            ")\n"
            "print(f\"Strehl with fit-role DM: {out_corrected['strehls']['filter1']:.3f}\")\n"
            "print('fit values shape:', out_corrected['actuations']['zernike_dm'].shape)\n"
            "print(f'residual OPD RMS: {1e9 * out_corrected[\"pupil_opd\"].std():.1f} nm')\n"
            "\n"
            "# Three side-by-side OPD+PSF panels — clean / atmos uncorrected /\n"
            "# atmos corrected. The OPD panels make the recovery story explicit:\n"
            "# atmospheric phase in panel 2 nearly vanishes in panel 3.\n"
            "plot_opd_and_psfs(sim, out_clean, suptitle='no atmos')\n"
            "plot_opd_and_psfs(sim, out_atmos, suptitle='atmos, no correction')\n"
            "plot_opd_and_psfs(fit_sim, out_corrected, suptitle='atmos + fit-role DM')\n"
            "plt.show()\n"
        ),
    ],
    "04_fiber_mmf": [
        _md(
            "# 4. Fiber MMF coupling\n\n"
            "Demonstrates the `physical` focal plane (focal grid in metres, with "
            "an explicit `focal_length`) paired with the `fiber_dual` output tap. "
            "The tap produces a `(2, H, W, 1)` stack: focal-plane intensity in "
            "channel 0 and multi-mode fiber-coupled intensity in channel 1.\n\n"
            "This tutorial uses a deliberately small fiber setup (32-pixel focal "
            "plane, 3 wavelengths, smaller fiber core) so it executes in seconds. "
            "The full-fidelity reproduction lives at "
            "`fixtures/configs/15_fiber_mmf.yaml` and is exercised by "
            "`pytest --runslow tests/fixtures/`."
        ),
        _code(
            "import numpy as np\n"
            "import matplotlib.pyplot as plt\n"
            "import yaml\n"
            "from telescope_sim import TelescopeSim\n"
            "from telescope_sim.config.schema import SimConfig\n"
            "from telescope_sim.config.loader import build\n"
            "\n"
            "config = {\n"
            "    'pupil': {'resolution': 128, 'extent': 3.675},\n"
            "    'aperture': {\n"
            "        'type': 'segmented_circular',\n"
            "        'segment_diameter': 3.5,\n"
            "        'layout': 'custom',\n"
            "        'positions': [[0.0, 0.0]],\n"
            "    },\n"
            "    'correctors': {\n"
            "        'zernike_dm': {\n"
            "            'type': 'zernike', 'n_modes': 10, 'zernike_diameter': 3.5,\n"
            "            'starting_mode': 1, 'actuate_scale': 1e-6,\n"
            "            'wavefront_role': 'actuate',\n"
            "            'target_strategy': 'actuators', 'target': True,\n"
            "        }\n"
            "    },\n"
            "    'corrector_chain': ['zernike_dm'],\n"
            "    'focal_planes': {\n"
            "        'filter1': {\n"
            "            'type': 'physical',\n"
            "            'central_lam': 6.35e-7, 'focal_extent': 5.25e-4,\n"
            "            'focal_res': 32, 'focal_length': 32.5,\n"
            "            'fractional_bandwidth': 0.001574803, 'num_samples': 3,\n"
            "            'wavefront_total_power': 1.0,\n"
            "        }\n"
            "    },\n"
            "    'outputs': {\n"
            "        'x': {\n"
            "            'tap': {\n"
            "                'type': 'fiber_dual', 'focal_plane_name': 'filter1',\n"
            "                'fiber': {\n"
            "                    'type': 'step_index', 'core_radius': 1.0e-4,\n"
            "                    'NA': 0.1, 'fiber_length': 7.4e-3, 'max_in_cache': 3,\n"
            "                },\n"
            "            },\n"
            "            'post_processing': [],\n"
            "        }\n"
            "    },\n"
            "}\n"
            "import hcipy\n"
            "from matplotlib.colors import LogNorm\n"
            "\n"
            "sim = build(SimConfig.model_validate(config))\n"
            "out = sim.sample(meas_pupil_opd=True)\n"
            "stack = out['images']['x']   # (2, H, W, 1)\n"
            "focal_psf, mmf_psf = stack[0, ..., 0], stack[1, ..., 0]\n"
            "print('focal shape:', focal_psf.shape, ' mmf shape:', mmf_psf.shape)\n"
            "print('pupil_opd RMS (nm):', 1e9 * out['pupil_opd'].std())\n"
            "\n"
            "# Custom layout because the fiber_dual tap stacks channels [focal, mmf]\n"
            "# rather than per-filter — the package's plot_opd_and_psfs helper\n"
            "# assumes the per-filter stack from the intensity tap.\n"
            "fig, axes = plt.subplots(1, 3, figsize=(13, 4))\n"
            "opd_nm = hcipy.Field(\n"
            "    1e9 * np.asarray(out['pupil_opd']), out['pupil_opd'].grid\n"
            ")\n"
            "plt.sca(axes[0])\n"
            "im0 = hcipy.imshow_field(\n"
            "    opd_nm, mask=sim.aperture.field, cmap='RdBu_r'\n"
            ")\n"
            "axes[0].set_title('cumulative pupil OPD')\n"
            "axes[0].set_axis_off()\n"
            "fig.colorbar(im0, ax=axes[0], shrink=0.7, label='OPD [nm]')\n"
            "for ax, img, title in [\n"
            "    (axes[1], focal_psf, 'focal-plane intensity'),\n"
            "    (axes[2], mmf_psf, 'multi-mode fiber coupling'),\n"
            "]:\n"
            "    vmax = float(img.max())\n"
            "    vmin = vmax * 1e-8 if vmax > 0 else 1.0\n"
            "    im = ax.imshow(img, norm=LogNorm(vmin=vmin, vmax=vmax), cmap='inferno')\n"
            "    ax.set_title(title)\n"
            "    ax.set_axis_off()\n"
            "    fig.colorbar(im, ax=ax, shrink=0.7, label='intensity')\n"
            "plt.tight_layout()\n"
            "plt.show()\n"
        ),
    ],
    "05_custom_components": [
        _md(
            "# 5. Extending telescope-sim with custom components\n\n"
            "Every pipeline stage in `telescope-sim` — apertures, correctors, "
            "coronagraphs, focal planes, output taps, post-processors — is a "
            "**registered** implementation of a small ABC. To add a new component, "
            "subclass the appropriate ABC, decorate the class with "
            '`@register("<kind>", "<name>")`, and import the module so the '
            "decorator runs. The YAML/Python config then references it by name.\n\n"
            "This tutorial walks through two concrete examples of the extension "
            "pattern:\n\n"
            "- **Part 1** — a custom `XineticsDM` corrector that auto-fits the\n"
            "  surface of a stock `segmented_ptt` corrector (the legacy\n"
            "  `aprox_ptt_with_dm` workflow).\n"
            "- **Part 2** — a custom `PowScale` post-processor for dynamic-range\n"
            "  compression of PSF images (the legacy `extra_processing.pow_scale`\n"
            "  field).\n\n"
            "> **Note** (v2.1.0): the capability Part 1 builds — an actuator-grid\n"
            "> DM with Xinetics influence functions — now ships in-package as the\n"
            "> `actuator_grid` corrector (tutorial 6), which since v2.2.0 also\n"
            "> implements `fit_surface` (aperture-masked regularized least squares\n"
            "> onto the influence basis), so it can run fit-role / residual-fit\n"
            "> configurations directly. Part 1 remains the reference walkthrough\n"
            "> for the extension pattern itself.\n\n"
            "Both follow the same recipe: write a class, decorate, reference by "
            "name. The same pattern extends to all six pluggable kinds."
        ),
        _md(
            "## Part 1 — Custom Corrector: a Xinetics DM\n\n"
            "HCIPy ships `make_xinetics_influence_functions` for a regular Xinetics "
            "actuator grid. The `Corrector` ABC needs `apply`, `set_actuators`, "
            "`n_actuators`, `actuators`, and — for fit-role usage — `fit_surface`. "
            "Optionally implement `_bind_pupil_grid` to defer HCIPy DM construction "
            "until the loader has a pupil grid (so the class can be instantiated "
            "before sim build).\n\n"
            "The structure below mirrors the package's existing `ZernikeCorrector` "
            "almost line-for-line: same lstsq-on-aperture-masked-pixels fit, same "
            "OPD-vs-surface convention (`fit_surface` returns *matching* "
            "caller-facing actuator values; the pipeline negates at the apply site "
            'for `wavefront_role="fit"`).'
        ),
        _code(
            "import hcipy\n"
            "import numpy as np\n"
            "import matplotlib.pyplot as plt\n"
            "\n"
            "from telescope_sim.abc import Corrector\n"
            "from telescope_sim.registry import register\n"
            "\n"
            "\n"
            "@register('corrector', 'xinetics_dm')\n"
            "class XineticsDM(Corrector):\n"
            "    '''Xinetics-style DM with lstsq fit_surface, for fit-role usage.'''\n"
            "\n"
            "    def __init__(\n"
            "        self,\n"
            "        num_actuators_across_pupil,\n"
            "        actuator_spacing,\n"
            "        *,\n"
            "        actuate_scale=1.0,\n"
            "        name='xinetics_dm',\n"
            "        wavefront_role='actuate',\n"
            "        target_strategy='none',\n"
            "        fit_source=None,\n"
            "        target=False,\n"
            "    ):\n"
            "        self.name = name\n"
            "        self.num_actuators_across_pupil = int(num_actuators_across_pupil)\n"
            "        self.actuator_spacing = float(actuator_spacing)\n"
            "        self.actuate_scale = float(actuate_scale)\n"
            "        self.wavefront_role = wavefront_role\n"
            "        self.target_strategy = target_strategy\n"
            "        self.fit_source = fit_source\n"
            "        self.target = target\n"
            "        self._dm = None\n"
            "        self._basis_matrix = None\n"
            "        self._aperture_mask = None\n"
            "\n"
            "    def _bind_pupil_grid(self, pupil_grid, aperture_field):\n"
            "        basis = hcipy.make_xinetics_influence_functions(\n"
            "            pupil_grid, self.num_actuators_across_pupil, self.actuator_spacing\n"
            "        )\n"
            "        self._dm = hcipy.DeformableMirror(basis)\n"
            "        self._basis_matrix = np.column_stack(\n"
            "            [np.asarray(m, dtype=float).ravel() for m in basis]\n"
            "        )\n"
            "        self._aperture_mask = np.asarray(aperture_field, dtype=float).ravel() > 0\n"
            "\n"
            "    def apply(self, wf):\n"
            "        return self._dm(wf)\n"
            "\n"
            "    def set_actuators(self, values):\n"
            "        arr = np.asarray(values, dtype=float).reshape(-1)\n"
            "        self._dm.actuators = arr * self.actuate_scale\n"
            "\n"
            "    def flatten(self):\n"
            "        if self._dm is not None:\n"
            "            self._dm.actuators = np.zeros(self.n_actuators)\n"
            "\n"
            "    def fit_surface(self, phase):\n"
            "        phase = np.asarray(phase, dtype=float).ravel()\n"
            "        phase = phase - phase[self._aperture_mask].mean()\n"
            "        B = self._basis_matrix[self._aperture_mask]\n"
            "        rhs = phase[self._aperture_mask]\n"
            "        amps, _, _, _ = np.linalg.lstsq(B, rhs, rcond=None)\n"
            "        return amps / (2.0 * self.actuate_scale)\n"
            "\n"
            "    @property\n"
            "    def n_actuators(self):\n"
            "        return self.num_actuators_across_pupil ** 2\n"
            "\n"
            "    @property\n"
            "    def actuators(self):\n"
            "        if self._dm is None:\n"
            "            return np.zeros(self.n_actuators)\n"
            "        return np.asarray(self._dm.actuators) / self.actuate_scale\n"
            "\n"
            "print('xinetics_dm registered')\n"
        ),
        _md(
            "Build two sims sharing the same single-segment circular aperture and "
            "a stock `segmented_ptt` corrector. Sim A has only the PTT corrector; "
            "Sim B adds the new `xinetics_dm` downstream as a fit-role corrector "
            'with `fit_source="cumulative_phase_pre_self"`, so it auto-fits '
            "(and cancels) whatever the PTT corrector imposed.\n\n"
            '`target_strategy="residual_fit_only"` makes `out["actuations"]["xinetics_dm"]` '
            "report the actuator values needed — the legacy `aprox_ptt_with_dm` "
            "output."
        ),
        _code(
            "from telescope_sim.config.schema import SimConfig\n"
            "from telescope_sim.config.loader import build\n"
            "\n"
            "base_cfg = {\n"
            "    'pupil': {'resolution': 256, 'extent': 1.05},\n"
            "    'aperture': {\n"
            "        'type': 'segmented_circular',\n"
            "        'segment_diameter': 1.0,\n"
            "        'layout': 'custom',\n"
            "        'positions': [[0.0, 0.0]],\n"
            "        'supersample': 16,\n"
            "    },\n"
            "    'correctors': {\n"
            "        'segments': {\n"
            "            'type': 'segmented_ptt',\n"
            "            'piston_scale': 1.0e-6, 'tip_tilt_scale': 1.0e-6,\n"
            "            'wavefront_role': 'impose',\n"
            "            'target_strategy': 'actuators', 'target': True,\n"
            "        },\n"
            "    },\n"
            "    'corrector_chain': ['segments'],\n"
            "    'focal_planes': {\n"
            "        'filter1': {\n"
            "            'type': 'angular', 'central_lam': 1.0e-6,\n"
            "            'focal_extent': 2.0, 'focal_res': 128,\n"
            "            'fractional_bandwidth': 0.0, 'num_samples': 1,\n"
            "        },\n"
            "    },\n"
            "    'outputs': {\n"
            "        'psf': {\n"
            "            'tap': {'type': 'intensity', 'focal_planes': ['filter1']},\n"
            "            'post_processing': [{'type': 'max_intensity_norm'}],\n"
            "        },\n"
            "    },\n"
            "}\n"
            "ptt_only_sim = build(SimConfig.model_validate(base_cfg))\n"
            "\n"
            "# Sim B adds xinetics_dm downstream as a fit-role corrector.\n"
            "ptt_plus_dm_cfg = {**base_cfg}\n"
            "ptt_plus_dm_cfg['correctors'] = {\n"
            "    **base_cfg['correctors'],\n"
            "    'xinetics_dm': {\n"
            "        'type': 'xinetics_dm',\n"
            "        'num_actuators_across_pupil': 12,\n"
            "        'actuator_spacing': 1.0 / 12,\n"
            "        'actuate_scale': 1.0e-6,\n"
            "        'wavefront_role': 'fit',\n"
            "        'fit_source': 'cumulative_phase_pre_self',\n"
            "        'target_strategy': 'residual_fit_only',\n"
            "        'target': True,\n"
            "    },\n"
            "}\n"
            "ptt_plus_dm_cfg['corrector_chain'] = ['segments', 'xinetics_dm']\n"
            "ptt_plus_dm_sim = build(SimConfig.model_validate(ptt_plus_dm_cfg))\n"
            "print('sim A correctors:', list(ptt_only_sim.correctors))\n"
            "print('sim B correctors:', list(ptt_plus_dm_sim.correctors))\n"
        ),
        _md(
            "Push a non-trivial PTT on the segmented mirror and sample both sims. "
            "The fit-role DM in Sim B should auto-cancel the imposed PTT."
        ),
        _code(
            "ptt = np.array([[0.2, 1.5, 0.7]])  # piston=0.2um, tip=1.5um-rad, tilt=0.7um-rad\n"
            "actuations = {'segments': ptt}\n"
            "\n"
            "out_ptt = ptt_only_sim.sample(actuations=actuations, meas_strehl=True)\n"
            "out_corr = ptt_plus_dm_sim.sample(actuations=actuations, meas_strehl=True)\n"
            "print(f\"Strehl, PTT only:  {out_ptt['strehls']['filter1']:.3f}\")\n"
            "print(f\"Strehl, PTT + DM:  {out_corr['strehls']['filter1']:.3f}\")\n"
            "print('xinetics_dm fit actuator vector shape:',\n"
            "      out_corr['actuations']['xinetics_dm'].shape)\n"
        ),
        _md(
            "Visualize the pupil-plane OPDs: the PTT-imposed OPD, the DM's "
            "matching fit, and the residual after fit. The fit-role DM applies "
            "`-matching_amps` to cancel; the residual is what's left."
        ),
        _code(
            "# Reach into the sim's internals to pull the actual surfaces.\n"
            "# (Tutorial code can do this; library code shouldn't.)\n"
            "segments_corr = ptt_plus_dm_sim.correctors['segments']\n"
            "xinetics_corr = ptt_plus_dm_sim.correctors['xinetics_dm']\n"
            "shape2d = (256, 256)\n"
            "\n"
            "ptt_opd = 2.0 * np.asarray(segments_corr._sm.surface).reshape(shape2d)\n"
            "# After fit-role apply: dm.surface = -matching_surface; DM OPD = -matching_opd.\n"
            "# Flip sign for the 'reconstruction' display so it visibly matches PTT.\n"
            "dm_opd_match = -2.0 * np.asarray(xinetics_corr._dm.surface).reshape(shape2d)\n"
            "residual_opd = ptt_opd - dm_opd_match\n"
            "aperture_mask = (\n"
            "    np.asarray(ptt_plus_dm_sim.aperture.field).reshape(shape2d) > 0\n"
            ")\n"
            "# Common color scale for the OPD panels (in nm).\n"
            "vmax = 1e9 * np.max(np.abs(ptt_opd[aperture_mask]))\n"
            "\n"
            "fig, axes = plt.subplots(1, 3, figsize=(13, 4))\n"
            "for ax, opd, title in zip(\n"
            "    axes,\n"
            "    [ptt_opd, dm_opd_match, residual_opd],\n"
            "    ['PTT-imposed OPD', 'DM fit (matching)', 'residual after fit'],\n"
            "):\n"
            "    masked = np.where(aperture_mask, 1e9 * opd, np.nan)\n"
            "    im = ax.imshow(masked, cmap='RdBu_r', vmin=-vmax, vmax=vmax)\n"
            "    ax.set_title(title)\n"
            "    ax.set_axis_off()\n"
            "    fig.colorbar(im, ax=ax, shrink=0.7, label='OPD [nm]')\n"
            "plt.tight_layout()\n"
            "plt.show()\n"
            "\n"
            "rms_ptt = 1e9 * np.std(ptt_opd[aperture_mask])\n"
            "rms_res = 1e9 * np.std(residual_opd[aperture_mask])\n"
            "print(f'PTT OPD RMS:      {rms_ptt:.1f} nm')\n"
            "print(f'residual OPD RMS: {rms_res:.1f} nm')\n"
        ),
        _md(
            "And the focal-plane PSFs side-by-side. (Sharing one LogNorm across "
            "both panels keeps the colorbars directly comparable.)"
        ),
        _code(
            "from matplotlib.colors import LogNorm\n"
            "\n"
            "psf_ptt = out_ptt['images']['psf'][..., 0]\n"
            "psf_corr = out_corr['images']['psf'][..., 0]\n"
            "vmax = float(max(psf_ptt.max(), psf_corr.max()))\n"
            "norm = LogNorm(vmin=vmax * 1e-6, vmax=vmax)\n"
            "\n"
            "fig, axes = plt.subplots(1, 2, figsize=(10, 4))\n"
            "for ax, img, title in [\n"
            "    (axes[0], psf_ptt,\n"
            "     f\"PTT only — Strehl {out_ptt['strehls']['filter1']:.3f}\"),\n"
            "    (axes[1], psf_corr,\n"
            "     f\"PTT + DM — Strehl {out_corr['strehls']['filter1']:.3f}\"),\n"
            "]:\n"
            "    im = ax.imshow(img, norm=norm, cmap='inferno')\n"
            "    ax.set_title(title)\n"
            "    ax.set_axis_off()\n"
            "    fig.colorbar(im, ax=ax, shrink=0.7, label='intensity')\n"
            "plt.tight_layout()\n"
            "plt.show()\n"
        ),
        _md(
            "## Part 2 — Custom PostProcessor: `PowScale`\n\n"
            "Post-processors are simpler: one method (`__call__`) on the "
            "`PostProcessor` ABC. The processor receives the current image and a "
            "`PipelineContext` carrying reference values and per-sample overrides. "
            "We use the overrides channel here so the same registered class works "
            "with either a YAML-default `power` or a per-sample `power` override.\n\n"
            "This re-creates the legacy `extra_processing.pow_scale` field — useful "
            "for compressing the dynamic range of training data (square-root or "
            "fourth-root of the PSF surfaces both peak and wings together)."
        ),
        _code(
            "from telescope_sim.abc import PostProcessor\n"
            "\n"
            "\n"
            "@register('post_processor', 'pow_scale')\n"
            "class PowScale(PostProcessor):\n"
            "    name = 'pow_scale'\n"
            "\n"
            "    def __init__(self, power=1.0):\n"
            "        self.power = float(power)\n"
            "\n"
            "    def __call__(self, image, context):\n"
            "        # Per-sample override beats YAML default. Clamp to avoid\n"
            "        # NaNs from non-integer powers of tiny-negative noise pixels.\n"
            "        power = float(context.overrides.get('power', self.power))\n"
            "        return np.clip(image, 0.0, None) ** power\n"
            "\n"
            "print('pow_scale registered')\n"
        ),
        _md(
            "Build a sim with `pow_scale` (power=0.5 from YAML) layered after "
            "`max_intensity_norm`. Then sweep `power` per-sample via "
            "`output_overrides`."
        ),
        _code(
            "pow_cfg = {**base_cfg}\n"
            "pow_cfg['outputs'] = {\n"
            "    'psf': {\n"
            "        'tap': {'type': 'intensity', 'focal_planes': ['filter1']},\n"
            "        'post_processing': [\n"
            "            {'type': 'max_intensity_norm'},\n"
            "            {'type': 'pow_scale', 'power': 0.5},\n"
            "        ],\n"
            "    },\n"
            "}\n"
            "pow_sim = build(SimConfig.model_validate(pow_cfg))\n"
            "\n"
            "# At-rest PSF — the centered Airy disk shows pow_scale's dynamic-range\n"
            "# compression most clearly.\n"
            "out_lin = pow_sim.sample(output_overrides={'psf': {'power': 1.0}})\n"
            "out_sqrt = pow_sim.sample()  # YAML default power=0.5\n"
            "out_qrt = pow_sim.sample(output_overrides={'psf': {'power': 0.25}})\n"
            "\n"
            "# Linear colormaps here — pow_scale itself is the dynamic-range\n"
            "# compression, so log on top would hide what we're showing.\n"
            "fig, axes = plt.subplots(1, 3, figsize=(13, 4))\n"
            "for ax, out_i, label in zip(\n"
            "    axes,\n"
            "    [out_lin, out_sqrt, out_qrt],\n"
            "    ['power=1.0 (linear)', 'power=0.5 (sqrt, YAML default)', 'power=0.25'],\n"
            "):\n"
            "    im = ax.imshow(out_i['images']['psf'][..., 0], cmap='inferno')\n"
            "    ax.set_title(label)\n"
            "    ax.set_axis_off()\n"
            "    fig.colorbar(im, ax=ax, shrink=0.7, label='intensity^power')\n"
            "plt.tight_layout()\n"
            "plt.show()\n"
        ),
        _md(
            "## Recap\n\n"
            "Both examples follow the same recipe:\n\n"
            "1. Subclass the right ABC (`Corrector`, `PostProcessor`, …).\n"
            "2. Implement the required methods.\n"
            '3. Decorate with `@register("<kind>", "<name>")`.\n'
            "4. Reference by name in YAML or a Python config dict.\n\n"
            "The same pattern extends to **all six** pluggable component kinds: "
            "`aperture`, `corrector`, `coronagraph`, `focal_plane`, `output_tap`, "
            "`post_processor`. As long as the class is imported before the loader "
            "runs (the `@register` decorator has to execute), the YAML loader "
            "picks it up automatically — no fork of `telescope-sim` required."
        ),
    ],
    "06_actuator_grid_dm": [
        _md(
            "# 6. Actuator-grid DM with baked-in misalignment\n\n"
            "The `actuator_grid` corrector is an N×N influence-function "
            "deformable mirror driven by **raw per-actuator commands** — the "
            "command surface a hardware DM actually exposes, as opposed to the "
            "modal (Zernike) interface of the `zernike` corrector. Two "
            "influence models are available (`gaussian` and `xinetics`, via the "
            "matching HCIPy factories), and DM **misalignment relative to the "
            "pupil is baked in at construction**: `rotation_deg` rotates the "
            "influence-function geometry, `flip_x`/`flip_y` mirror the "
            "command-array indexing (a mirrored cable/mapping). That makes this "
            "corrector the natural simulation stand-in for a real bench DM whose "
            "mounting and wiring a calibration routine must recover.\n\n"
            "Command conventions (pinned by the unit tests):\n\n"
            "- `set_actuators` accepts a flat `(N²,)` or shaped `(N, N)` array;\n"
            "  a shaped command indexes `cmd[iy, ix]` — axis 0 walks y ascending,\n"
            "  axis 1 walks x ascending.\n"
            "- Positive `rotation_deg` rotates the DM **counterclockwise**\n"
            "  relative to the pupil (x right, y up) as seen in a plotted\n"
            "  surface.\n"
            "- Flips mirror the command indexing *before* the rotated geometry\n"
            "  renders it (**flip-then-rotate**), and the `actuators` readback\n"
            "  un-applies them, so callers always round-trip their own values.\n\n"
            "This demo uses a realistic scale: a 50×50 actuator grid at 0.17 m "
            "projected pitch over an 8 m circular pupil (2500 actuators on a "
            "256² grid — the influence-function basis is sparse, and building "
            "it takes a few seconds at bind time)."
        ),
        _code(
            "import numpy as np\n"
            "import matplotlib.pyplot as plt\n"
            "from matplotlib.colors import LogNorm\n"
            "import hcipy\n"
            "\n"
            "from telescope_sim.config.schema import SimConfig\n"
            "from telescope_sim.config.loader import build\n"
            "from telescope_sim.helpers.diagnostics import plot_opd_and_psfs\n"
            "\n"
            "config = {\n"
            "    'pupil': {'resolution': 256, 'extent': 8.65},\n"
            "    'aperture': {\n"
            "        'type': 'external_pupil', 'mode': 'callable',\n"
            "        'module': 'hcipy', 'function': 'make_circular_aperture',\n"
            "        'kwargs': {'diameter': 8.0},\n"
            "        'area': float(np.pi * 4.0 ** 2),\n"
            "    },\n"
            "    'correctors': {\n"
            "        'dm': {\n"
            "            'type': 'actuator_grid',\n"
            "            'num_actuators': 50,\n"
            "            'actuator_pitch': 0.17,\n"
            "            'influence': 'gaussian',\n"
            "            'rotation_deg': 3.5,\n"
            "            'actuate_scale': 1.0e-6,   # caller units -> meters of surface\n"
            "            'wavefront_role': 'actuate',\n"
            "            'target_strategy': 'actuators', 'target': True,\n"
            "        },\n"
            "    },\n"
            "    'corrector_chain': ['dm'],\n"
            "    'focal_planes': {\n"
            "        'filter1': {\n"
            "            'type': 'angular', 'central_lam': 1.55e-6,\n"
            "            'focal_extent': 1.6, 'focal_res': 128,\n"
            "            'fractional_bandwidth': 0.0, 'num_samples': 1,\n"
            "        },\n"
            "    },\n"
            "    'outputs': {\n"
            "        'psf': {\n"
            "            'tap': {'type': 'intensity', 'focal_planes': ['filter1']},\n"
            "            'post_processing': [],\n"
            "        },\n"
            "    },\n"
            "}\n"
            "sim = build(SimConfig.model_validate(config))\n"
            "dm = sim.correctors['dm']\n"
            "print('n_actuators:', dm.n_actuators)\n"
        ),
        _md(
            "Poke two actuators (0.12 command units × `actuate_scale` = 120 nm "
            "of surface each) and sample. The OPD panel shows the two localized "
            "influence-function bumps; the PSF picks up the corresponding "
            "speckle structure."
        ),
        _code(
            "cmd = np.zeros((50, 50))\n"
            "cmd[25, 30] = 0.12   # cmd[iy, ix]: y index 25, x index 30\n"
            "cmd[18, 20] = -0.12\n"
            "\n"
            "out_rest = sim.sample(meas_strehl=True, meas_pupil_opd=True)\n"
            "out_poke = sim.sample(\n"
            "    actuations={'dm': cmd}, meas_strehl=True, meas_pupil_opd=True,\n"
            ")\n"
            "print(f\"Strehl at rest: {out_rest['strehls']['filter1']:.3f}\")\n"
            "print(f\"Strehl poked:   {out_poke['strehls']['filter1']:.3f}\")\n"
            "plot_opd_and_psfs(sim, out_poke, suptitle='two poked actuators')\n"
            "plt.show()\n"
        ),
        _md(
            "## Misalignment: rotation and command flips\n\n"
            "To make the conventions visible, render one asymmetric command "
            "pattern (a blocky letter **F** — unambiguous under both rotations "
            "and mirror flips) through three DM variants: aligned, rotated, and "
            "rotated + `flip_x`. The correctors are constructed directly and "
            "bound to the sim's existing pupil grid — the same call the YAML "
            "loader makes at build time.\n\n"
            "In the aligned panel the F reads normally (axis 0 of the command is "
            "y, so the top bar of the F is at high y). With `rotation_deg=10` "
            "the whole surface turns counterclockwise. Adding `flip_x` mirrors "
            "the command indexing first, then the rotated geometry renders it — "
            "flip-then-rotate."
        ),
        _code(
            "from telescope_sim.correctors.actuator_grid import ActuatorGridCorrector\n"
            "\n"
            "cmd_f = np.zeros((50, 50))\n"
            "cmd_f[10:40, 12:17] = 1.0   # vertical stroke\n"
            "cmd_f[35:40, 12:32] = 1.0   # top bar (high y)\n"
            "cmd_f[22:27, 12:27] = 1.0   # middle bar\n"
            "\n"
            "pupil_grid = sim.aperture.field.grid\n"
            "variants = [\n"
            "    ('aligned', {}),\n"
            "    ('rotation_deg=10', {'rotation_deg': 10.0}),\n"
            "    ('rotation_deg=10 + flip_x', {'rotation_deg': 10.0, 'flip_x': True}),\n"
            "]\n"
            "\n"
            "fig, axes = plt.subplots(1, 4, figsize=(17, 4))\n"
            "im0 = axes[0].imshow(\n"
            "    cmd_f, origin='lower', cmap='RdBu_r', vmin=-1, vmax=1,\n"
            ")\n"
            "axes[0].set_title('command array cmd[iy, ix]')\n"
            "axes[0].set_xlabel('ix')\n"
            "axes[0].set_ylabel('iy')\n"
            "fig.colorbar(im0, ax=axes[0], shrink=0.7, label='command')\n"
            "\n"
            "for ax, (label, kwargs) in zip(axes[1:], variants):\n"
            "    dm_i = ActuatorGridCorrector(50, 0.17, actuate_scale=1.0e-6, **kwargs)\n"
            "    dm_i._bind_pupil_grid(pupil_grid, sim.aperture.field)\n"
            "    dm_i.set_actuators(cmd_f)\n"
            "    plt.sca(ax)\n"
            "    # (Tutorial code reaches into ._dm for the surface; library\n"
            "    # code shouldn't.)\n"
            "    im = hcipy.imshow_field(\n"
            "        dm_i._dm.surface * 1e6, cmap='RdBu_r', vmin=-1.1, vmax=1.1,\n"
            "    )\n"
            "    ax.set_title(label)\n"
            "    ax.set_axis_off()\n"
            "    fig.colorbar(im, ax=ax, shrink=0.7, label='surface [µm]')\n"
            "plt.tight_layout()\n"
            "plt.show()\n"
        ),
        _md(
            "The readback stays in the caller's frame — with flips active the "
            "DM-facing command is mirrored, but `dm.actuators` returns exactly "
            "what was set:"
        ),
        _code(
            "dm_flipped = ActuatorGridCorrector(\n"
            "    50, 0.17, actuate_scale=1.0e-6, flip_x=True, flip_y=True,\n"
            ")\n"
            "dm_flipped._bind_pupil_grid(pupil_grid, sim.aperture.field)\n"
            "dm_flipped.set_actuators(cmd_f)\n"
            "print('round-trip exact:',\n"
            "      np.allclose(dm_flipped.actuators, cmd_f.reshape(-1)))\n"
        ),
        _md(
            "## Influence models: `gaussian` vs `xinetics`\n\n"
            "`gaussian` builds Gaussian pokes with a configurable "
            "nearest-neighbour `crosstalk` (default 0.15); `xinetics` uses "
            "HCIPy's measured Xinetics actuator shape. Both accept the same "
            "misalignment parameters. A single poked actuator on a small demo "
            "grid shows the difference in footprint."
        ),
        _code(
            "mini_grid = hcipy.make_pupil_grid(128, 1.05)\n"
            "mini_aper = hcipy.evaluate_supersampled(\n"
            "    hcipy.make_circular_aperture(1.0), mini_grid, 4,\n"
            ")\n"
            "poke = np.zeros((8, 8))\n"
            "poke[4, 5] = 1.0\n"
            "\n"
            "fig, axes = plt.subplots(1, 2, figsize=(9, 4))\n"
            "for ax, influence in zip(axes, ['gaussian', 'xinetics']):\n"
            "    dm_i = ActuatorGridCorrector(\n"
            "        8, 1.0 / 8, influence=influence, actuate_scale=1.0e-6,\n"
            "    )\n"
            "    dm_i._bind_pupil_grid(mini_grid, mini_aper)\n"
            "    dm_i.set_actuators(poke)\n"
            "    plt.sca(ax)\n"
            "    im = hcipy.imshow_field(dm_i._dm.surface * 1e6, cmap='RdBu_r')\n"
            "    ax.set_title(f'influence={influence!r}')\n"
            "    ax.set_axis_off()\n"
            "    fig.colorbar(im, ax=ax, shrink=0.7, label='surface [µm]')\n"
            "plt.tight_layout()\n"
            "plt.show()\n"
        ),
        _md(
            "## Notes for real-DM workflows\n\n"
            "- `actuate_scale` carries the caller-units → meters-of-surface\n"
            "  calibration; commands stay in the units your control software\n"
            "  already uses.\n"
            "- Building the influence basis dominates sim construction (a few\n"
            "  seconds for 50×50 on a 256² grid); per-sample cost is\n"
            "  sub-millisecond. Build once, sample many times.\n"
            "- `fit_surface` (since v2.2.0) enables fit-role / residual-fit\n"
            "  participation: the DM least-squares-fits any upstream OPD (an\n"
            "  imposed corrector, or an atmosphere passed via\n"
            "  `sample(atmos=...)`) onto its influence basis and the pipeline\n"
            "  cancels it — ideal AO, fitting-error-limited. The fit is\n"
            "  aperture-masked and mean-subtracted (piston is never\n"
            "  commanded), with a tiny Tikhonov term pinning unilluminated\n"
            "  actuators near zero.\n"
        ),
    ],
    "07_jax_backend_batches": [
        _md(
            "# 7. JAX backend: batched sampling and the pure forward model\n\n"
            "Everything so far ran on the default hcipy backend. Setting "
            '`backend: jax` in a config (or `backend="jax"` on the '
            "constructors) swaps wavefront propagation onto JAX — same YAML, "
            "same correctors and outputs, same `sample()` — with results "
            "matching hcipy to float64 round-off. Install the extra with "
            '`pip install "telescope-sim[jax]"` (Python ≥ 3.11).\n\n'
            "Under the hood, propagation runs as a jitted, wavelength-vmapped "
            "matrix Fourier transform and the corrector chain composes into a "
            "single summed pupil-plane OPD (thin phase screens commute). That "
            "pure core is what enables the two features this tutorial covers: "
            "**`sample_batch`** (one device dispatch for a whole batch) and "
            "**`forward_fn`** (a pure function you can `jit` / `vmap` / "
            "`grad` and build your own samplers on)."
        ),
        _code(
            "import numpy as np\n"
            "import matplotlib.pyplot as plt\n"
            "from matplotlib.colors import LogNorm\n"
            "from telescope_sim import TelescopeSim\n"
            "\n"
            'sim_h = TelescopeSim.from_preset("elf_15seg")\n'
            'sim_j = TelescopeSim.from_preset("elf_15seg", backend="jax")\n'
            "\n"
            "rng = np.random.default_rng(0)\n"
            "ptt = rng.normal(scale=0.1, size=(15, 3))\n"
            'img_h = sim_h.sample({"segments": ptt})["images"]["psf"]\n'
            'img_j = sim_j.sample({"segments": ptt})["images"]["psf"]\n'
            "print('cross-backend max |diff| (peak-normalized):',\n"
            "      np.max(np.abs(img_j - img_h)) / img_h.max())\n"
        ),
        _md(
            "## Batched sampling\n\n"
            "`sample_batch` takes actuation arrays with a **leading batch "
            "dimension** and returns a `sample()`-shaped dict with a batch "
            "axis on every image, echo, and Strehl value. On the jax backend "
            "the whole batch propagates as one jitted + vmapped device "
            "dispatch (the first call at a given batch shape pays the jit "
            "compile; subsequent calls reuse it — prefer a fixed batch size "
            "inside loops). On the hcipy backend the same call falls back to "
            "an equivalent Python loop, so code written against "
            "`sample_batch` is backend-portable."
        ),
        _code(
            "B = 16\n"
            "ptt_batch = rng.normal(scale=0.1, size=(B, 15, 3))\n"
            'batch = sim_j.sample_batch({"segments": ptt_batch}, meas_strehl=True)\n'
            "print('images:', batch['images']['psf'].shape)\n"
            "print('echoes:', batch['actuations']['segments'].shape)\n"
            "print('strehls:', batch['strehls']['filter1'].shape)\n"
            "\n"
            "fig, axes = plt.subplots(1, 4, figsize=(14, 3.6))\n"
            "for b, ax in enumerate(axes):\n"
            "    im = ax.imshow(batch['images']['psf'][b, ..., 0], cmap='inferno',\n"
            "                   norm=LogNorm(vmin=1e-6, vmax=1.0))\n"
            "    ax.set_title(f\"sample {b} — S={batch['strehls']['filter1'][b]:.3f}\")\n"
            "    ax.set_axis_off()\n"
            "fig.colorbar(im, ax=list(axes), fraction=0.02, label='normalized intensity')\n"
            "plt.show()\n"
        ),
        _code(
            "import time\n"
            "\n"
            "sim_j.sample_batch({'segments': ptt_batch})  # warm the jit for this shape\n"
            "t0 = time.perf_counter()\n"
            "sim_j.sample_batch({'segments': ptt_batch})\n"
            "t_batch = time.perf_counter() - t0\n"
            "t0 = time.perf_counter()\n"
            "for b in range(B):\n"
            "    sim_h.sample({'segments': ptt_batch[b]})\n"
            "t_loop = time.perf_counter() - t0\n"
            "print(f'jax sample_batch: {1e3 * t_batch / B:.2f} ms/sample')\n"
            "print(f'hcipy loop:       {1e3 * t_loop / B:.2f} ms/sample')\n"
        ),
        _md(
            "Those numbers are CPU; the batch dispatch is where GPUs and the "
            "`precision: float32` config field (half-memory kernels, float32 "
            "outputs — handy for ML consumers) pay off, with no code changes: "
            "JAX picks up whatever accelerator its install supports.\n\n"
            "## Fully on-device batches: `key=`\n\n"
            "By default `sample_batch` runs output taps and post-processing "
            "on the host, per sample — bit-consistent with `sample()`. "
            "Passing **`key=`** (an int seed or a JAX PRNG key) moves the "
            "whole output stage into the device dispatch too: detector noise "
            "on JAX PRNG streams, convolution, normalizations, actuation "
            'echoes, and Strehl. That is the "parameters in → training data '
            'out" path: nothing round-trips through host numpy.\n\n'
            "Noisy outputs are reproducible per key *within* the jax backend; "
            "they deliberately do **not** bit-match the host path's numpy "
            "draws (the detector's flat-field fixed pattern *is* shared, and "
            "noise-free chains match the host exactly). In key-mode, "
            "per-sample overrides like `int_phot_flux` accept arrays with a "
            "leading batch dimension — below, every sample in the batch gets "
            "its own photon flux."
        ),
        _code(
            "from telescope_sim.config.schema import SimConfig\n"
            "from telescope_sim.config.loader import build\n"
            "\n"
            "# Single-filter noisy config (noisy_detector needs exactly one\n"
            "# focal grid); same sELF aperture + PTT corrector as the preset.\n"
            "noisy_cfg = {\n"
            "    'backend': 'jax',\n"
            "    'pupil': {'resolution': 256, 'extent': 3.675},\n"
            "    'aperture': {\n"
            "        'type': 'segmented_circular',\n"
            "        'segment_diameter': 0.5,\n"
            "        'layout': 'elf', 'n_segments': 15, 'ring_radius': 1.5,\n"
            "        'supersample': 16,\n"
            "    },\n"
            "    'correctors': {\n"
            "        'segments': {\n"
            "            'type': 'segmented_ptt',\n"
            "            'piston_scale': 1.0e-6, 'tip_tilt_scale': 1.0e-6,\n"
            "            'wavefront_role': 'actuate',\n"
            "            'target_strategy': 'actuators', 'target': True,\n"
            "        },\n"
            "    },\n"
            "    'corrector_chain': ['segments'],\n"
            "    'focal_planes': {\n"
            "        'filter1': {\n"
            "            'type': 'angular', 'central_lam': 0.975e-6,\n"
            "            'focal_extent': 3.2, 'focal_res': 640,\n"
            "            'fractional_bandwidth': 0.1538, 'num_samples': 11,\n"
            "        },\n"
            "    },\n"
            "    'outputs': {\n"
            "        'psf': {\n"
            "            'tap': {'type': 'intensity', 'focal_planes': ['filter1']},\n"
            "            'post_processing': [\n"
            "                {'type': 'noisy_detector',\n"
            "                 'int_phot_flux': 1.0e7,\n"
            "                 'detector': {\n"
            "                     'read_noise': 5.0,\n"
            "                     'dark_current_rate': 0.0,\n"
            "                     'flat_field': 0.0,\n"
            "                     'include_photon_noise': True,\n"
            "                 }},\n"
            "            ],\n"
            "        },\n"
            "    },\n"
            "    'strehl_method': 'matched_filter',\n"
            "    'strehl_core_rad': 1.2e-6,\n"
            "}\n"
            "noisy_sim = build(SimConfig.model_validate(noisy_cfg))\n"
            "\n"
            "fluxes = np.array([1.0e5, 1.0e6, 1.0e7, 1.0e9])\n"
            "batch4 = rng.normal(scale=0.1, size=(4, 15, 3))\n"
            "out = noisy_sim.sample_batch(\n"
            "    {'segments': batch4},\n"
            "    key=0,\n"
            "    output_overrides={'psf': {'int_phot_flux': fluxes}},\n"
            "    meas_strehl=True,\n"
            ")\n"
            "\n"
            "fig, axes = plt.subplots(1, 4, figsize=(14, 3.6))\n"
            "for b, ax in enumerate(axes):\n"
            "    img = out['images']['psf'][b, ..., 0]\n"
            "    vmax = float(img.max())\n"
            "    floor = max(vmax * 1e-4, 1e-3)\n"
            "    # Clip zero-count pixels to the LogNorm floor (zeros are 'bad'\n"
            "    # under LogNorm and would render as blank specks).\n"
            "    im = ax.imshow(np.maximum(img, floor), cmap='inferno',\n"
            "                   norm=LogNorm(vmin=floor, vmax=vmax))\n"
            "    ax.set_title(f\"flux={fluxes[b]:.0e} — S={out['strehls']['filter1'][b]:.3f}\")\n"
            "    ax.set_axis_off()\n"
            "    fig.colorbar(im, ax=ax, shrink=0.7, label='counts')\n"
            "fig.suptitle('On-device noisy batch — per-sample photon flux')\n"
            "plt.tight_layout()\n"
            "plt.show()\n"
            "\n"
            "again = noisy_sim.sample_batch(\n"
            "    {'segments': batch4}, key=0,\n"
            "    output_overrides={'psf': {'int_phot_flux': fluxes}},\n"
            ")\n"
            "print('same key reproduces exactly:',\n"
            "      np.array_equal(out['images']['psf'], again['images']['psf']))\n"
        ),
        _md(
            "## The pure forward model: `forward_fn`\n\n"
            "`sample_batch` is a *reference composition* — curriculum "
            "samplers, temporal sequences, RL loops, and DataLoader-style "
            "pipelines should compose the primitive underneath it instead: "
            "`sim.forward_fn()` returns a pure, jit/vmap/grad-compatible "
            "function from caller-facing actuation values to raw summed "
            "focal-plane intensities, with the two stages exposed "
            "separately:\n\n"
            "- `opd_from_actuations(acts)` → total pupil-plane OPD (meters)\n"
            "- `intensity_from_opd(opd)` → per-focal-plane raw intensity — "
            "and the hook for **external OPD**: add an atmosphere screen or "
            "any extra pupil OPD to the actuation OPD before this stage\n\n"
            "plus `actuation_echo(acts)` (training-target echoes) and "
            "`strehls_from_intensities(...)` (in-graph Strehl). Because the "
            "forward is pure, gradients flow through the full optical model — "
            "the demo below recovers the sensitivity of Strehl to every "
            "piston/tip/tilt command in one `jax.grad` call."
        ),
        _code(
            "import jax\n"
            "\n"
            "fwd = sim_j.forward_fn()\n"
            "print('inputs:', fwd.corrector_names, fwd.n_actuators)\n"
            "print('focal planes:', fwd.focal_plane_names)\n"
            "\n"
            "# Stage decomposition: actuations -> OPD -> intensities\n"
            "opd = fwd.opd_from_actuations({'segments': ptt})\n"
            "images = fwd.intensity_from_opd(opd)\n"
            "print('opd:', opd.shape, '-> intensity:', images['filter1'].shape)\n"
            "\n"
            "# Gradient of Strehl w.r.t. every PTT command, through the full model\n"
            "def neg_strehl(acts):\n"
            "    intensities = fwd({'segments': acts})\n"
            "    return -fwd.strehls_from_intensities(intensities)['filter1']\n"
            "\n"
            "grads = jax.grad(neg_strehl)(ptt)\n"
            "print('d(-Strehl)/d(ptt):', np.asarray(grads).shape)\n"
            "\n"
            "# vmap composes the same way sample_batch does internally\n"
            "batched_fwd = jax.jit(jax.vmap(fwd))\n"
            "raw = batched_fwd({'segments': ptt_batch})\n"
            "print('vmapped raw intensities:', raw['filter1'].shape)\n"
        ),
        _md(
            "## Fit-role correctors are part of the graph\n\n"
            "Chains with `wavefront_role: fit` correctors work through "
            "`forward_fn` and `sample_batch` too: at build time the "
            "corrector's least-squares `fit_surface` is probed through every "
            "upstream corrector's contribution map (*composed-fit probing*), "
            "so the fit state — and residual-fit training echoes — become "
            "precomputed linear maps of the input actuations. No host "
            "round-trip per sample.\n\n"
            "One boundary to know: the composed fit responds to "
            "*actuation-driven* OPD only. External OPD added at the "
            "`intensity_from_opd` hook bypasses fit-role correctors — for "
            "atmosphere the fit should react to (and cancel), use "
            "`sample(atmos=...)`.\n\n"
            "Below, an imposed Zernike DM disturbs the wavefront and a "
            "second, fit-role Zernike DM cancels it inside the graph: the "
            "batch comes back at the reference Strehl, and the echo reports "
            "the fitted (matching) state of the disturbance."
        ),
        _code(
            "fit_cfg = {\n"
            "    'backend': 'jax',\n"
            "    'pupil': {'resolution': 128, 'extent': 1.05},\n"
            "    'aperture': {\n"
            "        'type': 'external_pupil', 'module': 'hcipy',\n"
            "        'function': 'make_circular_aperture', 'mode': 'callable',\n"
            "        'kwargs': {'diameter': 1.0}, 'area': 0.7853981633974483,\n"
            "    },\n"
            "    'correctors': {\n"
            "        'disturbance': {\n"
            "            'type': 'zernike', 'n_modes': 8, 'zernike_diameter': 1.0,\n"
            "            'starting_mode': 2, 'actuate_scale': 5.0e-8,\n"
            "            'wavefront_role': 'impose',\n"
            "        },\n"
            "        'ao_dm': {\n"
            "            'type': 'zernike', 'n_modes': 8, 'zernike_diameter': 1.0,\n"
            "            'starting_mode': 2, 'actuate_scale': 5.0e-8,\n"
            "            'wavefront_role': 'fit',\n"
            "            'fit_source': 'cumulative_phase_pre_self',\n"
            "            'target_strategy': 'actuators', 'target': True,\n"
            "        },\n"
            "    },\n"
            "    'corrector_chain': ['disturbance', 'ao_dm'],\n"
            "    'focal_planes': {\n"
            "        'filter1': {\n"
            "            'type': 'angular', 'central_lam': 1.0e-6,\n"
            "            'focal_extent': 2.0, 'focal_res': 64,\n"
            "        },\n"
            "    },\n"
            "    'outputs': {\n"
            "        'psf': {'tap': {'type': 'intensity', 'focal_planes': ['filter1']},\n"
            "                'post_processing': []},\n"
            "    },\n"
            "    'strehl_method': 'matched_filter',\n"
            "    'strehl_core_rad': 4.0e-6,\n"
            "}\n"
            "fit_sim = build(SimConfig.model_validate(fit_cfg))\n"
            "\n"
            "zernike_batch = rng.normal(size=(8, 8))\n"
            "corrected = fit_sim.sample_batch({'disturbance': zernike_batch},\n"
            "                                 key=0, meas_strehl=True)\n"
            "print('Strehl with in-graph AO fit :', corrected['strehls']['filter1'].round(6))\n"
            "print('echo (fitted disturbance) vs commands, max |diff|:',\n"
            "      np.max(np.abs(-corrected['actuations']['ao_dm'] - zernike_batch)))\n"
        ),
        _md(
            "## Notes\n\n"
            "- **Parity is pinned by tests**: cross-backend images agree at "
            "1e-12 (observed ~1e-15) on every supported chain, and the "
            "legacy-parity golden fixtures pass on the jax backend at the "
            "same tolerances as hcipy.\n"
            "- **jit recompiles per batch shape** — keep batch sizes fixed "
            "(or pad) inside training loops.\n"
            "- Components with no JAX path (the `fiber_dual` tap, "
            "atmospheres without `.phase_for`) are rejected at config time "
            "with clear errors; the hcipy backend remains the fully general "
            "path.\n"
            "- Custom linear correctors work automatically: the backend "
            "probes `set_actuators` numerically at build time, so anything "
            "whose surface is linear in its commands folds into the forward "
            "map with no extra code."
        ),
    ],
    "08_phase_retrieval": [
        _md(
            "# 8. Phase retrieval: exporting the model to zodiax/dLux\n\n"
            "The JAX backend's `forward_fn` is a pure function — which makes "
            "the whole telescope differentiable. This tutorial wraps it as a "
            "[zodiax](https://github.com/LouisDesdoigts/zodiax) model (the "
            "equinox-based framework underneath "
            "[dLux](https://github.com/LouisDesdoigts/dLux)) and uses "
            "gradient descent to solve a classically hard problem: "
            "recovering the full piston/tip/tilt state of a segmented "
            "telescope from a **single focal-plane image**.\n\n"
            "The instrument is the sELF 15-segment array from tutorial 1, "
            "observed through its focal-plane wavefront-sensing band: "
            "0.90–1.05 µm (15.4% fractional bandwidth at 0.975 µm), "
            "sampled at 11 wavelengths. "
            "Focal-plane phasing of this telescope with a deep CNN was "
            "demonstrated in the small-ELF project¹; two properties of the "
            "instrument make the single-frame problem well-posed, for a "
            "neural network and for gradient descent alike:\n\n"
            "- **The odd segment count** — an odd ring is not "
            "centrosymmetric, which eliminates the twin-image (even/odd) "
            "ambiguity of focal-plane phase retrieval; a centrosymmetric "
            "pupil admits a second wavefront with an identical PSF.\n"
            "- **Spectral bandwidth in one frame** — a monochromatic image "
            "cannot tell a segment piston from the same piston plus a whole "
            "wave, but across a band the wave count differs per wavelength, "
            "so one broadband image resolves the 2π wrap. The 15.4% band "
            "keeps pistons unambiguous out to roughly ±3 waves (λ²/Δλ); "
            "here we recover pistons as deep as **±2.9 waves of OPD** — "
            "nearly six times the monochromatic ±λ/2 capture range.\n\n"
            'Install the pieces with `pip install "telescope-sim[jax]" '
            "zodiax optax` (zodiax brings equinox; Python ≥ 3.11).\n\n"
            "---\n"
            "¹ J. Kuhn *et al.*, “The small-ELF project: toward an "
            "ultra-large coronagraphic optical receiver,” *Ground-Based and "
            "Airborne Telescopes IX*, Proc. SPIE 12182, 161–184 (2022)."
        ),
        _code(
            "import numpy as np\n"
            "import matplotlib.pyplot as plt\n"
            "from matplotlib.colors import LogNorm\n"
            "from telescope_sim.config.loader import build\n"
            "from telescope_sim.config.schema import SimConfig\n"
            "from telescope_sim.helpers.diagnostics import plot_opd_and_psfs\n"
            "\n"
            "# The sELF array observed in its FPWFS band: 0.90-1.05 um\n"
            "# (15.4% at 0.975 um), 11 wavelength samples, 5 mas sampling.\n"
            "config = {\n"
            '    "backend": "jax",\n'
            '    "pupil": {"resolution": 256, "extent": 3.675},\n'
            '    "aperture": {\n'
            '        "type": "segmented_circular", "layout": "elf",\n'
            '        "n_segments": 15, "ring_radius": 1.5,\n'
            '        "segment_diameter": 0.5, "supersample": 16,\n'
            "    },\n"
            '    "correctors": {\n'
            '        "segments": {\n'
            '            "type": "segmented_ptt",\n'
            '            "piston_scale": 1.0e-6, "tip_tilt_scale": 1.0e-6,\n'
            '            "wavefront_role": "actuate",\n'
            '            "target_strategy": "actuators", "target": True,\n'
            "        }\n"
            "    },\n"
            '    "corrector_chain": ["segments"],\n'
            '    "focal_planes": {\n'
            '        "fpwfs_band": {\n'
            '            "type": "angular", "central_lam": 0.975e-6,\n'
            '            "focal_extent": 3.2, "focal_res": 640,\n'
            '            "fractional_bandwidth": 0.1538, "num_samples": 11,\n'
            "        }\n"
            "    },\n"
            '    "outputs": {\n'
            '        "psf": {"tap": {"type": "intensity", "focal_planes": ["fpwfs_band"]},\n'
            '                "post_processing": [{"type": "max_intensity_norm"}]}\n'
            "    },\n"
            "}\n"
            "sim = build(SimConfig.model_validate(config))\n"
            "\n"
            "# The unknown state to recover: random piston/tip/tilt on all 15\n"
            "# segments. Pistons span ±1.4 um of surface = ±2.8 um of OPD —\n"
            "# ±2.9 waves at 0.975 um, close to the band's ±3.2-wave\n"
            "# ambiguity limit and far beyond any monochromatic capture range.\n"
            "PISTON_RANGE = 1.4\n"
            "rng = np.random.default_rng(0)\n"
            "ptt_true = np.zeros((15, 3))\n"
            "ptt_true[:, 0] = rng.uniform(-PISTON_RANGE, PISTON_RANGE, 15)\n"
            "ptt_true[:, 1:] = rng.uniform(-0.2, 0.2, (15, 2))\n"
            "\n"
            'out = sim.sample({"segments": ptt_true}, meas_strehl=True, meas_pupil_opd=True)\n'
            "plot_opd_and_psfs(sim, out,\n"
            '                  suptitle="The unknown PTT state and the frame it produces")\n'
            "plt.show()\n"
        ),
        _md(
            "## Exporting the forward model\n\n"
            "`sim.forward_fn()` returns the telescope as a pure callable — "
            "actuations in, per-filter focal intensities out, "
            "`jit`/`vmap`/`grad`-compatible. To use it the dLux way, wrap it "
            "in a `zodiax.Base` module whose array fields are pytree leaves: "
            "the PTT state becomes a *parameter of the model*, and "
            "everything zodiax/equinox offer (path-based `get`/`set`, "
            "filtered transforms, optax integration) applies to the "
            "telescope like any other dLux model.\n\n"
            "`model()` follows the dLux convention of returning the "
            "observation — here the flux-normalized broadband frame. The "
            "measurement we will fit against is the model evaluated at the "
            "true state, detected at a realistic photon budget (10⁷ photons "
            "of shot noise)."
        ),
        _code(
            "import equinox as eqx\n"
            "import jax\n"
            "import jax.numpy as jnp\n"
            "import optax\n"
            "import zodiax as zdx\n"
            "\n"
            "\n"
            "class PTTModel(zdx.Base):\n"
            "    ptt: jax.Array   # (15, 3) piston/tip/tilt per segment — the free parameters\n"
            "    forward: object  # the telescope's pure forward model (static)\n"
            "\n"
            "    def __init__(self, sim, ptt=None):\n"
            "        self.forward = sim.forward_fn()\n"
            "        self.ptt = jnp.zeros((15, 3)) if ptt is None else jnp.asarray(ptt, float)\n"
            "\n"
            "    def model(self):\n"
            '        images = self.forward({"segments": self.ptt})\n'
            "        return {name: img / img.sum() for name, img in images.items()}\n"
            "\n"
            "\n"
            "model = PTTModel(sim)  # parameters at zero: the starting guess\n"
            "\n"
            "# The single measured broadband frame: the true state rendered and\n"
            "# detected at a realistic photon budget - 1e7 photons of shot noise.\n"
            "noise_key = jax.random.PRNGKey(7)\n"
            "\n"
            "\n"
            "def measure(images):\n"
            "    counts = {name: jax.random.poisson(jax.random.fold_in(noise_key, i), img * 1e7)\n"
            "              for i, (name, img) in enumerate(sorted(images.items()))}\n"
            "    return {name: c / c.sum() for name, c in counts.items()}\n"
            "\n"
            "\n"
            "data = measure(PTTModel(sim, ptt_true).model())\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(5.2, 4.2))\n"
            'img = np.asarray(data["fpwfs_band"])\n'
            "peak = float(img.max())\n"
            'im = ax.imshow(np.maximum(img, peak * 1e-6), cmap="inferno",\n'
            "               norm=LogNorm(vmin=peak * 1e-6, vmax=peak))\n"
            'ax.set_title("the measured frame — everything the fit gets to see")\n'
            "ax.set_axis_off()\n"
            "fig.colorbar(im, ax=ax, fraction=0.046)\n"
            "plt.show()\n"
        ),
        _md(
            "## Retrieval: multi-start descent + integer wrap resolution\n\n"
            "The loss compares amplitudes (square roots of intensities) — "
            "better-behaved gradients than intensity MSE across a PSF's "
            "dynamic range. The landscape is not convex: its minima repeat "
            "along each segment's piston at whole-wave offsets (the *wrap "
            "comb*), with the true state deepest thanks to the bandwidth. A "
            "recipe that handles this reliably:\n\n"
            "1. **Multi-start descent** — run many optimizations from random "
            "starting points *simultaneously*: batch the parameter leaf to "
            "`(N, 15, 3)` and sum the per-start losses. Gradients don't "
            "couple across starts and adam is elementwise, so one ordinary "
            "training loop runs N independent descents on-device.\n"
            "2. **Integer wrap resolution** — descents converge quickly but "
            "may land a whole wave off on some pistons. Test comb hops per "
            "segment in one batched evaluation and greedily accept "
            "improvements; the broadband envelope makes hops toward the true "
            "state improve the loss.\n"
            "3. **Re-polish** the winner with a short low-learning-rate "
            "descent."
        ),
        _code(
            "def one_loss(ptt, model, data):\n"
            '    images = model.set("ptt", ptt).model()\n'
            "    return sum(jnp.mean((jnp.sqrt(images[k]) - jnp.sqrt(data[k])) ** 2)\n"
            "               for k in data)\n"
            "\n"
            "\n"
            "@eqx.filter_jit\n"
            "@eqx.filter_value_and_grad(has_aux=True)\n"
            "def loss_fn(params, model, data):\n"
            "    per_start = jax.vmap(one_loss, in_axes=(0, None, None))(\n"
            '        params["ptt"], model, data)\n'
            "    return per_start.sum(), per_start\n"
            "\n"
            "\n"
            "N_STARTS, ITERS = 16, 250\n"
            "start_rng = np.random.default_rng(99)\n"
            "starts = np.zeros((N_STARTS, 15, 3))\n"
            "starts[1:, :, 0] = start_rng.uniform(-PISTON_RANGE, PISTON_RANGE,\n"
            "                                     (N_STARTS - 1, 15))\n"
            "starts[1:, :, 1:] = start_rng.uniform(-0.2, 0.2, (N_STARTS - 1, 15, 2))\n"
            "\n"
            'params = {"ptt": jnp.asarray(starts)}\n'
            "optim, state = zdx.map_optimisers(\n"
            '    params, {"ptt": optax.adam(optax.cosine_decay_schedule(3e-2, ITERS))})\n'
            "\n"
            "history = []\n"
            "for _ in range(ITERS):\n"
            "    (_, per_start), grads = loss_fn(params, model, data)\n"
            "    updates, state = optim.update(grads, state)\n"
            "    params = optax.apply_updates(params, updates)\n"
            "    history.append(np.asarray(per_start))\n"
            "history = np.array(history)\n"
            "\n"
            "plt.figure(figsize=(7, 4))\n"
            'plt.semilogy(history, color="0.75", lw=0.7)\n'
            'plt.semilogy(history[:, history[-1].argmin()], color="C3", lw=1.8,\n'
            '             label="best start")\n'
            'plt.xlabel("iteration")\n'
            'plt.ylabel("loss")\n'
            "plt.legend()\n"
            'plt.title(f"{N_STARTS} descents in lockstep")\n'
            "plt.show()\n"
        ),
        _code(
            "batched_loss = eqx.filter_jit(jax.vmap(one_loss, in_axes=(0, None, None)))\n"
            "\n"
            "# Piston comb offsets, in caller units (um of surface): 0.4875 um\n"
            "# of surface is one wave of OPD at 0.975 um. Offsets up to four\n"
            "# waves let greedy hops compose their way out of the deepest wraps.\n"
            "WAVE = 0.975 / 2  # um of surface per wave of OPD\n"
            "DELTAS = [s * c * WAVE for c in (1, 2, 3, 4) for s in (1.0, -1.0)]\n"
            "\n"
            "\n"
            "def wrap_resolve(ptt, base):\n"
            '    """Greedy per-segment piston comb hops, one batched eval per step."""\n'
            "    hops = 0\n"
            "    for _ in range(30):\n"
            "        cands = np.repeat(ptt[None], 15 * len(DELTAS), axis=0)\n"
            "        for i in range(15):\n"
            "            for j, d in enumerate(DELTAS):\n"
            "                cands[i * len(DELTAS) + j, i, 0] += d\n"
            "        losses = np.asarray(batched_loss(jnp.asarray(cands), model, data))\n"
            "        k = int(np.argmin(losses))\n"
            "        if not losses[k] < base * 0.999:\n"
            "            return ptt, base, hops\n"
            "        ptt, base = cands[k], float(losses[k])\n"
            "        hops += 1\n"
            "    return ptt, base, hops\n"
            "\n"
            "\n"
            "final = history[-1]\n"
            "best_ptt, best_loss = None, np.inf\n"
            "for idx in np.argsort(final)[:2]:\n"
            "    hopped, loss_i, hops = wrap_resolve(\n"
            '        np.array(params["ptt"][idx], dtype=float), float(final[idx]))\n'
            '    print(f"start {idx}: {hops} wrap hops, loss {final[idx]:.2e} -> {loss_i:.2e}")\n'
            "    if loss_i < best_loss:\n"
            "        best_ptt, best_loss = hopped, loss_i\n"
            "\n"
            'polish = {"ptt": jnp.asarray(best_ptt[None])}\n'
            'optim, state = zdx.map_optimisers(polish, {"ptt": optax.adam(3e-3)})\n'
            "for _ in range(150):\n"
            "    (loss, _), grads = loss_fn(polish, model, data)\n"
            "    updates, state = optim.update(grads, state)\n"
            "    polish = optax.apply_updates(polish, updates)\n"
            'ptt_rec = np.asarray(polish["ptt"][0], dtype=float)\n'
            'print(f"re-polished loss: {float(loss):.2e}")\n'
        ),
        _md(
            "## The recovered state, in actuator space\n\n"
            "Global piston is unobservable — an equal piston on every "
            "segment moves no fringes — so both states are compared with "
            "their mean piston removed."
        ),
        _code(
            "t = ptt_true.copy()\n"
            "h = ptt_rec.copy()\n"
            "t[:, 0] -= t[:, 0].mean()\n"
            "h[:, 0] -= h[:, 0].mean()\n"
            "res = h - t\n"
            "\n"
            "fig, axes = plt.subplots(1, 2, figsize=(11, 4))\n"
            'for k, (label, color) in enumerate(zip(["piston", "tip", "tilt"],\n'
            '                                       ["C0", "C1", "C2"])):\n'
            "    axes[0].scatter(t[:, k], h[:, k], s=24, color=color, label=label,\n"
            "                    alpha=0.85)\n"
            "lim = PISTON_RANGE * 1.15\n"
            'axes[0].plot([-lim, lim], [-lim, lim], "k-", lw=0.6, alpha=0.5)\n'
            'axes[0].set_xlabel("true (caller units)")\n'
            'axes[0].set_ylabel("recovered")\n'
            'axes[0].set_title("45 parameters from one frame — pistons up to ±2.9 waves")\n'
            "axes[0].legend()\n"
            "\n"
            "x = np.arange(15)\n"
            'axes[1].bar(x - 0.25, res[:, 0] * 2e3, 0.25, label="piston (nm OPD)")\n'
            'axes[1].bar(x, res[:, 1] * 1e3, 0.25, label="tip (×1e-3)")\n'
            'axes[1].bar(x + 0.25, res[:, 2] * 1e3, 0.25, label="tilt (×1e-3)")\n'
            'axes[1].set_xlabel("segment")\n'
            "axes[1].set_title(\n"
            '    f"residual: {np.sqrt(np.mean(res[:, 0] ** 2)) * 2e3:.2f} nm RMS piston OPD")\n'
            "axes[1].legend()\n"
            "plt.show()\n"
        ),
        _md(
            "## Closing the loop\n\n"
            "Apply the recovered state, negated, on top of the (still "
            "unknown) true state — exactly what a controller would command — "
            "and the telescope returns to the diffraction limit."
        ),
        _code(
            'corrected = sim.sample({"segments": t - h}, meas_strehl=True,\n'
            "                       meas_pupil_opd=True)\n"
            "plot_opd_and_psfs(sim, corrected,\n"
            '                  suptitle="After applying the recovered correction")\n'
            "plt.show()\n"
            'print("strehls:", {k: round(float(v), 4)\n'
            "                   for k, v in corrected['strehls'].items()})\n"
        ),
        _md(
            "## Notes\n\n"
            "- The same pattern works for **any** config and corrector kind: "
            "`forward_fn` probes correctors numerically, so a Zernike DM or "
            "an `actuator_grid` DM exports identically — nothing here is "
            "specific to the segmented PTT corrector.\n"
            "- The model is an ordinary pytree. Swap adam for any optax "
            "optimiser, stage learning rates per parameter with "
            "`zdx.map_optimisers`, or hand the same loss to a JAX-native "
            "sampler (numpyro, blackjax) for posteriors instead of point "
            "estimates.\n"
            "- Runtime scales with starts × iterations, but every start "
            "shares one jitted program — widening the search is cheap, "
            "especially on accelerators."
        ),
    ],
    "09_differentiable_fast_furious": [
        _md(
            "# 9. Differentiable Fast & Furious\n\n"
            "Fast & Furious¹² is a celebrated focal-plane wavefront sensing "
            "algorithm: from just a science image, a *tiny known DM move*, and "
            "a second image, it reconstructs the wavefront — no wavefront "
            "sensor, no defocus diversity, no extra hardware. The catch is the "
            "machinery: a weak-phase linearization, even/odd image algebra, "
            "regularized inversions and filtering, and with them hard limits — "
            "aberrations below ~1 radian and a symmetric, unaberrated-amplitude "
            "pupil.\n\n"
            "Two ways past those limits have been shown: extending the "
            "algebra to specific symmetric coronagraphs, or replacing the "
            "estimator outright with a deep neural network trained on the "
            "instrument model — the *Tokyo Drift* sequential-diversity "
            "analog³, validated on the SCExAO optical bench with exactly "
            "these inputs. This tutorial takes a third path: **gradient "
            "descent through the telescope model itself**. The raw F&F "
            "inputs — frame, known nudge, frame — go into a generic "
            "least-squares loss, and `jit(vmap(grad))` does the rest.\n\n"
            "We run it on the same real instrument the coronagraph tutorials "
            "use: Subaru's aperture as seen by SCExAO/VAMPIRES in the F750 "
            "filter — a pupil whose bad-actuator masks violate the classical "
            "algorithm's symmetric-amplitude assumption outright. No "
            "linearization means no ~1 rad ceiling either: we recover a "
            "wavefront of **1.5 rad rms** (Strehl 16%) to nanometer "
            "residuals in one shot, through photon noise, and close the loop "
            "at gain 1 — and the same recipe keeps working several radians "
            "deeper.\n\n"
            "Why does F&F need two frames at all? A centrosymmetric pupil "
            "(circular, obstructed, symmetric spiders — most telescopes) has a "
            "*twin-image ambiguity*: the wavefront φ(x) and its parity twin "
            "−φ(−x) produce **pixel-identical** PSFs. In Zernike terms the twin "
            "negates every even-azimuthal-order mode (focus, astigmatism, "
            "spherical) and keeps the odd ones — so a single in-focus image "
            "cannot tell the two apart. The second frame, taken after a known "
            "DM move, breaks the tie: the twin predicts the wrong second image "
            "unless the move has zero even content. We will see all of this "
            "directly — including what the real pupil's small asymmetries do "
            "to it.\n\n"
            "This notebook runs in **two acts**. Act 1 is the F&F-equivalent "
            "workflow on the plain telescope — without the original "
            "algorithm's caveats. Act 2 sends the *same recipe, unchanged*, "
            "through VAMPIRES' **vector vortex coronagraph**, where the "
            "linear machinery cannot go at all — and ends on a twist the "
            "coronagraph's own symmetry provides.\n\n"
            'Install the pieces with `pip install "telescope-sim[jax]" zodiax '
            "optax` (zodiax brings equinox; Python ≥ 3.11).\n\n"
            "---\n"
            "¹ C. U. Keller *et al.*, “Extremely fast focal-plane wavefront "
            "sensing for extreme adaptive optics,” Proc. SPIE 8447, 844721 "
            "(2012).\n\n"
            "² V. Korkiakoski *et al.*, “Fast & Furious focal-plane wavefront "
            "sensing,” Appl. Opt. 53, 4565 (2014).\n\n"
            "³ *Tokyo Drift*: a deep-learning analog of F&F using the same "
            "two-frames-plus-move inputs, validated on the SCExAO optical "
            "bench: M. Bottom *et al.*, “Sequential coronagraphic low-order "
            "wavefront control,” AO4ELT7 proceedings (2023)."
        ),
        _code(
            "import copy\n"
            "\n"
            "import hcipy\n"
            "import numpy as np\n"
            "import matplotlib.pyplot as plt\n"
            "from matplotlib.colors import LogNorm\n"
            "from telescope_sim.config.loader import build\n"
            "from telescope_sim.config.schema import SimConfig\n"
            "from telescope_sim.helpers.diagnostics import plot_opd_and_psfs\n"
            "\n"
            "# The Subaru/SCExAO VAMPIRES F750 mode, inherited from the fixture\n"
            "# configs of tutorials 2-3: the parametric SCExAO pupil (7.92 m Subaru\n"
            "# aperture stopped to 7.79 m, 30% central obscuration, the real spider\n"
            "# geometry — vanes crossing 0.639 m off-center at 51.75° — plus two\n"
            "# bad-actuator masks), a 35-mode Zernike DM (Noll 2-36), and one F750\n"
            "# band: 748 nm, 6.4% bandwidth, 6 mas/pix.\n"
            "config = {\n"
            '    "backend": "jax",\n'
            '    "pupil": {"resolution": 256, "extent": 8.1795},\n'
            '    "aperture": {\n'
            '        "type": "external_pupil",\n'
            '        "module": "test_fixtures/helpers/miles_synthpsf/2024-05_vampires_vvc.py",\n'
            '        "function": "generate_pupil", "mode": "field",\n'
            '        "kwargs": {"outer": 7.79 / 7.92},\n'
            '        "area": 190.663,\n'
            "    },\n"
            '    "correctors": {\n'
            '        "dm": {\n'
            '            "type": "zernike", "n_modes": 35, "zernike_diameter": 7.79,\n'
            '            "starting_mode": 2, "actuate_scale": 1.0e-6,\n'
            '            "wavefront_role": "actuate",\n'
            '            "target_strategy": "actuators", "target": True,\n'
            "        }\n"
            "    },\n"
            '    "corrector_chain": ["dm"],\n'
            '    "focal_planes": {\n'
            '        "band_750": {\n'
            '            "type": "angular", "central_lam": 7.48e-7,\n'
            '            "focal_extent": 0.768, "focal_res": 128,\n'
            '            "fractional_bandwidth": 0.0642, "num_samples": 5,\n'
            "        }\n"
            "    },\n"
            '    "outputs": {\n'
            '        "psf": {"tap": {"type": "intensity", "focal_planes": ["band_750"]},\n'
            '                "post_processing": [{"type": "max_intensity_norm"}]}\n'
            "    },\n"
            '    "strehl_method": "matched_filter",\n'
            '    "strehl_core_rad": 5.8e-7,\n'
            "}\n"
            "sim = build(SimConfig.model_validate(config))\n"
            "\n"
            "# The unknown state: 35 Zernike coefficients, 1.5 rad rms of\n"
            "# wavefront — beyond the ~1 rad weak-phase limit of classical\n"
            "# focal-plane sensing. Strehl is 16%.\n"
            "rng = np.random.default_rng(0)\n"
            "coeffs_true = rng.normal(0.0, 0.06, 35)\n"
            "\n"
            'out = sim.sample({"dm": coeffs_true}, meas_strehl=True, meas_pupil_opd=True)\n'
            "plot_opd_and_psfs(sim, out,\n"
            '                  suptitle="The unknown wavefront and the frame it produces")\n'
            "plt.show()\n"
        ),
        _md(
            "## The twin: why one frame is not enough\n\n"
            "Build the parity twin — negate the even-azimuthal-order "
            "coefficients, keep the odd — and render both. The real pupil "
            "tells this story twice over. Subaru's aperture is point-"
            "symmetric, off-center spider crossing and all: the four vanes "
            "map onto each other under point reflection. Switch off the "
            "SCExAO bad-actuator masks (`actuators: false`) and the two "
            "wavefronts — entirely different — produce PSFs that agree to "
            "machine precision: no amount of fitting a single frame can "
            "choose between them.\n\n"
            "The full pupil's masks are the only asymmetric elements, and "
            "they leak a whisper of parity information: the twin's frame now "
            "differs by a speckle pattern one to two orders of magnitude "
            "below the local intensity. That whisper is real leverage — it "
            "is exactly the effect an *asymmetric-pupil* wavefront sensor "
            "engineers on purpose, and a patient multi-start fit can "
            "usually recover the wavefront from a single frame on this "
            "pupil. But the margin is only a factor of ~2 over the shot "
            "noise of the frames we are about to take, and it would vanish "
            "entirely on a cleaner telescope. The F&F move manufactures a "
            "discrimination an order of magnitude stronger, on *any* pupil "
            "— that robustness, not bare feasibility, is what the second "
            "frame buys."
        ),
        _code(
            "import jax\n"
            "import jax.numpy as jnp\n"
            "\n"
            "# Parity of Z_n^m under point reflection is (-1)^m: the twin\n"
            "# -phi(-x) negates even-m modes and keeps odd-m modes.\n"
            "m_orders = np.array([hcipy.noll_to_zernike(i + 2)[1] for i in range(35)])\n"
            "twin_sign = np.where(np.abs(m_orders) % 2 == 1, 1.0, -1.0)\n"
            "coeffs_twin = twin_sign * coeffs_true\n"
            "\n"
            "# The same instrument with the bad-actuator masks switched off is\n"
            "# exactly centrosymmetric.\n"
            "config_sym = copy.deepcopy(config)\n"
            'config_sym["aperture"]["kwargs"]["actuators"] = False\n'
            "sim_sym = build(SimConfig.model_validate(config_sym))\n"
            "\n"
            "forward, forward_sym = sim.forward_fn(), sim_sym.forward_fn()\n"
            "\n"
            "\n"
            "def frame(fwd, coeffs):\n"
            '    img = fwd({"dm": jnp.asarray(coeffs)})["band_750"]\n'
            "    return img / img.sum()  # flux-normalized: photometry-free\n"
            "\n"
            "\n"
            "rows = []\n"
            "for fwd, label in [(forward_sym, 'masks off (point-symmetric)'),\n"
            "                   (forward, 'full SCExAO pupil')]:\n"
            "    img_t = np.asarray(frame(fwd, coeffs_true))\n"
            "    img_w = np.asarray(frame(fwd, coeffs_twin))\n"
            "    rows.append((label, img_t, img_w))\n"
            "    print(f'{label}: twin max image difference '\n"
            "          f'{np.abs(img_t - img_w).max():.2e} (peak {img_t.max():.2e})')\n"
            "img_true, img_twin = rows[1][1], rows[1][2]\n"
            "\n"
            "fig, axes = plt.subplots(2, 3, figsize=(13, 8.4))\n"
            "for row_axes, (label, img_t, img_w) in zip(axes, rows):\n"
            "    peak = img_t.max()\n"
            "    for ax, img, title in [\n"
            "        (row_axes[0], img_t, f'true wavefront — {label}'),\n"
            "        (row_axes[1], img_w, 'parity twin'),\n"
            "        (row_axes[2], np.abs(img_t - img_w) + peak * 1e-16, '|difference|'),\n"
            "    ]:\n"
            '        im = ax.imshow(img, cmap="inferno",\n'
            "                       norm=LogNorm(vmin=peak * 1e-7, vmax=peak))\n"
            "        ax.set_title(title, fontsize=10)\n"
            '        ax.axis("off")\n'
            "        fig.colorbar(im, ax=ax, fraction=0.046)\n"
            "fig.tight_layout()\n"
            "plt.show()\n"
        ),
        _md(
            "## The measurement: frame, nudge, frame\n\n"
            "The F&F acquisition, verbatim: take a frame, command a small "
            "*random* DM move (a fraction of a radian — the star stays on "
            "target), take a second frame. Both frames are detected at a "
            "10⁶-photon budget — the shot-noise speckle is plainly visible in "
            "the wings, riding just above the actuator masks' faint parity "
            "signal; the known move is what breaks the twin with authority. "
            "Wrapping "
            "`forward_fn` in a `zodiax.Base` module makes the coefficient "
            "vector a model parameter, dLux-style; the known move enters as "
            "an offset at evaluation time."
        ),
        _code(
            "import equinox as eqx\n"
            "import optax\n"
            "import zodiax as zdx\n"
            "\n"
            "\n"
            "class FFModel(zdx.Base):\n"
            "    coeffs: jax.Array  # (35,) Zernike coefficients — the free parameters\n"
            "    forward: object    # the telescope's pure forward model (static)\n"
            "\n"
            "    def __init__(self, sim, coeffs=None):\n"
            "        self.forward = sim.forward_fn()\n"
            "        self.coeffs = jnp.zeros(35) if coeffs is None"
            " else jnp.asarray(coeffs, float)\n"
            "\n"
            "    def model(self, offset=0.0):\n"
            '        img = self.forward({"dm": self.coeffs + offset})["band_750"]\n'
            "        return img / img.sum()\n"
            "\n"
            "\n"
            "N_PHOTONS = 1e6\n"
            "noise_key = jax.random.PRNGKey(7)\n"
            "\n"
            "\n"
            "def measure(img, k):\n"
            "    counts = jax.random.poisson(jax.random.fold_in(noise_key, k), img * N_PHOTONS)\n"
            "    return counts / counts.sum()\n"
            "\n"
            "\n"
            "truth = FFModel(sim, coeffs_true)   # plays the sky + telescope\n"
            "model = FFModel(sim)                # what the fit gets to adjust\n"
            "\n"
            "start_rng = np.random.default_rng(99)\n"
            "move = jnp.asarray(start_rng.normal(0.0, 0.01, 35))  # the known nudge\n"
            "\n"
            "frame_1 = measure(truth.model(), 0)\n"
            "frame_2 = measure(truth.model(offset=move), 1)\n"
            "\n"
            "fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.2))\n"
            "for ax, img, title in [\n"
            '    (axes[0], np.asarray(frame_1), "frame 1"),\n'
            '    (axes[1], np.asarray(frame_2), "frame 2, after the known nudge"),\n'
            "]:\n"
            "    peak = float(img.max())\n"
            "    # clip zero-count pixels to the display floor - LogNorm would\n"
            "    # otherwise render them as blank 'bad' pixels\n"
            '    im = ax.imshow(np.maximum(img, peak * 1e-6), cmap="inferno",\n'
            "                   norm=LogNorm(vmin=peak * 1e-6, vmax=peak))\n"
            "    ax.set_title(title)\n"
            '    ax.axis("off")\n'
            "    fig.colorbar(im, ax=ax, fraction=0.046)\n"
            "fig.tight_layout()\n"
            "plt.show()\n"
        ),
        _md(
            "## The fit: multi-start descent, nothing else\n\n"
            "The loss is plain least squares on amplitudes (square roots of "
            "intensities), summed over the two frames — the second evaluated "
            "with the known move applied. That is the *entire* algorithm.\n\n"
            "The landscape is not convex, so we run many descents at once: "
            "batch the coefficient leaf to `(N, 35)`, `vmap` the per-start "
            "loss, and sum — gradients don't couple across starts and adam is "
            "elementwise, so one ordinary loop runs 32 independent descents in "
            "lockstep on-device. A few starts fall into the (now shallow) twin "
            "basin and simply lose the final argmin."
        ),
        _code(
            "def one_loss(coeffs, model, f1, f2, move):\n"
            '    m = model.set("coeffs", coeffs)\n'
            "    return (jnp.mean((jnp.sqrt(m.model()) - jnp.sqrt(f1)) ** 2)\n"
            "            + jnp.mean((jnp.sqrt(m.model(offset=move)) - jnp.sqrt(f2)) ** 2))\n"
            "\n"
            "\n"
            "@eqx.filter_jit\n"
            "@eqx.filter_value_and_grad(has_aux=True)\n"
            "def loss_fn(params, model, f1, f2, move):\n"
            "    per_start = jax.vmap(one_loss, in_axes=(0, None, None, None, None))(\n"
            '        params["coeffs"], model, f1, f2, move)\n'
            "    return per_start.sum(), per_start\n"
            "\n"
            "\n"
            "N_STARTS, ITERS = 32, 500\n"
            "starts = np.zeros((N_STARTS, 35))\n"
            "starts[1:] = start_rng.normal(0.0, 0.06, (N_STARTS - 1, 35))\n"
            "\n"
            'params = {"coeffs": jnp.asarray(starts)}\n'
            "optim, state = zdx.map_optimisers(\n"
            '    params, {"coeffs": optax.adam(optax.cosine_decay_schedule(3e-2, ITERS))})\n'
            "\n"
            "history = []\n"
            "for _ in range(ITERS):\n"
            "    (_, per_start), grads = loss_fn(params, model, frame_1, frame_2, move)\n"
            "    updates, state = optim.update(grads, state)\n"
            "    params = optax.apply_updates(params, updates)\n"
            "    history.append(np.asarray(per_start))\n"
            "history = np.array(history)\n"
            "\n"
            "best = int(history[-1].argmin())\n"
            'coeffs_rec = np.asarray(params["coeffs"][best], dtype=float)\n'
            "\n"
            "plt.figure(figsize=(7, 4))\n"
            'plt.semilogy(history, color="0.75", lw=0.7)\n'
            'plt.semilogy(history[:, best], color="C3", lw=1.8, label="best start")\n'
            'plt.xlabel("iteration")\n'
            'plt.ylabel("two-frame loss")\n'
            "plt.legend()\n"
            'plt.title(f"{N_STARTS} descents in lockstep")\n'
            "plt.show()\n"
        ),
        _md(
            "## The recovered wavefront\n\n"
            "All 35 coefficients land on the diagonal — including the even-m "
            "modes a single frame is (all but) blind to (red). The residuals "
            "are a few nanometers of OPD, set by the photon noise."
        ),
        _code(
            "t, h = coeffs_true, coeffs_rec\n"
            "res = h - t\n"
            "even = twin_sign < 0\n"
            "\n"
            "fig, axes = plt.subplots(1, 2, figsize=(11, 4))\n"
            'axes[0].scatter(t[~even], h[~even], s=24, color="C0",\n'
            '                label="odd-m modes", alpha=0.85)\n'
            'axes[0].scatter(t[even], h[even], s=24, color="C3",\n'
            '                label="even-m modes (twin-blind)", alpha=0.85)\n'
            "lim = np.abs(t).max() * 1.3\n"
            'axes[0].plot([-lim, lim], [-lim, lim], "k-", lw=0.6, alpha=0.5)\n'
            'axes[0].set_xlabel("true (caller units)")\n'
            'axes[0].set_ylabel("recovered")\n'
            'axes[0].set_title("35 Zernike coefficients from one frame pair")\n'
            "axes[0].legend()\n"
            "\n"
            "axes[1].bar(np.arange(35), res * 2e3,\n"
            '            color=["C3" if e else "C0" for e in even])\n'
            'axes[1].set_xlabel("Noll mode - 2")\n'
            'axes[1].set_ylabel("residual (nm OPD, mode peak)")\n'
            "axes[1].set_title(\n"
            '    f"residual rms: {np.sqrt(np.mean(res ** 2)) * 2e3:.2f} nm OPD per mode")\n'
            "fig.tight_layout()\n"
            "plt.show()\n"
        ),
        _md(
            "## Closing the loop, F&F style\n\n"
            "Because the estimate comes from the full nonlinear model, we can "
            "correct at **gain 1** — no leaky integrator nursing a "
            "linearization. And here is Fast & Furious' elegant trick, "
            "inherited whole: *the correction we just applied is itself the "
            "known move for the next frame pair*. Each new frame extends the "
            "loop for free; a short warm refit from the current estimate keeps "
            "up with 8 starts and 200 iterations."
        ),
        _code(
            "# Apply the correction (gain 1) and take the next frame. The DM\n"
            "# state changes by (-coeffs_rec - move) relative to frame 2 - that\n"
            "# difference is exactly known, and becomes the next pair's move.\n"
            "correction = -coeffs_rec\n"
            "frame_3 = measure(truth.model(offset=jnp.asarray(correction)), 2)\n"
            "move_23 = jnp.asarray(correction) - move\n"
            "\n"
            "# Fitting the (frame_2, frame_3) pair estimates the wavefront at\n"
            "# frame 2's DM state (truth + move): warm-start there, and subtract\n"
            "# the move afterwards to get back the estimate of the truth.\n"
            "warm = np.repeat((coeffs_rec + np.asarray(move))[None], 8, axis=0)\n"
            "warm[1:] += start_rng.normal(0.0, 0.02, (7, 35))\n"
            'params = {"coeffs": jnp.asarray(warm)}\n'
            "optim, state = zdx.map_optimisers(\n"
            '    params, {"coeffs": optax.adam(optax.cosine_decay_schedule(1e-2, 200))})\n'
            "for _ in range(200):\n"
            "    (_, per_start), grads = loss_fn(\n"
            "        params, model, frame_2, frame_3, move_23)\n"
            "    updates, state = optim.update(grads, state)\n"
            "    params = optax.apply_updates(params, updates)\n"
            'coeffs_rec2 = np.asarray(params["coeffs"][int(np.asarray(per_start).argmin())])\n'
            "coeffs_rec2 = coeffs_rec2 - np.asarray(move)\n"
            "\n"
            'corrected = sim.sample({"dm": coeffs_true - coeffs_rec2}, meas_strehl=True,\n'
            "                       meas_pupil_opd=True)\n"
            "plot_opd_and_psfs(sim, corrected,\n"
            '                  suptitle="After the loop\'s second correction")\n'
            "plt.show()\n"
            'print("strehls:", {k: round(float(v), 4)\n'
            "                   for k, v in corrected['strehls'].items()})\n"
        ),
        _md(
            "## Act 2: through the vector vortex coronagraph\n\n"
            "Focal-plane sensing *through a coronagraph* is where the linear "
            "F&F machinery stops entirely: the vortex's complex focal-plane "
            "mask breaks the even/odd image algebra it is built on — this is "
            "the regime the *Tokyo Drift* deep-learning analog³ was invented "
            "for. For gradient descent nothing changes: the coronagraph is "
            "just more differentiable model, and the recipe below is "
            "*identical* to act 1's. The configuration is VAMPIRES' vector "
            "vortex mode from the same fixture lineage: a charge-4 VVC with "
            "the matching parametric Lyot stop (undersized outer edge, "
            "oversized obscuration and spiders).\n\n"
            "The twin ambiguity survives the coronagraph in a surprising "
            "way. A charge +4 vortex maps the parity twin's image onto the "
            "truth's image *through charge −4* — a charge swap — and an "
            "unpolarized vector vortex averages the two charges, so for the "
            "point-symmetric part of the pupil the average is blind to the "
            "swap: behind the VVC the twin is exactly as hidden as behind no "
            "coronagraph — down to the same faint actuator-mask fingerprint, "
            "which a single-frame fit must again pick out of the shot noise "
            "by a whisker. The F&F pair lifts the twin clear of the noise, "
            "exactly as in act 1. (Hold on to that charge-swap fact — it "
            "pays off at the end.)\n\n"
            "A coronagraph operates behind an adaptive-optics system, so act "
            "2's unknown wavefront is a post-AO residual of ~1 rad rms. The "
            "vortex propagation train is heavier than a plain focal plane, so "
            "this act runs on a 128-pixel pupil grid to keep the notebook "
            "quick."
        ),
        _code(
            "config_vvc = copy.deepcopy(config)\n"
            'config_vvc["pupil"]["resolution"] = 128\n'
            'config_vvc["coronagraph"] = {\n'
            '    "type": "vector_vortex", "charge": 4,\n'
            '    "lyot": {"type": "external_pupil",\n'
            '             "module": config["aperture"]["module"],\n'
            '             "function": "generate_pupil", "mode": "field",\n'
            '             "kwargs": {"outer": 0.9, "inner": 0.43,\n'
            '                        "scale": 1.4, "spider_scale": 1.6}},\n'
            "}\n"
            "sim_vvc = build(SimConfig.model_validate(config_vvc))\n"
            "\n"
            "coeffs_vvc_true = rng.normal(0.0, 0.03, 35)  # ~1 rad rms post-AO residual\n"
            "\n"
            "truth_vvc = FFModel(sim_vvc, coeffs_vvc_true)\n"
            "twin_vvc = FFModel(sim_vvc, twin_sign * coeffs_vvc_true)\n"
            "img_t, img_w = np.asarray(truth_vvc.model()), np.asarray(twin_vvc.model())\n"
            'print(f"twin max image difference through the VVC: "\n'
            '      f"{np.abs(img_t - img_w).max():.2e} (peak {img_t.max():.2e})")\n'
            "\n"
            "fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))\n"
            "peak = img_t.max()\n"
            "for ax, img, title in [\n"
            '    (axes[0], img_t, "VVC frame of the true wavefront"),\n'
            '    (axes[1], img_w, "VVC frame of the parity twin"),\n'
            '    (axes[2], np.abs(img_t - img_w) + peak * 1e-16, "|difference|"),\n'
            "]:\n"
            '    im = ax.imshow(img, cmap="inferno", norm=LogNorm(vmin=peak * 1e-7, vmax=peak))\n'
            "    ax.set_title(title, fontsize=10)\n"
            "    ax.set_axis_off()\n"
            "    fig.colorbar(im, ax=ax, fraction=0.046)\n"
            "fig.tight_layout()\n"
            "plt.show()\n"
        ),
        _md(
            "## The same fit, behind the coronagraph\n\n"
            "Frame, nudge, frame — through the vortex this time, at the same "
            "10⁶-photon budget — and the identical multi-start descent. The "
            "only new line of physics is in the configuration."
        ),
        _code(
            "model_vvc = FFModel(sim_vvc)\n"
            "move_vvc = jnp.asarray(start_rng.normal(0.0, 0.01, 35))\n"
            "frame_v1 = measure(truth_vvc.model(), 3)\n"
            "frame_v2 = measure(truth_vvc.model(offset=move_vvc), 4)\n"
            "\n"
            "N_STARTS2, ITERS2 = 16, 350\n"
            "starts2 = np.zeros((N_STARTS2, 35))\n"
            "starts2[1:] = start_rng.normal(0.0, 0.03, (N_STARTS2 - 1, 35))\n"
            "\n"
            'params = {"coeffs": jnp.asarray(starts2)}\n'
            "optim, state = zdx.map_optimisers(\n"
            '    params, {"coeffs": optax.adam(optax.cosine_decay_schedule(3e-2, ITERS2))})\n'
            "for _ in range(ITERS2):\n"
            "    (_, per_start), grads = loss_fn(params, model_vvc, frame_v1, frame_v2,\n"
            "                                    move_vvc)\n"
            "    updates, state = optim.update(grads, state)\n"
            "    params = optax.apply_updates(params, updates)\n"
            'vvc_rec = np.asarray(params["coeffs"][int(np.asarray(per_start).argmin())],\n'
            "                     dtype=float)\n"
            "\n"
            "t2, h2 = coeffs_vvc_true, vvc_rec\n"
            "res2 = h2 - t2\n"
            "fig, axes = plt.subplots(1, 2, figsize=(11, 4))\n"
            'axes[0].scatter(t2[~even], h2[~even], s=24, color="C0",\n'
            '                label="odd-m modes", alpha=0.85)\n'
            'axes[0].scatter(t2[even], h2[even], s=24, color="C3",\n'
            '                label="even-m modes (twin-blind)", alpha=0.85)\n'
            "lim = np.abs(t2).max() * 1.3\n"
            'axes[0].plot([-lim, lim], [-lim, lim], "k-", lw=0.6, alpha=0.5)\n'
            'axes[0].set_xlabel("true (caller units)")\n'
            'axes[0].set_ylabel("recovered")\n'
            'axes[0].set_title("35 coefficients through the coronagraph")\n'
            "axes[0].legend()\n"
            "axes[1].bar(np.arange(35), res2 * 2e3,\n"
            '            color=["C3" if e else "C0" for e in even])\n'
            'axes[1].set_xlabel("Noll mode - 2")\n'
            'axes[1].set_ylabel("residual (nm OPD, mode peak)")\n'
            "axes[1].set_title(\n"
            '    f"residual rms: {np.sqrt(np.mean(res2 ** 2)) * 2e3:.2f} nm OPD per mode")\n'
            "fig.tight_layout()\n"
            "plt.show()\n"
        ),
        _md(
            "## Restoring the null\n\n"
            "Apply the recovered correction and the coronagraph does its job "
            "again: the leaked starlight collapses back into the null. The "
            "Strehl, measured through the plain telescope, confirms the "
            "wavefront itself is fixed."
        ),
        _code(
            "img_before = np.asarray(truth_vvc.model())\n"
            "img_after = np.asarray(FFModel(sim_vvc, coeffs_vvc_true - vvc_rec).model())\n"
            "\n"
            "fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.2))\n"
            "peak = img_before.max()\n"
            "for ax, img, title in [\n"
            '    (axes[0], img_before, "VVC frame before correction"),\n'
            '    (axes[1], img_after, "after the recovered correction"),\n'
            "]:\n"
            '    im = ax.imshow(img, cmap="inferno", norm=LogNorm(vmin=peak * 1e-7, vmax=peak))\n'
            "    ax.set_title(title)\n"
            "    ax.set_axis_off()\n"
            "    fig.colorbar(im, ax=ax, fraction=0.046)\n"
            "fig.tight_layout()\n"
            "plt.show()\n"
            "\n"
            'corrected_vvc = sim.sample({"dm": coeffs_vvc_true - vvc_rec}, meas_strehl=True)\n'
            'print("strehls after correction:",\n'
            "      {k: round(float(v), 4) for k, v in corrected_vvc['strehls'].items()})\n"
        ),
        _md(
            "## The polarization twist: one frame, no nudge\n\n"
            "Now cash in the charge-swap fact. A *vector* vortex acts on the "
            "two circular polarizations with opposite charges — the "
            "unpolarized image averages them, which is what preserved the "
            "twin degeneracy. But image a **single polarization channel** and "
            "only one charge remains: the twin now produces the "
            "*opposite-charge* image, which differs at full speckle "
            "amplitude — no faint mask whisper, but a factor-of-order-unity "
            "signature. The ambiguity that has driven this entire notebook "
            "simply never arises: in the validation campaigns behind this "
            "tutorial the twin basin doesn't just lose the argmin, it stops "
            "capturing descents at all.\n\n"
            "So through one polarization channel, a **single frame with no "
            "diversity move at all** determines the full wavefront:"
        ),
        _code(
            "config_pol = copy.deepcopy(config_vvc)\n"
            'config_pol["coronagraph"]["type"] = "vortex"'
            "  # one charge = one circular polarization\n"
            "sim_pol = build(SimConfig.model_validate(config_pol))\n"
            "\n"
            "truth_pol = FFModel(sim_pol, coeffs_vvc_true)   # same unknown wavefront\n"
            "twin_diff = float(np.abs(np.asarray(truth_pol.model())\n"
            "                         - np.asarray(FFModel(sim_pol, twin_sign"
            " * coeffs_vvc_true).model())).max())\n"
            'print(f"twin max image difference in one polarization channel: {twin_diff:.2e} "\n'
            '      f"- the twin the unpolarized VVC hid is exposed at full amplitude")\n'
            "\n"
            "frame_p = measure(truth_pol.model(), 5)  # ONE frame, no nudge\n"
            "\n"
            "\n"
            "def one_loss_single(coeffs, model, f1):\n"
            '    m = model.set("coeffs", coeffs).model()\n'
            "    return jnp.mean((jnp.sqrt(m) - jnp.sqrt(f1)) ** 2)\n"
            "\n"
            "\n"
            "@eqx.filter_jit\n"
            "@eqx.filter_value_and_grad(has_aux=True)\n"
            "def single_loss_fn(params, model, f1):\n"
            "    per_start = jax.vmap(one_loss_single, in_axes=(0, None, None))(\n"
            '        params["coeffs"], model, f1)\n'
            "    return per_start.sum(), per_start\n"
            "\n"
            "\n"
            "model_pol = FFModel(sim_pol)\n"
            'params = {"coeffs": jnp.asarray(starts2)}\n'
            "optim, state = zdx.map_optimisers(\n"
            '    params, {"coeffs": optax.adam(optax.cosine_decay_schedule(3e-2, 300))})\n'
            "for _ in range(300):\n"
            "    (_, per_start), grads = single_loss_fn(params, model_pol, frame_p)\n"
            "    updates, state = optim.update(grads, state)\n"
            "    params = optax.apply_updates(params, updates)\n"
            'pol_rec = np.asarray(params["coeffs"][int(np.asarray(per_start).argmin())],\n'
            "                     dtype=float)\n"
            "\n"
            "res_p = pol_rec - coeffs_vvc_true\n"
            'corrected_pol = sim.sample({"dm": coeffs_vvc_true - pol_rec}, meas_strehl=True)\n'
            'print(f"single-frame residual rms: {np.sqrt(np.mean(res_p ** 2)) * 2e3:.2f}'
            ' nm OPD per mode")\n'
            'print("strehls after correction:",\n'
            "      {k: round(float(v), 4) for k, v in corrected_pol['strehls'].items()})\n"
        ),
        _md(
            "## Notes\n\n"
            "- The classical algorithm's restrictions came from its "
            "linearization, not from the measurement: with the model "
            "differentiable end-to-end, the same two frames handle "
            "multi-radian aberrations, the real SCExAO pupil (asymmetric "
            "amplitude and all), a vector vortex coronagraph, and photon "
            "noise — and the recipe contains no step specific to any of "
            "them. The coronagraph entered this notebook purely as "
            "configuration.\n"
            "- The validation campaigns behind the pinned numbers "
            "(independently discretized frames + shot noise, three truth "
            "seeds per case) bound the envelope: the pair recovers ~2.5 rad "
            "rms of initial error reliably, hangs on near ~3 rad as "
            "truth-basin capture thins to ~1/32 (working that deep, scale "
            "the restarts — they are cheap under `vmap`), and collapses "
            "past ~4 rad. In photons it delivers nanometer residuals at "
            "10⁶–10⁷ per frame, ~10 nm at 10⁵, tens of nanometers at 10⁴, "
            "and breaks down at 10³. Behind the VVC the same recipe holds "
            "act-1 accuracy at post-AO depths and degrades gracefully down "
            "to 10⁴ photons.\n"
            "- The single-frame results replicate against independent "
            "discretization too: the masks' slim margin holds up (one frame "
            "usually suffices on this pupil, at 2× the noise floor), and "
            "the single-polarization-channel retrieval finishes with zero "
            "twin-basin captures in every campaign seed.\n"
            "- Swap the corrector or the aperture freely: `forward_fn` probes "
            "correctors numerically, so a segmented mirror or an "
            "`actuator_grid` DM exports identically.\n"
            "- The model is an ordinary pytree — hand the same loss to a "
            "JAX-native sampler (numpyro, blackjax) for posteriors, or fit "
            "instrument parameters alongside the wavefront."
        ),
    ],
}


def build_notebook(cells: list[tuple[str, str]]) -> nbformat.NotebookNode:
    """Build an nbformat NotebookNode from (kind, source) pairs."""
    nb = nbformat.v4.new_notebook()
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    nb.metadata["language_info"] = {"name": "python"}
    for kind, source in cells:
        if kind == "markdown":
            nb.cells.append(nbformat.v4.new_markdown_cell(source))
        elif kind == "code":
            nb.cells.append(nbformat.v4.new_code_cell(source))
        else:
            raise ValueError(f"unknown cell kind {kind!r}")
    return nb


def execute_and_save(nb: nbformat.NotebookNode, out_path: Path) -> None:
    """Execute the notebook in-process and write it (with outputs) to disk."""
    client = NotebookClient(
        nb,
        timeout=600,
        kernel_name="python3",
        resources={"metadata": {"path": str(REPO_ROOT)}},
    )
    client.execute()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        nbformat.write(nb, f)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "tutorials",
        nargs="*",
        help=f"Tutorial names to build (defaults to all). Available: {list(TUTORIALS)}",
    )
    args = parser.parse_args()

    selected = args.tutorials or list(TUTORIALS)
    unknown = [t for t in selected if t not in TUTORIALS]
    if unknown:
        print(f"unknown tutorials: {unknown}", file=sys.stderr)
        return 2

    for name in selected:
        print(f"=== building tutorial: {name}")
        nb = build_notebook(TUTORIALS[name])
        execute_and_save(nb, OUT_DIR / f"{name}.ipynb")
        print(f"    wrote {OUT_DIR / f'{name}.ipynb'}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
