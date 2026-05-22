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
            "# 1. Canonical mini-ELF\n\n"
            "The simplest pipeline: a 15-segment ring of circular sub-apertures, "
            "no atmosphere, no coronagraph. Loaded from the bundled `elf_15seg` preset."
        ),
        _code(
            "import numpy as np\n"
            "import matplotlib.pyplot as plt\n"
            "from telescope_sim import TelescopeSim\n"
            "\n"
            'sim = TelescopeSim.from_preset("elf_15seg")\n'
            'print("correctors:", list(sim.correctors))\n'
            'print("focal planes:", list(sim.focal_planes))\n'
        ),
        _md(
            "Sample at rest (no actuators applied) and inspect the result. The "
            "output `images['psf']` is channels-last over filters; the preset has "
            "two filters at 500 nm and 1 µm."
        ),
        _code(
            "out = sim.sample(meas_strehl=True)\n"
            'psf = out["images"]["psf"]\n'
            'print("psf shape:", psf.shape, "strehls:", out["strehls"])\n'
            "\n"
            "fig, axes = plt.subplots(1, 2, figsize=(8, 4))\n"
            "for ax, (filt, idx) in zip(axes, list(sim.focal_planes.items())):\n"
            '    im = ax.imshow(np.log10(psf[..., 0] + 1e-8), cmap="inferno")\n'
            "    ax.set_title(filt)\n"
            "    ax.set_axis_off()\n"
            'fig.suptitle("At-rest PSFs (log10)")\n'
            "plt.tight_layout()\n"
            "plt.show()\n"
        ),
        _md("Apply random per-segment piston/tip/tilt errors and resample."),
        _code(
            "rng = np.random.default_rng(0)\n"
            "ptt = rng.normal(scale=0.1, size=(15, 3))\n"
            'out = sim.sample(actuations={"segments": ptt}, meas_strehl=True)\n'
            'print("strehls with errors:", out["strehls"])\n'
            "\n"
            'plt.imshow(np.log10(out["images"]["psf"][..., 0] + 1e-8), cmap="inferno")\n'
            "plt.title(f\"PSF with random PTT errors — Strehl {out['strehls']['filter1']:.3f}\")\n"
            'plt.axis("off")\n'
            "plt.show()\n"
        ),
        _md(
            "## Detector noise\n\n"
            "The `noisy_detector` post-processor wraps HCIPy's `NoisyDetector` "
            "(read noise + dark current + flat-field + optional photon shot noise). "
            "It's a single-focal-plane processor — its underlying detector needs "
            "exactly one focal grid — so the simplest demo derives a single-filter "
            "config from the preset and stacks `noisy_detector` under the existing "
            "`intensity` tap.\n\n"
            "Per-sample photon flux is plumbed through "
            '`sim.sample(output_overrides={"psf": {"int_phot_flux": ...}})`: '
            "the same noisy sim covers a wide flux range without rebuilding."
        ),
        _code(
            "from telescope_sim.config.schema import SimConfig\n"
            "from telescope_sim.config.loader import build\n"
            "\n"
            "# Derived single-filter config so the single-focal-plane noisy_detector\n"
            "# constraint is satisfied. Re-uses the 15-segment ELF aperture + PTT\n"
            "# corrector from the preset.\n"
            "noisy_cfg = {\n"
            "    'pupil': {'resolution': 256, 'extent': 3.1563881637},\n"
            "    'aperture': {\n"
            "        'type': 'segmented_circular',\n"
            "        'segment_diameter': 0.5197792270,\n"
            "        'layout': 'elf', 'n_segments': 15, 'ring_radius': 1.25,\n"
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
            "            'type': 'angular', 'central_lam': 0.5e-6,\n"
            "            'focal_extent': 1.0, 'focal_res': 128,\n"
            "            'fractional_bandwidth': 0.05, 'num_samples': 5,\n"
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
            "fig, axes = plt.subplots(1, 3, figsize=(12, 4))\n"
            "for ax, out_i, title in zip(\n"
            "    axes,\n"
            "    [out_low, out_clean, out_high],\n"
            "    ['flux = 1e5', 'flux = 1e7 (YAML default)', 'flux = 1e9'],\n"
            "):\n"
            "    img = out_i['images']['psf'][..., 0]\n"
            "    ax.imshow(np.log10(img + 1e-3), cmap='inferno')\n"
            "    ax.set_title(title)\n"
            "    ax.set_axis_off()\n"
            "fig.suptitle('noisy_detector PSFs at varying photon flux (log10)')\n"
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
            "scene = np.zeros((128, 128), dtype=np.float64)\n"
            "ys, xs = np.indices(scene.shape)\n"
            "for cy, cx, amp, sig in [(45, 55, 3.0, 1.0),\n"
            "                          (70, 70, 1.5, 1.0),\n"
            "                          (60, 90, 0.8, 1.0)]:\n"
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
            "fig, axes = plt.subplots(1, 3, figsize=(12, 4))\n"
            "axes[0].imshow(scene, cmap='inferno')\n"
            "axes[0].set_title('input scene')\n"
            "axes[0].set_axis_off()\n"
            "axes[1].imshow(out_clean_scene['images']['psf'][..., 0], cmap='inferno')\n"
            "axes[1].set_title('intensity → convolve_image')\n"
            "axes[1].set_axis_off()\n"
            "axes[2].imshow(out_noisy_scene['images']['psf'][..., 0], cmap='inferno')\n"
            "axes[2].set_title('intensity → convolve → noisy_detector')\n"
            "axes[2].set_axis_off()\n"
            "plt.tight_layout()\n"
            "plt.show()\n"
        ),
    ],
    "02_coronagraph_vortex": [
        _md(
            "# 2. Vortex coronagraph\n\n"
            "Drop a `vortex` coronagraph into the chain. The reference PSF used for "
            "Strehl normalization is always generated with the coronagraph "
            "bypassed (matches the legacy convention). This tutorial loads the "
            "fixture config that reproduces the original VortexCoronagraph(2) "
            "setup against a Keck aperture (HCIPy built-in)."
        ),
        _code(
            "import numpy as np\n"
            "import matplotlib.pyplot as plt\n"
            "from telescope_sim import TelescopeSim\n"
            "\n"
            'sim = TelescopeSim.from_yaml("fixtures/configs/07_coro_original.yaml")\n'
            "out = sim.sample()\n"
            'psf = out["images"]["psf"][..., 0]\n'
            'print("psf shape:", psf.shape, " peak/min:", psf.max(), psf.min())\n'
            "\n"
            "plt.figure(figsize=(5, 5))\n"
            'plt.imshow(psf, cmap="inferno")\n'
            'plt.title("Vortex coronagraph @ rest (Keck aperture, charge=2)")\n'
            'plt.axis("off")\n'
            "plt.colorbar(shrink=0.8)\n"
            "plt.show()\n"
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
            "from telescope_sim import TelescopeSim\n"
            "\n"
            'sim = TelescopeSim.from_yaml("fixtures/configs/09_vampires_base.yaml")\n'
            "out = sim.sample()\n"
            'psf = out["images"]["psf"][..., 0]\n'
            "\n"
            "fig, axes = plt.subplots(1, 2, figsize=(9, 4))\n"
            'axes[0].imshow(sim.aperture.field.shaped, cmap="Greys_r")\n'
            'axes[0].set_title("miles_pupil aperture")\n'
            "axes[0].set_axis_off()\n"
            'axes[1].imshow(np.log10(psf + 1e-10), cmap="inferno")\n'
            'axes[1].set_title("PSF @ rest")\n'
            "axes[1].set_axis_off()\n"
            "plt.tight_layout()\n"
            "plt.show()\n"
        ),
        _md("Push a few Zernike modes and see the focal plane respond."),
        _code(
            "amps = np.zeros(10)\n"
            "amps[3] = 0.3   # one of the low-order modes\n"
            'out = sim.sample(actuations={"zernike_dm": amps})\n'
            'plt.imshow(np.log10(out["images"]["psf"][..., 0] + 1e-10), cmap="inferno")\n'
            'plt.title("PSF with one Zernike mode pushed")\n'
            'plt.axis("off")\n'
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
            "out_clean = sim.sample(meas_strehl=True)\n"
            "out_atmos = sim.sample(atmos=atmos, meas_strehl=True)\n"
            'print(f\'Strehl no atmos:  {out_clean["strehls"]["filter1"]:.3f}\')\n'
            'print(f\'Strehl atmos on: {out_atmos["strehls"]["filter1"]:.3f}\')\n'
            "\n"
            "fig, axes = plt.subplots(1, 2, figsize=(9, 4))\n"
            "axes[0].imshow(np.log10(out_clean['images']['psf'][..., 0] + 1e-10), cmap='inferno')\n"
            "axes[0].set_title(f\"no atmos — Strehl {out_clean['strehls']['filter1']:.3f}\")\n"
            "axes[0].set_axis_off()\n"
            "axes[1].imshow(np.log10(out_atmos['images']['psf'][..., 0] + 1e-10), cmap='inferno')\n"
            "axes[1].set_title(f\"atmos applied — Strehl {out_atmos['strehls']['filter1']:.3f}\")\n"
            "axes[1].set_axis_off()\n"
            "plt.tight_layout()\n"
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
            "out_corrected = fit_sim.sample(atmos=atmos, meas_strehl=True)\n"
            "print(f\"Strehl with fit-role DM: {out_corrected['strehls']['filter1']:.3f}\")\n"
            "print('fit values shape:', out_corrected['actuations']['zernike_dm'].shape)\n"
            "\n"
            "fig, axes = plt.subplots(1, 3, figsize=(12, 4))\n"
            "for ax, out_i, title in zip(\n"
            "    axes,\n"
            "    [out_clean, out_atmos, out_corrected],\n"
            "    [f\"no atmos — Strehl {out_clean['strehls']['filter1']:.3f}\",\n"
            "     f\"atmos, no correction — Strehl {out_atmos['strehls']['filter1']:.3f}\",\n"
            "     f\"atmos + fit-role DM — Strehl {out_corrected['strehls']['filter1']:.3f}\"],\n"
            "):\n"
            "    ax.imshow(np.log10(out_i['images']['psf'][..., 0] + 1e-10), cmap='inferno')\n"
            "    ax.set_title(title)\n"
            "    ax.set_axis_off()\n"
            "plt.tight_layout()\n"
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
            "sim = build(SimConfig.model_validate(config))\n"
            "out = sim.sample()\n"
            "stack = out['images']['x']   # (2, H, W, 1)\n"
            "focal_psf, mmf_psf = stack[0, ..., 0], stack[1, ..., 0]\n"
            "print('focal shape:', focal_psf.shape, ' mmf shape:', mmf_psf.shape)\n"
            "\n"
            "fig, axes = plt.subplots(1, 2, figsize=(9, 4))\n"
            "axes[0].imshow(np.log10(focal_psf + 1e-12), cmap='inferno')\n"
            "axes[0].set_title('focal-plane intensity')\n"
            "axes[0].set_axis_off()\n"
            "axes[1].imshow(np.log10(mmf_psf + 1e-12), cmap='inferno')\n"
            "axes[1].set_title('multi-mode fiber coupling')\n"
            "axes[1].set_axis_off()\n"
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
            "This tutorial walks through two concrete examples that close feature "
            "gaps without needing to modify the package itself:\n\n"
            "- **Part 1** — a custom `XineticsDM` corrector that auto-fits the\n"
            "  surface of a stock `segmented_ptt` corrector (the legacy\n"
            "  `aprox_ptt_with_dm` workflow).\n"
            "- **Part 2** — a custom `PowScale` post-processor for dynamic-range\n"
            "  compression of PSF images (the legacy `extra_processing.pow_scale`\n"
            "  field).\n\n"
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
        _md("And the focal-plane PSFs side-by-side."),
        _code(
            "fig, axes = plt.subplots(1, 2, figsize=(9, 4))\n"
            "axes[0].imshow(np.log10(out_ptt['images']['psf'][..., 0] + 1e-6), cmap='inferno')\n"
            "axes[0].set_title(f\"PTT only — Strehl {out_ptt['strehls']['filter1']:.3f}\")\n"
            "axes[0].set_axis_off()\n"
            "axes[1].imshow(np.log10(out_corr['images']['psf'][..., 0] + 1e-6), cmap='inferno')\n"
            "axes[1].set_title(f\"PTT + DM — Strehl {out_corr['strehls']['filter1']:.3f}\")\n"
            "axes[1].set_axis_off()\n"
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
            "fig, axes = plt.subplots(1, 3, figsize=(12, 4))\n"
            "for ax, out_i, label in zip(\n"
            "    axes,\n"
            "    [out_lin, out_sqrt, out_qrt],\n"
            "    ['power=1.0 (linear)', 'power=0.5 (sqrt, YAML default)', 'power=0.25'],\n"
            "):\n"
            "    ax.imshow(out_i['images']['psf'][..., 0], cmap='inferno')\n"
            "    ax.set_title(label)\n"
            "    ax.set_axis_off()\n"
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
