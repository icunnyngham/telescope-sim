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
        _md("# 1. Canonical mini-ELF\n\n"
            "The simplest pipeline: a 15-segment ring of circular sub-apertures, "
            "no atmosphere, no coronagraph. Loaded from the bundled `elf_15seg` preset."),
        _code(
            "import numpy as np\n"
            "import matplotlib.pyplot as plt\n"
            "from telescope_sim import TelescopeSim\n"
            "\n"
            "sim = TelescopeSim.from_preset(\"elf_15seg\")\n"
            "print(\"correctors:\", list(sim.correctors))\n"
            "print(\"focal planes:\", list(sim.focal_planes))\n"
        ),
        _md("Sample at rest (no actuators applied) and inspect the result. The "
            "output `images['psf']` is channels-last over filters; the preset has "
            "two filters at 500 nm and 1 µm."),
        _code(
            "out = sim.sample(meas_strehl=True)\n"
            "psf = out[\"images\"][\"psf\"]\n"
            "print(\"psf shape:\", psf.shape, \"strehls:\", out[\"strehls\"])\n"
            "\n"
            "fig, axes = plt.subplots(1, 2, figsize=(8, 4))\n"
            "for ax, (filt, idx) in zip(axes, list(sim.focal_planes.items())):\n"
            "    im = ax.imshow(np.log10(psf[..., 0] + 1e-8), cmap=\"inferno\")\n"
            "    ax.set_title(filt)\n"
            "    ax.set_axis_off()\n"
            "fig.suptitle(\"At-rest PSFs (log10)\")\n"
            "plt.tight_layout()\n"
            "plt.show()\n"
        ),
        _md("Apply random per-segment piston/tip/tilt errors and resample."),
        _code(
            "rng = np.random.default_rng(0)\n"
            "ptt = rng.normal(scale=0.1, size=(15, 3))\n"
            "out = sim.sample(actuations={\"segments\": ptt}, meas_strehl=True)\n"
            "print(\"strehls with errors:\", out[\"strehls\"])\n"
            "\n"
            "plt.imshow(np.log10(out[\"images\"][\"psf\"][..., 0] + 1e-8), cmap=\"inferno\")\n"
            "plt.title(f\"PSF with random PTT errors — Strehl {out['strehls']['filter1']:.3f}\")\n"
            "plt.axis(\"off\")\n"
            "plt.show()\n"
        ),
    ],
    "02_coronagraph_vortex": [
        _md("# 2. Vortex coronagraph\n\n"
            "Drop a `vortex` coronagraph into the chain. The reference PSF used for "
            "Strehl normalization is always generated with the coronagraph "
            "bypassed (matches the legacy convention). This tutorial loads the "
            "fixture config that reproduces the original VortexCoronagraph(2) "
            "setup against a Keck aperture (HCIPy built-in)."),
        _code(
            "import numpy as np\n"
            "import matplotlib.pyplot as plt\n"
            "from telescope_sim import TelescopeSim\n"
            "\n"
            "sim = TelescopeSim.from_yaml(\"fixtures/configs/07_coro_original.yaml\")\n"
            "out = sim.sample()\n"
            "psf = out[\"images\"][\"psf\"][..., 0]\n"
            "print(\"psf shape:\", psf.shape, \" peak/min:\", psf.max(), psf.min())\n"
            "\n"
            "plt.figure(figsize=(5, 5))\n"
            "plt.imshow(psf, cmap=\"inferno\")\n"
            "plt.title(\"Vortex coronagraph @ rest (Keck aperture, charge=2)\")\n"
            "plt.axis(\"off\")\n"
            "plt.colorbar(shrink=0.8)\n"
            "plt.show()\n"
        ),
    ],
    "03_vampires_zernike": [
        _md("# 3. Custom pupil + Zernike DM (VAMPIRES)\n\n"
            "Wraps an external pupil-generation function (`miles_pupil`) via the "
            "`external_pupil` aperture, and drives a Zernike-mode deformable "
            "mirror. The fixture config reproduces the 2023-10 VAMPIRES base "
            "setup against the captured digest."),
        _code(
            "import numpy as np\n"
            "import matplotlib.pyplot as plt\n"
            "from telescope_sim import TelescopeSim\n"
            "\n"
            "sim = TelescopeSim.from_yaml(\"fixtures/configs/09_vampires_base.yaml\")\n"
            "out = sim.sample()\n"
            "psf = out[\"images\"][\"psf\"][..., 0]\n"
            "\n"
            "fig, axes = plt.subplots(1, 2, figsize=(9, 4))\n"
            "axes[0].imshow(sim.aperture.field.shaped, cmap=\"Greys_r\")\n"
            "axes[0].set_title(\"miles_pupil aperture\")\n"
            "axes[0].set_axis_off()\n"
            "axes[1].imshow(np.log10(psf + 1e-10), cmap=\"inferno\")\n"
            "axes[1].set_title(\"PSF @ rest\")\n"
            "axes[1].set_axis_off()\n"
            "plt.tight_layout()\n"
            "plt.show()\n"
        ),
        _md("Push a few Zernike modes and see the focal plane respond."),
        _code(
            "amps = np.zeros(10)\n"
            "amps[3] = 0.3   # one of the low-order modes\n"
            "out = sim.sample(actuations={\"zernike_dm\": amps})\n"
            "plt.imshow(np.log10(out[\"images\"][\"psf\"][..., 0] + 1e-10), cmap=\"inferno\")\n"
            "plt.title(\"PSF with one Zernike mode pushed\")\n"
            "plt.axis(\"off\")\n"
            "plt.show()\n"
        ),
    ],
    "04_fiber_mmf": [
        _md("# 4. Fiber MMF coupling\n\n"
            "Demonstrates the `physical` focal plane (focal grid in metres, with "
            "an explicit `focal_length`) paired with the `fiber_dual` output tap. "
            "The tap produces a `(2, H, W, 1)` stack: focal-plane intensity in "
            "channel 0 and multi-mode fiber-coupled intensity in channel 1.\n\n"
            "This tutorial uses a deliberately small fiber setup (32-pixel focal "
            "plane, 3 wavelengths, smaller fiber core) so it executes in seconds. "
            "The full-fidelity reproduction lives at "
            "`fixtures/configs/15_fiber_mmf.yaml` and is exercised by "
            "`pytest --runslow tests/fixtures/`."),
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
        help="Tutorial names to build (defaults to all). "
        f"Available: {list(TUTORIALS)}",
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
