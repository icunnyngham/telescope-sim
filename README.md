# telescope-sim

[![CI](https://github.com/morphoptic/telescope-sim/actions/workflows/testing.yml/badge.svg)](https://github.com/morphoptic/telescope-sim/actions/workflows/testing.yml)
[![Docs](https://readthedocs.org/projects/telescope-sim/badge/?version=latest)](https://telescope-sim.readthedocs.io)
[![PyPI](https://img.shields.io/pypi/v/telescope-sim.svg)](https://pypi.org/project/telescope-sim/)
[![Python](https://img.shields.io/pypi/pyversions/telescope-sim.svg)](https://pypi.org/project/telescope-sim/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Composable, config-driven simulation of multi-aperture telescope PSFs, deformable
mirrors, coronagraphs, and fiber coupling — built on [HCIPy](https://hcipy.org).

`telescope-sim` provides a pluggable pipeline of optical stages (aperture,
correctors, coronagraph, focal plane, output taps, post-processing), a
YAML-driven configuration schema, and a fixture-based regression suite. Users
can register their own implementations of any stage without modifying the
package.

## Status

🟡 **Alpha (v2.0.0a1).** The pipeline is wired end-to-end and reproduces 10
reference fixtures spanning segmented/mini-ELF apertures, custom-pupil
generators, Zernike-mode DMs, vortex and vector-vortex coronagraphs, angular
and physical focal planes, and multi-mode-fiber dual outputs.

## Quick start

```python
from telescope_sim import TelescopeSim
import numpy as np

# Bundled preset (mini-ELF, 15 segments, 2 filters)
sim = TelescopeSim.from_preset("elf_15seg")

# Or a custom YAML
sim = TelescopeSim.from_yaml("path/to/config.yaml")

# Sample at rest with Strehl ratios
out = sim.sample(meas_strehl=True)
out["images"]["psf"]      # (H, W, n_filters)
out["strehls"]            # {filter_name: ratio}

# Apply per-segment piston/tip/tilt actuations
ptt = np.random.normal(scale=0.1, size=(15, 3))
out = sim.sample(actuations={"segments": ptt}, meas_strehl=True)
```

See [docs/tutorials/](docs/tutorials) for runnable notebooks that exercise the
canonical mini-ELF, vortex coronagraph, custom-pupil + Zernike DM, and fiber
MMF paths.

## Architecture

The pipeline is a linear chain of pupil-plane stages (aperture → correctors →
optional coronagraph) followed by a controlled fan-out at the pupil → focal
boundary, where one or more named focal planes consume the same pupil-plane
wavefront. Each focal plane feeds one or more `OutputTap`s, whose outputs flow
through ordered post-processors. Every stage is a registered, pluggable
implementation of a small ABC. Configs are YAML, validated by pydantic v2.

```
[aperture] → [correctors: c1 → c2 → ... → cN] → [coronagraph?]
                                                    │
                                       ┌────────────┼────────────┐
                                       ▼            ▼            ▼
                                  focal plane  focal plane   focal plane
                                       │            │            │
                                     tap(s)       tap(s)       tap(s)
                                       │            │            │
                                    post-proc    post-proc    post-proc
```

Corrector roles (`actuate` / `impose` / `fit` plus a `target_strategy`) express
the patterns observed across years of research code: model-driven DMs, imposed
atmospheres, fit-residual training targets, and stacked combinations of those.
See [docs/concepts.rst](docs/concepts.rst) for the full discussion.

## Development

```bash
# Create dev environment
conda env create -f envs/env-dev.yaml
conda activate telescope-sim-dev

# After creation, pip may have picked up a local hcipy/ clone; force-replace
# with the PyPI build:
pip uninstall -y hcipy && pip install "hcipy>=0.6"

# Install editable with dev extras
pip install -e ".[dev,doc]"

# Run tests
pytest                  # fast tests only
pytest --runslow        # includes the full fixture regression suite

# Build docs
cd docs && make html
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full development workflow.

## License

MIT — see [LICENSE](LICENSE).
