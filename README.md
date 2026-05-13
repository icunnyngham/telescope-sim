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
YAML-driven configuration schema, and a fixture-based regression suite. It's
designed to be extensible — users can register their own implementations of
each stage without modifying the package.

## Status

🚧 **Alpha.** Architecture and scaffolding only; concrete optical-stage
implementations land in subsequent phases.

## Quick start (planned API)

```python
from telescope_sim import TelescopeSim
import numpy as np

# One-liner: load a preset
sim = TelescopeSim.from_preset("elf_7seg")

# Or load custom YAML
sim = TelescopeSim.from_yaml("path/to/config.yaml")

# Step atmosphere (caller-driven, RL-friendly)
sim.atmosphere.evolve_until(0.01)

# Generate a sample
out = sim.sample(
    actuations={"segments": np.zeros((7, 3))},
    meas_strehl=True,
)
out["images"]["psf"]      # ndarray, (channels, H, W)
out["actuations"]         # echoed back per target corrector
out["strehls"]            # if requested
```

## Architecture (one paragraph)

The pipeline is a linear chain of pupil-plane stages (atmosphere → aperture →
correctors → coronagraph) that fans out at the pupil→focal boundary to one or more
named focal planes, each producing tapped outputs (intensity, fiber-coupled, phase)
that flow through ordered post-processing. Every stage is a registered, pluggable
implementation of a small ABC. Configs are YAML (validated by pydantic). Atmosphere
lives in the chain but the caller drives time evolution. See
[DESIGN_CHALLENGES.md](DESIGN_CHALLENGES.md) for the long-form design discussion.

## Development

```bash
# Create dev environment
conda env create -f envs/env-dev.yaml
conda activate telescope-sim-dev

# Install editable with dev extras
pip install -e ".[dev,doc]"

# Run tests
pytest                  # fast tests only
pytest --runslow        # includes fixture regression suite

# Build docs
cd docs && make html
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full development workflow.

## License

MIT — see [LICENSE](LICENSE).
