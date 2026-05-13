# Contributing to telescope-sim

This document is a stub during the v2.0 alpha. Expect it to fill in as we go.

## Development setup

```bash
conda env create -f envs/env-dev.yaml
conda activate telescope-sim-dev
pip install -e ".[dev,doc]"
pytest
```

## Project layout

- `src/telescope_sim/` — the installable package
- `tests/` — unit tests + fixture regression tests
- `fixtures/` — golden-output digests for the regression suite
- `docs/` — Sphinx documentation, including tutorial notebooks
- `envs/` — conda env YAMLs (current dev, plus era-matched envs for the
  regression suite)

Several local-only directories are used as dev scratch space and are *not*
committed to the repo (see `.gitignore`):

- `hcipy/` — local clone of HCIPy for cross-reference. Clone from
  https://github.com/ehpor/hcipy.git into this directory if you need it.
- Any other ad-hoc reference material developers keep alongside the repo.

## Adding a new optical stage

Each pluggable stage type (Aperture, Corrector, Coronagraph, FocalPlane, OutputTap,
PostProcessor) has an ABC in `src/telescope_sim/abc/` and a registry. To add a new
implementation, write a class that subclasses the ABC and decorate it with
`@register("<kind>", "<name>")`. Users can register their own implementations
externally without modifying the package.

## Running fixture regressions

`pytest tests/fixtures/` runs the slow regression suite (also via
`pytest --runslow`). Each test reconstructs a reference simulation via the
public API and asserts numerical agreement against a committed output digest.
