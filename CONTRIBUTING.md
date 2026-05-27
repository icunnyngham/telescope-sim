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

Cross-reference material (HCIPy clone, the legacy v1 package, the 15
historical variants, scraped test fixtures, working notes) is **not** stored
inside this repo. Developers who need it typically keep it in a parent
"lab" directory one level up — e.g. `../external/hcipy/`,
`../external/TelescopeSim/`, `../variants/`, `../test_fixtures/`. None of
those paths are referenced by the installable package or its tests.

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
