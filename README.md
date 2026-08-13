# telescope-sim

[![CI](https://github.com/icunnyngham/telescope-sim/actions/workflows/testing.yml/badge.svg)](https://github.com/icunnyngham/telescope-sim/actions/workflows/testing.yml)
[![Docs](https://readthedocs.org/projects/telescope-sim/badge/?version=latest)](https://telescope-sim.readthedocs.io)
[![PyPI](https://img.shields.io/pypi/v/telescope-sim.svg)](https://pypi.org/project/telescope-sim/)
[![Python](https://img.shields.io/pypi/pyversions/telescope-sim.svg)](https://pypi.org/project/telescope-sim/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Composable, config-driven simulation of multi-aperture telescope PSFs, deformable
mirrors, coronagraphs, and fiber coupling — built on [HCIPy](https://hcipy.org),
with an optional [JAX](https://docs.jax.dev) backend that makes the whole
telescope differentiable.

`telescope-sim` provides a pluggable pipeline of optical stages (aperture,
correctors, coronagraph, focal plane, output taps, post-processing), a
YAML-driven configuration schema, and a fixture-based regression suite. Users
can register their own implementations of any stage without modifying the
package.

The same YAML runs on either of two compute backends. The default HCIPy
backend is the fully general path; setting `backend: jax` swaps propagation
onto a jitted, batchable, differentiable core with results matching HCIPy to
float64 round-off — one device dispatch per training batch, gradients through
the entire optical model, and export as a [zodiax](https://github.com/LouisDesdoigts/zodiax)/[dLux](https://github.com/LouisDesdoigts/dLux)-style
model for gradient-based phase retrieval, calibration, and ML pipelines.

## Status

**v2.3.0 — beta, on PyPI.** The pipeline is wired end-to-end and reproduces 10
reference fixtures spanning segmented/mini-ELF apertures, custom-pupil
generators, Zernike-mode DMs, vortex and vector-vortex coronagraphs, angular
and physical focal planes, and multi-mode-fiber dual outputs; every
JAX-eligible fixture passes on the JAX backend at the same tolerances.

### What's new since v2.0.0

- **JAX compute backend** (v2.3.0) — `backend: jax` runs the same YAML,
  correctors, outputs, and `sample()` on a jitted, wavelength-vmapped
  matrix-Fourier-transform core with float64-round-off parity against
  HCIPy. On top of it: `sample_batch` (one device dispatch per batch;
  fully on-device noise, echoes, and Strehl with `key=`), `forward_fn`
  (a pure jit/vmap/grad-compatible forward model), in-graph fit-role
  correctors, and a `precision: float32` option. Tutorial 08
  demonstrates the payoff: the telescope exported as a zodiax/dLux
  model and a full segmented-PTT state recovered from a single
  broadband frame by gradient descent.
- **Actuator-grid DM** — the `actuator_grid` corrector: an N×N
  influence-function deformable mirror (gaussian or xinetics actuator
  shapes) driven by raw per-actuator commands, with DM misalignment
  (rotation, mirrored command indexing) baked in at construction. Since
  v2.2.0 it implements `fit_surface`, so it can run in fit-role: the DM
  least-squares-fits any upstream OPD (imposed correctors or a
  `sample(atmos=...)` screen) onto its influence basis and the pipeline
  cancels it — ideal AO, fitting-error-limited.
- **Atmosphere** — pass any HCIPy atmosphere (or any wf→wf callable) as
  `sim.sample(atmos=...)`. Atmospheres that expose `.phase_for(lam)` couple
  automatically into fit-role correctors for cancellation. The reference
  PSF is atmosphere-free by construction.
- **Detector noise** — the `noisy_detector` post-processor wraps HCIPy's
  `NoisyDetector` (read noise, dark current, flat-field, photon shot
  noise) with optional `int_phot_flux` photometry. Per-sample overrides
  via `sim.sample(output_overrides={...})`.
- **Extended-source convolution** — the `convolve_image` post-processor
  convolves the PSF with a caller-supplied scene. Composes with
  `noisy_detector` for noisy extended-source imaging.
- **Cumulative-OPD fit-role correctors** — `wavefront_role="fit"` with
  `fit_source="cumulative_phase_pre_self"` lets a DM auto-fit any upstream
  disturbance (atmosphere, imposed PTT, …) without bespoke wiring.
- **Strehl methods** — `strehl_method: peak | matched_filter`, with the
  matched-filter variant using a circular core mask of radius
  `strehl_core_rad`.
- **Extension tutorial** — `docs/tutorials/05_custom_components.ipynb`
  walks through writing your own `Corrector` and `PostProcessor` via the
  `@register(...)` registry.

## Installation

```bash
pip install telescope-sim
```

Requires Python ≥ 3.10. See [Development](#development) below for an editable
install with dev/doc extras.

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

## JAX compute backend (optional)

```bash
pip install "telescope-sim[jax]"   # requires Python >= 3.11
```

The extra is fully additive: the base install depends only on HCIPy and
never imports JAX, and requesting `backend: jax` without the extra fails
with the install command. The pin resolves to the CPU wheel; for GPU,
install JAX's accelerator build per the
[JAX install docs](https://docs.jax.dev/en/latest/installation.html)
(e.g. `pip install -U "jax[cuda12]"`) — the backend picks it up with no
code changes.

Setting `backend: jax` in a config (or `backend="jax"` on
`from_yaml`/`from_preset`) swaps wavefront propagation onto JAX while
keeping the same YAML schema, correctors, outputs, and `sample()`
semantics — results match the default hcipy backend to float64 round-off
(pinned by the test suite). On top of it:

```python
sim = TelescopeSim.from_preset("elf_15seg", backend="jax")

# Batched sampling: one jitted+vmapped device dispatch for the whole batch
batch = sim.sample_batch({"segments": ptt_batch})            # host-side post
batch = sim.sample_batch({"segments": ptt_batch}, key=0,     # fully on-device:
                         meas_strehl=True)                   # noise, echoes, Strehl

# The pure forward model: jit / vmap / grad it, or build your own sampler
fwd = sim.forward_fn()
images = fwd({"segments": ptt})                  # actuations -> raw intensities
opd = fwd.opd_from_actuations({"segments": ptt}) # ... or stage by stage
images = fwd.intensity_from_opd(opd + screen_opd)  # external-OPD hook
targets = fwd.actuation_echo({"segments": ptt})    # training Y outputs
```

- `sample_batch(...)` returns a `sample()`-shaped dict with a leading batch
  axis; on the hcipy backend it falls back to an equivalent loop.
- `sample_batch(key=...)` (int seed or JAX PRNG key) runs detector noise,
  post-processing, actuation echoes, and Strehl inside the device dispatch
  for end-to-end on-device training-data generation. Noise is reproducible
  per key within the jax backend (it does not bit-match the host path's
  numpy draws; noise-free chains match exactly).
- Fit-role correctors are folded into the forward model at build time
  (composed-fit probing), so residual-fit training targets need no host
  round-trip.
- `precision: float32` in the config halves kernel memory for faster
  sampling; `float64` (the default) is the parity-first setting.
- Components with no JAX path (vortex coronagraphs, `fiber_dual`,
  atmospheres without `.phase_for`) are rejected at config time with clear
  errors — the hcipy backend remains the fully general path.
- Tutorial 07 walks the backend and batched sampling; tutorial 08 exports
  `forward_fn` as a zodiax/dLux-style differentiable model and recovers a
  full segmented-PTT state — pistons several waves deep — from a single
  broadband frame by gradient descent.

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

# If pip discovers a sibling hcipy/ clone (developers often keep one in
# ../external/hcipy/ for cross-reference), force-replace with the PyPI build:
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
