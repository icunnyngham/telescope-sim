# Changelog

All notable changes to `telescope-sim` are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project adheres
to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Optional JAX compute backend for focal-plane propagation, selected via
  `backend: jax` in the config YAML or the new `backend=` keyword on
  `TelescopeSim.from_yaml()` / `from_preset()`. The pipeline, apertures,
  correctors, output taps, and post-processors are shared with the default
  hcipy backend; wavefront propagation runs as a jitted, wavelength-vmapped
  matrix Fourier transform built from the same grid geometry, composing the
  corrector chain as summed pupil-plane OPD. Verified to reproduce
  hcipy-backend PSFs to ~1e-15 (max abs, normalized) on the packaged
  preset. Install with `pip install telescope-sim[jax]`. The pure-JAX
  propagation core is the substrate for planned dLux/Zodiax-compatible
  differentiable-model export and batched sampling.
  - Components with no JAX path are rejected at config time with clear
    errors: the `fiber_dual` output tap, `vortex` / `vector_vortex`
    coronagraphs, correctors exposing no mirror surface, and
    `sample(atmos=...)` objects lacking `.phase_for`.
  - The registry gains an additive per-backend overlay:
    `@register(kind, name, backend="jax")` shadows the backend-agnostic
    registration for that backend only; existing registrations and custom
    user components are unaffected.
- `TelescopeSim.forward_fn()` (jax backend): a pure, jit/vmap/grad-compatible
  forward model from caller-facing actuation values to raw summed
  focal-plane intensities, with the `actuations → opd` and
  `opd → intensity` stages exposed separately — the latter accepts any
  extra pupil-plane OPD (e.g. atmosphere screens) added in. Actuate/impose
  corrector chains collapse to precomputed linear maps (verified against a
  probe at build time; nonlinear correctors are rejected with a clear
  error); fit-role chains and non-identity coronagraphs are refused. This
  is the stable primitive for custom batch samplers, gradient-based
  optimization, and the planned differentiable-model export.
- `TelescopeSim.sample_batch()`: reference iid batch sampler — actuation
  arrays with a leading batch dimension in, a `sample()`-shaped dict with
  a leading batch axis out. On the jax backend, propagation for the whole
  batch runs as one jitted+vmapped dispatch over `forward_fn()` (~3.3×
  the per-sample hcipy loop on CPU for the packaged preset at batch 256)
  while output taps, post-processing, actuation echoes, and Strehl reuse
  the exact per-sample `sample()` code path; on the hcipy backend it is a
  plain loop over `sample()` with identical semantics. Custom sampling
  strategies (curriculum, temporal sequences, RL) should compose
  `forward_fn()` directly instead of extending this method.
- `precision: float32` config field (jax backend only; default `float64`):
  builds the propagation kernels in single precision for half-memory,
  faster sampling; images and reference PSFs come out `float32`.
- On-device post-processing for `sample_batch` via a new `key=` argument
  (jax backend; an int seed or JAX PRNG key): each output's tap and
  post-processing chain — including `noisy_detector` photon/read/dark
  noise on JAX PRNG streams, `convolve_image`, the norms, and
  `channels_first` — runs inside the batched device dispatch, so noisy
  training data is generated end-to-end on-device. Noisy outputs are
  reproducible per key within the jax backend (per-sample and per-output
  key splitting) but deliberately do not bit-match the host path's numpy
  draws; the detector's flat-field fixed pattern is shared with the host
  path, and noise-free chains reproduce host results exactly. In
  key-mode, `output_overrides` for `int_phot_flux` / `convolve_image`
  accept one value or an array with a leading batch dimension. Chains
  containing a component with no in-graph equivalent are rejected with a
  clear error naming it (drop `key=` for host-side post-processing).
  Actuation echoes and Strehl remain host-side.

### Changed

- The jax backend's focal planes no longer construct the (unused) hcipy
  propagator and wavefront objects at build time, cutting their build
  overhead; their `lam_setup.propagator` is now `None`. Complex aperture
  fields are rejected on the jax backend with a clear error (the MFT path
  propagates a real transmission amplitude).
- The `intensity` output tap preserves the propagation dtype instead of
  forcing `float64` (visible only with `precision: float32`).

## [2.2.0] - 2026-07-23

Feature release: the `actuator_grid` DM learns `fit_surface()`, so raw-command
actuator-grid mirrors can now run in fit-role — in-pipeline "ideal AO,
fitting-error-limited" modeling with no external fitting code. A cross-kind
contract suite pins the `fit_surface` convention for every corrector that
implements it.

### Added

- `ActuatorGridCorrector.fit_surface()` — the `actuator_grid` corrector can
  now participate in fit-role and residual-fit configurations
  (`wavefront_role: fit`, residual-fit target strategies), enabling
  in-pipeline "ideal AO, fitting-error-limited" modeling: an upstream OPD
  (imposed correctors, or an atmosphere passed via `sample(atmos=...)`) is
  least-squares-fit onto the DM's influence basis and cancelled by the
  pipeline, leaving the fitting-error residual to form the PSF.
  - Aperture-masked regularized least squares on the (sparse) influence
    matrix: the Gram matrix gets a tiny Tikhonov term (`1e-9 ×` mean
    diagonal) so actuators with little or no aperture support stay pinned
    near zero on sparse pupils; the factorized solver is cached after the
    first fit.
  - The aperture-masked mean of the input OPD is subtracted before the fit
    (piston is never commanded), matching the `zernike` corrector's
    convention — a global offset is unobservable, but the DM's imperfect
    flat would print through as an observable ripple if fitted.
  - Fitted commands are returned in caller-facing indexing: `flip_x` /
    `flip_y` are un-applied, so `set_actuators(fit_surface(opd))`
    reproduces the fit under any configured misalignment.
- Cross-kind `fit_surface` contract test suite
  (`tests/unit/test_corrector_fit_contract.py`): every corrector kind
  overriding `fit_surface` is held to the shared convention (matching sign,
  surface→OPD factor of 2, caller-units scaling, piston rejection) by one
  parametrized suite, with a registry guard that fails when a new
  fit-capable kind lacks a contract case.

### Changed

- Development status classifier promoted from Alpha to Beta.
- README: added an Installation section (`pip install telescope-sim`) and
  refreshed the status line; docs and contributing notes updated now that
  the package is published on PyPI.

## [2.1.1] - 2026-07-08

Packaging release: `telescope-sim` is now available on PyPI —
`pip install telescope-sim`. No functional changes to the package.

### Changed

- Package maintainer contact email updated in project metadata.

### Packaging

- First release published to PyPI via automated trusted-publishing (OIDC)
  from CI on tag push.

## [2.1.0] - 2026-07-01

Feature release: the first new component kind since 2.0.0 — a generic
actuator-grid deformable mirror, driven by raw per-actuator commands
the way hardware DMs are, with mount/wiring misalignment simulated at
construction.

### Added

- `actuator_grid` corrector — a generic N×N actuator-grid deformable
  mirror with influence functions, the first in-package DM driven by
  raw per-actuator commands (`ActuatorGridCorrector` in
  `src/telescope_sim/correctors/actuator_grid.py`).
  - `influence: gaussian | xinetics` selects the HCIPy
    influence-function factory (`make_gaussian_influence_functions`
    with configurable nearest-neighbour `crosstalk`, or
    `make_xinetics_influence_functions` with the measured Xinetics
    actuator shape).
  - Misalignment is baked in at construction: `rotation_deg` rotates
    the DM counterclockwise relative to the pupil (x right, y up) via
    the factories' native `z_tilt`; `flip_x` / `flip_y` mirror the
    command-array indexing (`fliplr` / `flipud`) before commands reach
    the DM, mimicking a mirrored cable/mapping. Composition order is
    flip-then-rotate. The `actuators` readback un-applies the flips so
    callers always round-trip their own values.
  - `set_actuators` accepts flat `(N²,)` or shaped `(N, N)` commands;
    a shaped command indexes `cmd[iy, ix]` (axis 0 = y ascending,
    axis 1 = x ascending, matching HCIPy's actuator ordering).
    `actuate_scale` converts caller units to meters of surface.
  - Deferred construction via `_bind_pupil_grid` (same pattern as
    `zernike`); no loader special-case needed.
  Example:
  ```yaml
  correctors:
    dm:
      type: actuator_grid
      num_actuators: 50          # per side; n_actuators = 2500
      actuator_pitch: 0.17       # meters, pupil-plane projected
      influence: gaussian
      rotation_deg: 3.5
      actuate_scale: 1.0e-6
  ```
  `fit_surface` (fit-role / residual-fit participation) is deliberately
  not implemented for this corrector yet.

### Tests

- `tests/unit/test_actuator_grid_corrector.py` — 16 tests pinning the
  geometric conventions (poke localization, the +90° counterclockwise
  rotation sign, both flips, flip-then-rotate composition), command
  handling (flat/shaped equivalence, readback round-trip with and
  without flips, size validation), `actuate_scale` linearity, both
  influence models, registry lookup, and a YAML round-trip through
  `TelescopeSim.from_yaml` + `sample(actuations=...)`
  (`tests/unit/data/actuator_grid_dm.yaml`).

### Reference docs

- `docs/configuration.rst`: `actuator_grid` field reference, including
  the rotation sign convention and flip-then-rotate composition order.
- `docs/api/index.rst`: `telescope_sim.correctors.actuator_grid` entry.
- README: feature bullet; status line caught up (it still said v2.0.1).
- New tutorial `docs/tutorials/06_actuator_grid_dm.ipynb` — SCExAO-scale
  50×50 demo: poked actuators, the rotation/flip misalignment
  visualization (a letter-F command rendered aligned / rotated /
  rotated+flipped), readback round-trip, and a gaussian-vs-xinetics
  influence comparison.
- Tutorial 05 notes that Part 1's capability now ships in-package;
  Part 1 remains the extension-pattern + fit-role reference.
- CHANGELOG footer compare-links repaired (they had stopped at
  v2.0.0a2).

### Feature gaps (CLAUDE.md update)

The Xinetics-DM row (previously "addressed via tutorial, no in-package
implementation unless a fixture needs one") is superseded: a consumer
materialized, and `actuator_grid` with `influence: xinetics` now covers
the ELF/minielf Xinetics lineage natively. `aprox_ptt_with_dm` remains
tutorial-addressed (fit-role machinery; `actuator_grid` deliberately
omits `fit_surface` for now).

## [2.0.2] - 2026-05-22

Small feature + tutorial-polish release. Adopts a uniform diagnostic
convention across all tutorials: every PSF is shown next to the
cumulative pupil-plane OPD that produced it, with colorbars on every
panel and `matplotlib.colors.LogNorm` for log-intensity displays
(colorbar labels carry raw intensity, not `log10`).

### Added

- `sim.sample(meas_pupil_opd=True)` returns `out["pupil_opd"]` as an
  `hcipy.Field` of the cumulative pupil-plane OPD in meters. Sums the
  atmosphere seed (when `atmos.phase_for` is available) and every
  DM-backed corrector's surface × 2 — the same `running_opd` the
  pipeline already computes internally for fit-role correctors, now
  exposed for diagnostic display.
- `telescope_sim.helpers.diagnostics.plot_opd_and_psfs(sim, out)` —
  bundled plotting helper that lays out one OPD panel
  (`hcipy.imshow_field(..., mask=sim.aperture.field)`) plus one PSF
  panel per filter (`norm=LogNorm`), all with colorbars. Used across
  tutorials 01–05.

### Tutorial polish

- Tutorial 01: surface the `elf_15seg` preset YAML directly (via
  `importlib.resources`) so the two-filter config is visible; fix a
  long-standing index bug that showed `psf[..., 0]` in both filter
  panels (the 500 nm and 1 µm PSFs now actually differ in scale, as
  λ/D dictates); show both filters under random PTT errors with the
  same OPD reference.
- Tutorial 02: add OPD panel + colorbar to the vortex coronagraph
  at-rest demo.
- Tutorial 03: OPD panels accompany the at-rest, Zernike-pushed,
  atmosphere-on, and fit-role-corrected samples — the atmospheric
  phase screen is directly visible alongside its PSF, and the
  recovery story is explicit in the OPD residual.
- Tutorial 04: OPD panel + LogNorm + colorbars on the fiber MMF demo
  (uses an inline layout because `fiber_dual` stacks
  `[focal, mmf]` rather than per-filter).
- Tutorial 05: the PTT-only-vs-PTT+DM PSF comparison shares a single
  `LogNorm` across both panels for direct colorbar comparability; the
  `pow_scale` visualization keeps linear colormaps (pow_scale itself
  is the dynamic-range compression) but now has colorbars.

### Reference docs

- `docs/configuration.rst`: document `meas_pupil_opd` alongside
  `meas_strehl` in the per-sample-state section, with a pointer to
  the diagnostics helper.
- `docs/api/index.rst`: add `telescope_sim.helpers.diagnostics`.

## [2.0.1] - 2026-05-21

Documentation polish release — no new package features. Catches the
reference docs and tutorials up to nine alpha releases worth of additions
(`v2.0.0a1`–`v2.0.0a9`) and closes the remaining feature gaps from the
v2.0.0a6 audit (Xinetics DM, `aprox_ptt_with_dm`, `pow_scale`) via
extension-pattern tutorials rather than in-package features.

### Reference docs

- README: drop the alpha-series status line; add a "What's new since
  v2.0.0" callout covering atmosphere, `noisy_detector`,
  `convolve_image`, fit-role correctors, and `strehl_method`.
- `docs/concepts.rst`: replace the stale "atmosphere on backlog"
  section with the shipped per-sample `sim.sample(atmos=...)` contract.
  Atmosphere is an external input — not a corrector. Document the
  callable contract, optional `phase_for(lam)` coupling into the
  cumulative-OPD stream for fit-role correctors, and reference-PSF
  immunity by construction.
- `src/telescope_sim/abc/corrector.py`: scrub the same
  atmosphere-as-corrector framing from the Corrector ABC docstring;
  replace with a `cumulative_phase_pre_self` example.
- `docs/configuration.rst`: document `strehl_method` /
  `strehl_core_rad`, a "Per-sample state" section covering `atmos` and
  `output_overrides` on `sample()`, and the `noisy_detector` /
  `convolve_image` post-processors (parameters, single-focal-plane
  restriction, per-sample-override semantics, composition order).
- `docs/api/index.rst`: add `telescope_sim.post.noisy_detector` and
  `telescope_sim.post.convolve` module entries (autodoc was silently
  skipping them since v2.0.0a7/a9).
- `pyproject.toml`: add `atmosphere` to keywords.

### Tutorials

- Tutorial 01 (canonical mini-ELF) gains a "Detector noise" section
  (`noisy_detector` post-processor + per-sample `int_phot_flux`
  overrides at varying flux) and an "Extended-source imaging via
  `convolve_image`" section (synthetic scene injected via
  `output_overrides`, showing the `intensity → convolve → noisy_detector`
  composition).
- Tutorial 03 (VAMPIRES Zernike) gains an "Atmosphere as a per-sample
  input" section: HCIPy `InfiniteAtmosphericLayer` injected via
  `sim.sample(atmos=...)`, then a derived sim with the Zernike DM
  flipped to `wavefront_role="fit"` +
  `fit_source="cumulative_phase_pre_self"` recovering Strehl from
  0.058 → 0.804 with no caller-provided actuations.
- Tutorial 02 (coronagraph) and 04 (fiber MMF) unchanged — neither
  composes meaningfully with the new features.

### New tutorial

- `docs/tutorials/05_custom_components.ipynb` — combined extension-
  pattern walkthrough. Part 1 defines a custom `XineticsDM(Corrector)`
  using `hcipy.make_xinetics_influence_functions` with the same
  lstsq-on-aperture-masked-pixels `fit_surface` pattern as the
  package's `ZernikeCorrector`, demonstrating the legacy
  `aprox_ptt_with_dm` workflow via the existing fit-role machinery.
  Part 2 defines a custom `PowScale(PostProcessor)` that re-creates
  the legacy `extra_processing.pow_scale` field with per-sample
  `power` overrides. Both classes use `@register(...)` at module
  scope; the YAML loader picks them up by name.



### Added

- `convolve_image` post-processor (kind: `convolve_image`) — closes the
  `convolve_im` feature gap from the v2.0.0a6 audit. Convolves the
  focal-plane PSF (as a kernel, normalized by the at-rest reference
  PSF sum) with a caller-provided scene. Bit-identical to the legacy
  formula `fftconvolve(scene, psf / reference_psf_sum, mode='same')`.
  Per-sample image override via `sim.sample(output_overrides=...)`.
  Single-focal-plane (kernel normalization needs one reference sum);
  multi-channel outputs that want convolution must split into one
  output per filter.

- `PipelineContext.overrides` — first-class field carrying per-sample
  tap-config overrides into post-processors. Populated by the
  pipeline before running each output's post-processing list with
  `output_overrides[output_name]`. Existing four normalization
  post-processors ignore the field (no signature changes); new
  post-processors that need per-sample state read from it.

- `LoaderBindable` marker mixin on the `PostProcessor` ABC. Processors
  that need focal-grid / aperture-area / reference-PSF-sum injected
  at sim-build time implement `_bind_loader_dependencies(...)`. The
  loader walks the post-processing list and calls the hook on any
  processor exposing it. Replaces the v2.0.0a7 noisy-tap-specific
  `_bind_focal_grid` mechanism with a general pattern.

### Refactored

- `NoisyIntensityOutputTap` → `NoisyDetectorPostProcessor`. The math
  is unchanged (single-call `Detector.integrate(power_field, dt=1)`
  with the field built from `intensity * grid.weights` and optionally
  rescaled to `flux * area`), but noise is conceptually a
  transformation on the PSF, not an extraction from the focal-plane
  chain — so it belongs in `src/telescope_sim/post/` not
  `src/telescope_sim/outputs/`. This refactor enables the
  `intensity → convolve → noise` composition that's the physical
  ordering for extended-source noisy imaging.

  YAML schema for noisy outputs moves the noise block from the `tap`
  section to the `post_processing` list:

  Before (v2.0.0a7-a8):
  ```yaml
  outputs:
    psf:
      tap:
        type: noisy_intensity
        focal_planes: [filter1]
        int_phot_flux: 1.0e6
        detector: {...}
  ```

  After (v2.0.0a9+):
  ```yaml
  outputs:
    psf:
      tap:
        type: intensity
        focal_planes: [filter1]
      post_processing:
        - type: noisy_detector
          int_phot_flux: 1.0e6
          detector: {...}
  ```

  Eager detector binding for RNG-burn determinism is preserved (now
  via the generalized `_bind_loader_dependencies` hook).

### Removed

- `src/telescope_sim/outputs/noisy_intensity.py` and its
  `NoisyIntensityOutputTap` class — replaced by the post-processor.
- The loader's noisy_intensity-tap special-case (aperture_area
  injection + `_bind_focal_grid` call) — generalized into the
  post-processor walk.

### Tests

- Migrated `test_noisy_intensity_output_tap_parity.py` →
  `test_noisy_detector_post_processor_parity.py`. Same 16 assertions
  retargeted at the post-processor. Fixture #17 unchanged at the
  digest level; v2 reproducer passes bit-for-bit against the captured
  legacy data.
- New `tests/unit/test_convolve_image_post_processor.py` — 12
  assertions covering structural, math, composition (with
  noisy_detector), and validation.

### Feature gaps (CLAUDE.md update)

Removed from the local feature-gap table:
- `convolve_im` PSF-convolution path — addressed by the
  `convolve_image` post-processor.

Remaining deferred gaps: `include_fft`, `pow_scale`, `gauss_noise`,
`aprox_ptt_with_dm`, Xinetics DM corrector, Lyot/perfect
coronagraphs.

## [2.0.0a8] - 2026-05-19

### Added

- `sim.sample(atmos=...)` — per-sample atmosphere kwarg, externally
  managed (matches legacy semantics; the caller owns the atmosphere
  object and time evolution, v2 holds no atmosphere state). Accepts any
  callable `Wavefront → Wavefront`; the canonical case is an HCIPy
  `InfiniteAtmosphericLayer` or `MultiLayerAtmosphere`.

  Chain ordering: atmosphere → corrector chain → (coronagraph?) → focal
  propagator. The reference PSF is built once at sim-build with no
  atmosphere — by construction it cannot be polluted by per-sample
  atmospheric phase.

- Fit-role coupling: when `atmos` exposes `.phase_for(lam)` (HCIPy
  convention: phase = 2π·OPD/λ), the pipeline seeds `running_opd` with
  the atmospheric OPD at the top of step 2's chain walk. Fit-role
  correctors using `fit_source="cumulative_phase_pre_self"` then
  naturally project the atmosphere into their actuator basis. The
  legacy's separate `_measure_atmos_ptt` / `_aprox_via_dm` paths
  (`multi_aperture_psf.py:375-389, 397-409`) are unnecessary in v2 —
  the cumulative-OPD + fit-role machinery from commit 33914ee covers
  the entire atmosphere-residual workflow.

  Without `.phase_for` (e.g., a plain `wf → wf` test callable), the
  wavefront is still modified but fit-role correctors see
  `cum_opd_pre = 0`. Documented in the `sample()` docstring as the
  contract.

### Removed

- `src/telescope_sim/atmosphere.py` — the never-implemented stub with
  `evolve_until` / YAML-declared time evolution. The legacy pattern
  (caller-managed external atmosphere) is simpler and more RL-friendly;
  the stub's design was overdesigned for v2's needs.
- `self.atmosphere` placeholder attribute on `TelescopeSim`.

### Tests

- `tests/unit/test_atmosphere_chain.py` — 6 tests covering the contract:
  reference PSF is never polluted, atmos measurably affects the sample
  PSF, a Z4 atmosphere is cancelled by a Zernike fit-role corrector
  (PSF matches reference at rtol=1e-7), fit-corrector actuator echo
  matches the analytical amplitude, opaque atmos (no phase_for) still
  modifies wavefront without fit-role coupling, and closed-loop:
  feeding the echo into an impose-role corrector cancels the
  atmosphere. Uses minimal `_FakeAtmos` / `_OpaqueAtmos` stand-ins;
  no HCIPy `InfiniteAtmosphericLayer` required.

### Feature gaps (CLAUDE.md update)

Removed from the local feature-gap table:
- HCIPy atmosphere wiring — addressed by the per-sample `atmos` kwarg.

Remaining deferred gaps (unchanged from v2.0.0a7): `convolve_im`,
`include_fft`, `pow_scale`, `gauss_noise`, `aprox_ptt_with_dm`,
Xinetics DM corrector, Lyot/perfect coronagraphs.

## [2.0.0a7] - 2026-05-19

### Added

- `NoisyIntensityOutputTap` (`output_tap` kind: `noisy_intensity`) —
  closes the long-standing `NoisyDetector` feature gap catalogued in
  v2.0.0a6's audit. Wraps `hcipy.NoisyDetector` and integrates the
  wavelength-summed PSF in a single `integrate(power_field, dt=1)` call.
  The legacy `_addNoiseToObservation` (in
  `multi_aperture_psf.py:548-585`) reconstructed a fake Wavefront from
  `sqrt(intensity)` before integrating — unnecessary, since HCIPy's
  `Detector.integrate()` accepts a `hcipy.Field` directly. Same
  single-exposure semantic, ~20 lines lighter, no fake-Wavefront
  overhead.
  Config:
  ```yaml
  outputs:
    psf:
      tap:
        type: noisy_intensity
        focal_planes: [filter1]
        int_phot_flux: 1.0e8         # default; can be overridden per-sample
        aperture_area: null           # null → loader injects from aperture
        clamp_nonnegative: true       # legacy np.abs() workaround
        detector:
          read_noise: 5.0
          dark_current_rate: 1.0
          flat_field: 0.0
          include_photon_noise: true
  ```
- Per-sample tap-config overrides on `sim.sample()`:
  ```python
  sim.sample(
      actuations={"segments": ptt},
      output_overrides={"psf": {"int_phot_flux": 5.0e7}},
      meas_strehl=True,
  )
  ```
  The `output_overrides` dict is keyed by output name; the value's keys
  override the corresponding tap-constructor fields for that sample
  only. `IntensityOutputTap` and `FiberDualOutputTap` accept the kwarg
  and ignore it (they have no per-sample state).
- `OutputTap.extract()` ABC now takes `*, overrides=None`. Existing
  taps updated to accept-and-ignore. No functional change for callers
  using the previous signature.
- Eager detector binding via `_bind_focal_grid()` hook on
  `NoisyIntensityOutputTap`. The loader calls this after building each
  focal plane so the underlying `hcipy.NoisyDetector` (and the
  `np.random.normal(...)` call in its flat-field setter) construct at
  sim-build time — not lazily on first sample. This makes seeded-RNG
  outputs deterministic and matches legacy behavior, which built the
  detector during sampler `__init__`.

### Tests

- `tests/unit/test_noisy_intensity_output_tap_parity.py` — 17 tests
  organized in four layers for the stochastic component:
  - Structural (RNG-free): shape, kwarg honoring, per-sample override
    beats YAML default, wavefronts not mutated, detector built once,
    clamp_nonnegative flag.
  - Noise-off identity: every noise source zeroed → exact equality
    with the manually-constructed `power_field * dt = flux*area`
    expectation. Decouples wiring from RNG.
  - Statistical (seeded, N=96): read-only → `std ≈ read_noise` and
    mean unchanged; dark-only → `mean += rate*dt` exactly;
    photon-only → `var ≈ mean` (Poisson) on bright pixels.
  - Determinism: same `np.random.seed` → bit-for-bit identical
    outputs; different seeds diverge.
- Legacy-parity fixture #17 (`17_noisy_psf`): three configurations
  (at-rest + two photon fluxes, plus a tip-tilt + flux case) captured
  under `np.random.seed(42)`. Both clean and noisy outputs match the
  legacy `_addNoiseToObservation` reference bit-for-bit. Wired into
  `tests/fixtures/test_canonical.py::test_noisy_psf_v2_reproduces_digest`
  with `@pytest.mark.slow`.

### Feature gaps (CLAUDE.md update)

Removed from the local CLAUDE.md feature-gap table:
- `NoisyDetector per-filter` — addressed by `noisy_intensity` tap.
- `int_phot_flux` photon-flux scaling — addressed by the per-sample
  override mechanism.

Remaining deferred gaps (unchanged from v2.0.0a6): `convolve_im`,
`include_fft`, `pow_scale`, `gauss_noise`, `aprox_ptt_with_dm`,
Xinetics DM corrector, HCIPy atmosphere wiring, Lyot/perfect
coronagraphs.

## [2.0.0a6] - 2026-05-19

### Audit campaign — v2.0 → legacy parity sweep

This release is a defensive audit pass. The strehl bug (commit 1c0f519,
v2.0.0a5) and the three latent bugs from commit 33914ee (Zernike fit
`/2` factor, SegmentedPTT block-layout actuator scramble, segment-mask
overlap) had all shipped without crashing — caught only when regression
tests were finally written. Every fixture before #16 captured an at-rest
sample (all-zero actuations), and at rest most buggy code paths happen
to evaluate correctly. This release closes that coverage gap.

Fifteen components audited (14 from the inventory + a variant-side cache
walk). Each got a parity unit test in `tests/unit/test_*_parity.py`
that asserts behavior against either a direct HCIPy reproduction of the
legacy one-liner OR a synthetic input where the legacy formula is
hand-derivable. **No bugs surfaced** — every v2 wrapper / formula /
construction matches its legacy counterpart bit-for-bit (or within
floating-point noise at atol=1e-12 to atol=1e-30 where measured).

Audit summary by tier (component → commit):
- Tier 1 (complex math / non-canonical lineage): VortexCoronagraph
  (`582518f`), VectorVortexCoronagraph (`78e51f3`), FiberDualOutputTap
  (`8f99503`), ExternalPupilAperture (`a0c4a4f`), ZernikeCorrector
  (`59b5d58`), PhysicalFocalPlane (`e187be7`).
- Tier 2 (canonical lineage / simpler math): AngularFocalPlane
  (`9806863`), SegmentedCircularAperture (`48456a8`), IntensityOutputTap
  (`b9b1d3c`), IdentityCoronagraph (`ce447d4`).
- Tier 3 (post-processors): MaxIntensityNorm, PerSampleNorm,
  MaxImageNorm, ChannelsFirst bundled (`577408a`). Closes the only
  C-bucket (zero-coverage) component (ChannelsFirst).
- Tier 4 (cache audit): variant-side `self.*` walk (`7c2f01a`).

Net: 14 of 15 inventoried v2 components moved from B-bucket
(at-rest fixture only) to A-bucket (formula-asserted). The fifteenth
(SegmentedPTTCorrector) was already A-bucket via
`tests/unit/test_residual_fit.py`.

Feature gaps catalogued during the audit (legacy features not yet
ported, all deferred — not blocking current fixture coverage):
- `NoisyDetector` per-filter + `int_phot_flux` photon-flux scaling
  (canonical `multi_aperture_psf.py:260-266, 463-505`)
- HCIPy atmosphere wiring — pipeline.py:80 has a placeholder; legacy
  `sample()` accepts an `atmos` arg
- Xinetics DM corrector (`make_xinetics_influence_functions` +
  lstsq actuator selection) used by ELF/minielf
- `aprox_ptt_with_dm` PTT→DM fitting (depends on Xinetics)
- Lyot and `perfect` coronagraphs — VVC + Identity are present, Lyot
  exists in legacy `variants/vampires_lyot_*` but not in v2
- `convolve_im` (extended-source PSF convolution), `include_fft` (FFT
  channels), `pow_scale` (dynamic-range compression), `gauss_noise`
  (post-norm Gaussian noise) — all from legacy `extra_processing`

NOT a gap: legacy auto-derives the Zernike basis diameter from
aperture geometry; v2 makes it a YAML field. Equivalent semantics,
moved from runtime derivation to config.

### Added
See per-commit entries below.

### Audit notes (no code changes)

- **Variant-side cache audit** (audit campaign #15): walked the
  per-init `self.*` attributes in `variants/coro__coro_mas_psf.py` and
  `variants/fiber_rms__multi_aperture_psf.py` (the two highest-
  divergence variants) and confirmed each cached quantity has an
  equivalent in v2. Items checked:
  - `pupil_grid`, `aper`, `segments`, `sm`, `aper_area` → all built
    once in the v2 loader (aperture + focal plane construction).
  - `dm_basis`, `dm`, `actuate_scale` → cached in
    `ZernikeCorrector._bind_pupil_grid` and `SegmentedPTTCorrector`.
  - `lyot_mask` → built once in the coronagraph `_bind_pupil_grid`
    via `_resolve_lyot_field`.
  - `lam_setups[i]` (per-filter HCIPy artefacts: focal grid,
    propagator, wavefronts, filter_lams, reference PSF stats) → all
    cached in `_LamSetup` dataclasses inside both
    `AngularFocalPlane` and `PhysicalFocalPlane`.
  - `seg_coords` (per-segment pixel `inds`, centered `xs`/`ys`,
    `offset`) → cached as `_segment_pixel_data` in
    `SegmentedPTTCorrector` (lines 204-211).
  - `multi_mode_fiber` → built lazily and cached by
    `FiberDualOutputTap` on first `extract()`.
  Only items NOT cached in v2 are tied to as-yet-unported features
  (DM influence matrix, `dm_act_selection` mask, `aprox_ptt_with_dm`,
  `NoisyDetector`) — those are tracked as feature gaps, not cache gaps.
  One config-style change worth noting: legacy auto-derives the
  Zernike basis diameter from the aperture geometry
  (`self.diameter = 1.01 * max-pairwise(outer-5%-pixels)`); v2 expects
  it as the YAML's `zernike_diameter` field. Equivalent semantics,
  just moved from runtime derivation to user-supplied config.

### Added (audit campaign — v2.0 → legacy parity sweep)

- `tests/unit/test_vortex_coronagraph_parity.py`: pins
  `VortexCoronagraphImpl` against a direct `hcipy.VortexCoronagraph` call
  built the legacy way (`variants/coro__coro_mas_psf.py:189`). Five
  assertions: pupil-plane electric field (atol=1e-14), focal-plane
  intensity (rtol=1e-12), charge-pass-through with charge=4, Lyot
  supersample override, and unbound `apply()` raising. The wrapper has
  no math of its own — these are structural-faithfulness tests. No bugs
  surfaced.
- `tests/unit/test_vector_vortex_coronagraph_parity.py`: pins
  `VectorVortexCoronagraphImpl` against direct
  `hcipy.VectorVortexCoronagraph` calls. Legacy reference uses charge=4
  (all three active variants: vampires_vvc / scexao_vvc /
  fp_rl_ff_vvc). Four assertions: pupil-plane field parity (atol=1e-14),
  focal intensity parity (rtol=1e-12), charge-pass-through with both
  charge=2 and charge=4 (incl. default check), unbound `apply()`
  raising. No bugs surfaced.
- `tests/unit/test_post_normalization_parity.py`: covers four
  post-processors in `post/normalization.py` (audit campaign items
  11-14). Twelve assertions across `MaxIntensityNorm`, `MaxImageNorm`,
  `PerSampleNorm`, and `ChannelsFirst`:
  - `MaxIntensityNorm` divides per-channel by `reference_peak_intensities`
    matching legacy `psf /= lam_setup['peak_int']`; raises when extras
    missing or channel count mismatches.
  - `MaxImageNorm` matches legacy `out_samp /= out_samp.max()` for 2D
    (coro lineage); per-channel global max for 3D stacks; safe at zero.
  - `PerSampleNorm` matches legacy `(out - min) / (max - min)` for 2D;
    per-channel for 3D (matches legacy per-filter then concat behavior
    on single-channel-per-filter inputs); safely returns zeros for a
    constant channel.
  - `ChannelsFirst` transposes (H, W, C) → (C, H, W) for PyTorch
    (canonical 2024-09 addition — was the only C-bucket component);
    2D passthrough; round-trippable. Closes the last C-bucket gap.
  No bugs surfaced.
- `tests/unit/test_identity_coronagraph_parity.py`: pins
  `IdentityCoronagraph` as a true passthrough. Four assertions:
  `apply(wf) is wf` (identity, not just equality), aberrated input
  preserved bit-for-bit, `_bind_pupil_grid` is a no-op and idempotent,
  constructor swallows extra kwargs (loader-friendly). No bugs surfaced.
- `tests/unit/test_intensity_output_tap_parity.py`: pins
  `IntensityOutputTap` against the canonical legacy stacking pattern
  (`TelescopeSim/.../multi_aperture_psf.py:520-523`,
  `Xs += [out_samp[..., None]]` then `np.concatenate(Xs, axis=2)`).
  Eight assertions: single-focal-plane (H, W, 1) shape, multi-focal-plane
  channels-last stack, channel order follows the config list (NOT dict
  order), `.intensity` is the source (not per-WF wavefronts), and clear
  errors for empty names / missing focal plane / wrong input type. No
  bugs surfaced.
- `tests/unit/test_segmented_circular_aperture_parity.py`: pins
  `SegmentedCircularAperture` against the canonical legacy construction
  (`TelescopeSim/.../multi_aperture_psf.py:146-166, 212-242`). Ten
  assertions: aperture field matches legacy supersampled
  `make_segmented_aperture` (rtol=0, atol=0), per-segment masks each
  match the legacy fields, default supersample is 16, ELF ring centers
  match `linspace(0, 2pi, N+1)[:-1]`, area = N · π(D/2)² and metadata
  carries geometry, spider applies to the aperture mask but NOT to
  segment masks (legacy convention since commit 33914ee), spider
  `angle` in degrees → radians conversion, custom layout matches user
  positions, and validation paths (elf without n_segments, unknown
  layout). No bugs surfaced.
- `tests/unit/test_angular_focal_plane_parity.py`: pins
  `AngularFocalPlane` against the canonical legacy construction at
  `TelescopeSim/telescope_sim/multi_aperture_psf.py:249-276`. Nine
  assertions: arcsec→radians via `* np.pi / (180 * 3600)`, focal grid
  via `hcipy.make_uniform_grid` (NOT `make_pupil_grid`), propagator
  built WITHOUT focal_length (vs PhysicalFocalPlane which sets it),
  broadband formula
  `central * linspace(1 - h/2, 1 + h/2, N)`, num_samples=1
  shortcut, no `total_power=1` override (vs PhysicalFocalPlane's
  configurable normalization), per-wavelength chain loop matches the
  legacy `_psf` body bit-for-bit, reference-PSF peak+sum caching, and
  unbuilt-state error. No bugs surfaced.
- `tests/unit/test_physical_focal_plane_parity.py`: pins
  `PhysicalFocalPlane` against the legacy fiber variant
  (`variants/fiber_rms__multi_aperture_psf.py:267-269, 290`). Nine
  assertions: focal_grid construction
  (`make_pupil_grid(focal_res, focal_extent)`), propagator-with-
  focal_length cross-check (atol=1e-30), broadband-wavelength formula
  (`central * linspace(1-h/2, 1+h/2, N)`), `wavefront_total_power=1.0`
  honored (legacy hardcodes total_power=1), default total_power left
  alone when `None`, num_samples=1 path, reference-PSF caching
  (peak + sum), unbuilt-state error, and the per-wavelength chain loop
  matching the legacy `for wf in wfs: ... wf_foc = prop(wf_sm)`. No
  bugs surfaced.
- `tests/unit/test_zernike_corrector_parity.py`: pins `ZernikeCorrector`
  against the legacy DM construction
  (`variants/coro__coro_mas_psf.py:144-148, 328`). Ten assertions:
  basis matches legacy bit-for-bit (rtol=0, atol=0), per-mode peak
  normalization is exactly 1.0, `set_actuators(v)` writes
  `v * actuate_scale` internally and the `actuators` getter returns
  caller-facing `v` (not the scaled internal value),
  `flatten()` zeros both views, `apply()` matches a direct
  `hcipy.DeformableMirror(basis)(wf)` call (atol=1e-15),
  `starting_mode=1` vs `starting_mode=2` produce the right basis
  (piston vs tip — concrete cross-checks). Plus validation paths
  for wrong-length actuators and unbound calls. No bugs surfaced —
  prior fixes from commit 33914ee (the `/2` OPD factor + lstsq
  switchover) are independently covered by
  `tests/unit/test_residual_fit.py`.
- `tests/unit/test_external_pupil_aperture_parity.py`: pins
  `ExternalPupilAperture` against both legacy paths:
  `hcipy.evaluate_supersampled(aper_func(), grid, 8)` (coro lineage,
  callable mode) and `vp.generate_pupil(outer=..., pupil_grid=...)`
  (VAMPIRES lineage, field mode). Nine assertions: bit-for-bit field
  match against the legacy callable path, supersample-default-is-16
  cross-check, field-mode call-signature verification (via a
  CALL_LOG-instrumented stub module), dotted-module-AND-filesystem-path
  loading parity, plus 4 validation tests (unknown mode, missing
  module, missing path, missing function). No bugs surfaced.
- `tests/unit/test_fiber_dual_output_tap_parity.py`: pins
  `FiberDualOutputTap` against the legacy fiber loop
  (`variants/fiber_rms__multi_aperture_psf.py:369-386`). Five
  assertions: stacked focal+mmf output matches the legacy
  `np.stack([focal_total, mmf_total])` (focal channel atol=1e-14, mmf
  channel rtol=1e-12), `max_in_cache` override is honored on the
  underlying `hcipy.StepIndexFiber`, the v2 focal channel equals the
  pre-summed `FocalPlaneResult.intensity` (no double-counting),
  and clear errors for missing focal plane / unknown fiber type. Marked
  `@pytest.mark.fiber` (LP-fiber mode solves dominate runtime, ~2.5 min).
  No bugs surfaced.

## [2.0.0a5] - 2026-05-17

### Fixed
- Strehl estimators now reproduce the canonical
  `TelescopeSim/telescope_sim/multi_aperture_psf.py:_strehl` formulas
  bit-for-bit. Two real bugs were latent in the previous functions:
  - **Peak mode** used `np.max(psf) / reference_peak` instead of the
    legacy `psf.flat[reference_argmax] / reference_peak`. A PSF moved
    off the reference position by tip-tilt has the same `np.max`, so
    the previous implementation reported Strehl ~1.0 for tilts that
    should have read substantially lower. Fixture #16 shows a 0.5 µm
    global tip on a 3-segment aperture now correctly reads 0.658
    instead of ~1.0.
  - **Core mode** computed a flat sum-of-energy ratio
    `Σ(psf[core]) / Σ(ref[core])`. The legacy formula is a matched-
    filter projection `Σ(psf[core] · ref[core]) / Σ(ref[core]²)` —
    pixels are upweighted by the reference brightness, not treated
    equally. The mask center also moved from `argmax(reference_psf)`
    back to the focal-grid origin to match legacy.
- All reference-PSF-derived quantities (argmax, core mask, weighted
  sums) are precomputed once at construction in cached
  `_PeakStrehl` / `_MatchedFilterStrehl` estimator objects, so the
  per-sample work stays O(1) for peak and O(core_pixels) for
  matched-filter — mirroring the legacy `lam_setup` cache and
  avoiding a regression on RL rollouts that query Strehl every step.

### Added
- Legacy-environment fixture `16_strehl_zernike`. Captures both Strehl
  modes through the canonical `MultiAperturePSFSampler._strehl` over an
  8-case PTT actuation sequence (tip, tilt, piston, differential
  piston, combined). Pinned in
  `fixtures/runner/digests/16_strehl_zernike/expected.json` and
  reproduced by `tests/fixtures/test_canonical.py::test_strehl_zernike_v2_reproduces_digest`.
- Twelve new unit tests in `tests/unit/test_strehl_parity.py`
  (toy-PSF, no HCIPy) that hand-verify each formula and the schema's
  backwards-compat behavior.

### Changed
- New `strehl_method: "peak" | "matched_filter"` config field on
  `SimConfig`. Explicit selection replaces the implicit
  "is `strehl_core_rad` None?" branch in the pipeline.
- A pydantic `@model_validator` auto-promotes legacy YAMLs that set
  `strehl_core_rad` without `strehl_method` to `matched_filter`, and
  rejects `matched_filter` without a positive `strehl_core_rad`.

## [2.0.0a4] - 2026-05-14

### Fixed
- `Corrector.fit_surface` now subtracts the aperture-masked mean of
  the input OPD before the lstsq fit (`ZernikeCorrector`,
  `SegmentedPTTCorrector`). Defensive against non-zero-mean inputs —
  atmosphere phase screens and fit-source corrector surfaces can both
  carry a mean offset, which previously got absorbed into non-piston
  basis modes and distorted them. Uniform offsets are unobservable
  in Fraunhofer PSFs, so this is the correct semantic. Idempotent
  with the existing post-fit per-segment mean removal in
  `SegmentedPTTCorrector`.
- New tests `test_zernike_fit_surface_immune_to_constant_offset` and
  `test_segmented_ptt_fit_surface_immune_to_constant_offset` verify
  that adding a large constant to the input doesn't change recovered
  actuator amplitudes.

### Changed
- Tolerances on the existing `test_residual_fit.py` actuator-level
  assertions relaxed from `atol=1e-10` to `atol=1e-5`. The pre-fit
  mean subtract introduces a rank-1 perturbation in the lstsq
  solution (Zernike modes `Z_2..Z_n` aren't exactly zero-mean over
  the *discrete* aperture, so subtracting the input mean shifts the
  recovered coefficients by ~1e-6). PSF-level assertions retain
  their existing tolerance (phase noise at this scale produces
  intensity noise ~1e-12, well below the PSF tolerance).

## [2.0.0a3] - 2026-05-13

### Added
- Cumulative pupil-plane OPD tracking in `TelescopeSim.sample()`,
  enabling two previously-stubbed paths:
  - `wavefront_role="fit"` correctors are now resolved at sample
    time. `fit_source` accepts the special value
    `"cumulative_phase_pre_self"` (default for residual-fit targets,
    fits to the cumulative OPD from all earlier correctors in the
    chain) or another corrector's name (must appear earlier in the
    chain; raises `ValueError` for forward references or unknown
    names).
  - `target_strategy="actuators_plus_residual_fit"` and
    `"residual_fit_only"` Y echoes are now computed from per-corrector
    cumulative-OPD snapshots — matches the legacy v1
    `out_actuate = caller + matching_fit(atmos)` semantic. ML targets
    report the wavefront state in the corrector's basis; the model
    trainer applies `-Y` downstream to drive corrections.
- Headline regression test
  `tests/unit/test_residual_fit.py::test_three_identical_zernike_fit_cancels_impose`:
  three identical Zernike DMs over a clean circular aperture, two
  `impose` with random actuators, one `fit` with
  `cumulative_phase_pre_self`; verifies the fit DM's actuators land at
  `-(a1 + a2)` exactly and PSF matches the at-rest reference to
  machine precision. Plus 7 supporting tests covering the matching
  convention, all three target-strategy echo formulas (including the
  ML-residual semantic with idle / perfect-cancellation / partial-
  cancellation sub-cases), and `fit_source` variants.

### Changed
- New ``@pytest.mark.fiber`` marker for the MMF fixture, plus a
  matching ``--runfiber`` flag in ``tests/conftest.py``. The fiber/MMF
  fixture dominated ``--runslow`` runtime (~14 min total) and isn't run
  by GitHub CI. ``pytest --runslow`` now runs the other 10 canonical
  fixtures only; opt in to fiber with ``pytest --runslow --runfiber``.

### Fixed
- `ZernikeCorrector.fit_surface` (latent — previously unreachable
  through the pipeline):
  - Missing `/2` surface→OPD round-trip factor added; pipeline feeds
    OPD = 2×surface, so the factor must live inside the method to
    match the v1 `_aprox_via_dm` precedent.
  - Switched from diagonal mode projection (approximate when modes
    aren't strictly orthogonal on a discrete grid) to lstsq over
    aperture-masked pixels, matching legacy `_aprox_via_dm` and
    giving exact recovery on clean apertures.
- `SegmentedPTTCorrector.set_actuators` and `actuators`
  property — actuator layout bug. HCIPy's `SegmentedDeformableMirror`
  stores actuators in **block** layout
  (`[p_0..p_{n-1}, t_0..t_{n-1}, T_0..T_{n-1}]`), but both the setter
  (row-major `reshape(-1)`) and getter (`reshape(n, 3)`) treated it
  as row-major, scrambling P/T/T across segments whenever the caller
  set non-zero values. Switched to HCIPy's `set_segment_actuators`
  API (the v1 pattern) and block-layout de-interleaving in the
  getter. Canonical regression digests unaffected (all canonical
  fixtures sample at zero actuators).
- `SegmentedPTTCorrector._bind_pupil_grid` — segment-mask overlap.
  Previously identified each segment's pixels via `_sm.segments[i]
  != 0`, but anti-aliased segment masks extend across geometric
  boundaries, so touching segments (e.g. the elf_15seg ring where
  center-to-center spacing equals segment diameter) shared pixels.
  Now argmax-assigns each transmitting pixel to its single dominant
  segment, making per-segment fits disjoint and exact.
- Clarified `Corrector.fit_surface` ABC docstring: returns *matching*
  (positive) caller-facing actuator values for the input pupil-plane
  OPD; the pipeline negates at the apply site for
  `wavefront_role="fit"`. Y-echo formulas (`actuators + fit_surface`
  etc.) consume the unnegated value — Y reports the wavefront state,
  the ML controller applies `-Y` to cancel.

## [2.0.0a2] - 2026-05-13

### Changed
- Repository URLs in `pyproject.toml`, `README.md`, and `CHANGELOG.md`
  point at `github.com/icunnyngham/telescope-sim`.

### Fixed
- CI lint workflow: 24 ruff auto-fixes, 20 file reformats, per-file
  PLC0415 ignore for `tests/` and `fixtures/runner/` (intentional lazy
  / sys.path-manipulating imports), and `noqa` on three legitimate
  deferred-import sites in `src/` (pipeline ↔ loader cycle breaks).
- CI docs workflow: install `pandoc` (nbsphinx needs it).
- Ruff isort config: explicit `src = ["src"]` plus
  `known-first-party`/`known-third-party` lists so the classification
  matches across local-dev (where a gitignored `hcipy/` clone shadows
  site-packages) and CI environments.

### Added
- Read the Docs configuration (`.readthedocs.yaml`): Ubuntu 24.04 +
  Python 3.12 + pandoc apt package + pip install of `.[doc]`.

## [2.0.0a1] - 2026-05-13

### Added
- Initial package scaffolding: `pyproject.toml`, `src/telescope_sim/` package
  layout, dev environment YAML, CI workflows, MIT license.
- Regression-digest infrastructure (`fixtures/runner/digest_lib.py`) with a
  JSON-serializable v1.0 schema and round-trip / drift-detection tests.
- 11 captured fixture digests covering the canonical (mini-ELF), DM,
  Zernike-DM + custom-pupil, vortex/vector-vortex coronagraph, and
  fiber-coupled output paths.
- Canonical optical path: `segmented_circular` aperture (mini-ELF + custom
  layouts), `segmented_ptt` corrector with role/fit support, `angular`
  focal plane (multi-wavelength broadband sampling + reference-PSF
  computation), `intensity` output tap (channels-last per-filter stack),
  and the canonical normalization post-processors (`max_intensity_norm`,
  `max_image_norm`, `per_sample_norm`, `channels_first`).
- Pipeline orchestrator (`TelescopeSim`) supporting at-rest sample
  + per-corrector role-based actuation + peak-pixel and core-integral
  Strehl ratios.
- Pydantic v2 config schema + YAML loader + registered-implementation
  resolution. Packaged `elf_15seg` preset for the mini-ELF base case.
- Fixture regression suite (`tests/fixtures/test_canonical.py`)
  reproducing fixtures #01, #02, #10, #11 within numerical tolerance
  against their committed legacy digests.
- `segmented_circular` aperture: optional spider config (two perpendicular
  spiders) matching the canonical implementation.
- `external_pupil` aperture: wraps an arbitrary callable that produces an
  HCIPy aperture field or aperture function. Supports importing from a
  dotted module name or a filesystem path. Unlocks fixtures that use
  `miles_pupil`, `miles_synthpsf`, or HCIPy's built-in
  `make_keck_aperture` / `make_obstructed_circular_aperture`.
- `zernike` corrector: Zernike-mode deformable mirror with peak-normalized
  modes and pupil-grid late binding (so configs stay HCIPy-object-free).
- `vortex`, `vector_vortex`, and `identity` coronagraphs. Reference PSFs
  are generated with the coronagraph bypassed (legacy convention).
- `physical` focal plane: metric focal grid + explicit `focal_length`,
  with an optional `wavefront_total_power` for variant-specific
  normalization. Retains per-wavelength focal Wavefronts for fiber-style
  taps.
- `fiber_dual` output tap: stacks focal-plane intensity with multi-mode
  fiber coupling along axis 0. Matches the legacy fiber variant's
  `(2, H, W, 1)` layout.
- Regression suite now covers all 10 in-scope fixtures (`#01`–`#15`
  excluding `#04`/`#05`/`#06` pre-HCIPy prototypes and `#12` which has
  broken variant code).
- Sphinx documentation: getting-started, concepts, configuration
  reference, and an auto-built tutorial suite (`docs/tutorials/`) with
  one notebook per major capability path (canonical mini-ELF, vortex
  coronagraph, miles_pupil + Zernike DM, fiber MMF). Notebooks are
  pre-executed and rendered by `nbsphinx`.
- `telescope_sim.legacy.SimulateMultiApertureTelescope` — a best-effort
  v1.x compatibility shim that maps the common kwargs (ELF / monolithic
  layouts, single filter, segmented PTT) to a v2 config. Emits a
  `DeprecationWarning`; explicitly rejects v1 kwargs it can't represent
  yet.

[Unreleased]: https://github.com/icunnyngham/telescope-sim/compare/v2.2.0...HEAD
[2.2.0]: https://github.com/icunnyngham/telescope-sim/compare/v2.1.1...v2.2.0
[2.1.1]: https://github.com/icunnyngham/telescope-sim/compare/v2.1.0...v2.1.1
[2.1.0]: https://github.com/icunnyngham/telescope-sim/compare/v2.0.2...v2.1.0
[2.0.2]: https://github.com/icunnyngham/telescope-sim/compare/v2.0.1...v2.0.2
[2.0.1]: https://github.com/icunnyngham/telescope-sim/compare/v2.0.0a9...v2.0.1
[2.0.0a9]: https://github.com/icunnyngham/telescope-sim/compare/v2.0.0a8...v2.0.0a9
[2.0.0a8]: https://github.com/icunnyngham/telescope-sim/compare/v2.0.0a7...v2.0.0a8
[2.0.0a7]: https://github.com/icunnyngham/telescope-sim/compare/v2.0.0a6...v2.0.0a7
[2.0.0a6]: https://github.com/icunnyngham/telescope-sim/compare/v2.0.0a5...v2.0.0a6
[2.0.0a5]: https://github.com/icunnyngham/telescope-sim/compare/v2.0.0a4...v2.0.0a5
[2.0.0a4]: https://github.com/icunnyngham/telescope-sim/compare/v2.0.0a3...v2.0.0a4
[2.0.0a3]: https://github.com/icunnyngham/telescope-sim/compare/v2.0.0a2...v2.0.0a3
[2.0.0a2]: https://github.com/icunnyngham/telescope-sim/compare/v2.0.0a1...v2.0.0a2
[2.0.0a1]: https://github.com/icunnyngham/telescope-sim/releases/tag/v2.0.0a1
