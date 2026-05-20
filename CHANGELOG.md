# Changelog

All notable changes to `telescope-sim` are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project adheres
to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

[Unreleased]: https://github.com/icunnyngham/telescope-sim/compare/v2.0.0a2...HEAD
[2.0.0a2]: https://github.com/icunnyngham/telescope-sim/releases/tag/v2.0.0a2
[2.0.0a1]: https://github.com/icunnyngham/telescope-sim/releases/tag/v2.0.0a1
