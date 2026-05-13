# Changelog

All notable changes to `telescope-sim` are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project adheres
to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

[Unreleased]: https://github.com/morphoptic/telescope-sim/compare/HEAD...HEAD
