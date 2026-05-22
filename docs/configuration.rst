Configuration reference
========================

Pipelines are described in YAML and validated by pydantic v2. The
top-level shape is::

    pupil:
      resolution: <int>
      extent: <float>

    aperture: { type: <name>, ... }

    correctors:
      <name>:
        type: <name>
        wavefront_role: actuate | impose | fit
        target_strategy: none | actuators | actuators_plus_residual_fit | residual_fit_only
        fit_source: <corrector_name> | <wavefront_name> | null
        target: <bool>
        # ... type-specific kwargs

    corrector_chain: [<name>, <name>, ...]

    coronagraph: { type: <name>, ... }      # optional

    focal_planes:
      <name>: { type: <name>, ... }

    outputs:
      <name>:
        tap: { type: <name>, ... }
        post_processing:
          - <name>                          # bare string = no-arg post-processor
          - { type: <name>, ... }           # mapping form for parameterized ones

    strehl_method: peak | matched_filter | null
    strehl_core_rad: <float | null>

Strehl
------

``strehl_method``
    Selects the Strehl estimator. ``"peak"`` reads the current PSF at the
    *reference* PSF's argmax — so a tip-tilt that moves the peak off-center
    correctly drops Strehl below 1. ``"matched_filter"`` is a normalized
    matched-filter projection over a circular core mask of radius
    ``strehl_core_rad`` centered on the focal-grid origin; pixels are
    weighted by the reference intensity. If unset, defaults to
    ``"matched_filter"`` when ``strehl_core_rad`` is given (legacy
    behavior), else ``"peak"``.

``strehl_core_rad``
    Required when ``strehl_method == "matched_filter"``; ignored for
    ``"peak"``.

Per-sample state
----------------

``sample()`` accepts a few non-actuator inputs per call:

- ``actuations`` — per-corrector actuator state, keyed by corrector name.
- ``atmos`` — external atmosphere (any wf→wf callable), see
  :doc:`concepts` for the contract.
- ``output_overrides`` — per-output, per-tap-config overrides.
  Shape ``{output_name: {key: value, ...}}``. The pipeline puts the
  inner mapping on each post-processor's
  :class:`~telescope_sim.abc.PipelineContext` as ``context.overrides``;
  processors that consume per-sample state (e.g. ``noisy_detector`` reads
  ``int_phot_flux``, ``convolve_image`` reads ``convolve_image``) check
  the override and fall back to their YAML default. Processors with no
  per-sample state ignore the field.
- ``meas_strehl`` — if true, return ``out["strehls"]``.
- ``meas_pupil_opd`` — if true, return ``out["pupil_opd"]`` as an
  ``hcipy.Field`` of cumulative pupil-plane OPD in meters (atmosphere
  seed when ``phase_for`` is available, plus every DM-backed corrector's
  surface × 2). Pair with ``sim.aperture.field`` for masked display via
  ``hcipy.imshow_field(out["pupil_opd"], mask=sim.aperture.field)``; the
  bundled :func:`telescope_sim.helpers.diagnostics.plot_opd_and_psfs`
  uses this contract.

Stages
------

Apertures
~~~~~~~~~

``segmented_circular``
    Fields: ``segment_diameter``, ``layout`` (``"elf"`` or ``"custom"``),
    ``n_segments``, ``ring_radius``, ``positions``, ``supersample``,
    ``spider`` (optional ``{width, angle}``).

``external_pupil``
    Fields: ``module`` (dotted name or path to a ``.py`` file),
    ``function``, ``mode`` (``"field"`` or ``"callable"``), ``kwargs``,
    ``area``, ``supersample`` (only when ``mode == "callable"``).

Correctors
~~~~~~~~~~

``segmented_ptt``
    Fields: ``piston_scale``, ``tip_tilt_scale``. Actuator shape
    ``(n_segments, 3)``.

``zernike``
    Fields: ``n_modes``, ``zernike_diameter``, ``starting_mode``,
    ``actuate_scale``.

Coronagraphs
~~~~~~~~~~~~

``identity``
    Passthrough (no-op).

``vortex`` / ``vector_vortex``
    Fields: ``charge``, ``lyot`` (an aperture sub-config; built into a
    transmission field).

Focal planes
~~~~~~~~~~~~

``angular``
    Fields: ``central_lam``, ``focal_extent`` (arcsec), ``focal_res``,
    ``fractional_bandwidth``, ``num_samples``.

``physical``
    Fields: same as ``angular`` plus ``focal_length`` (meters),
    ``focal_extent`` in meters, optional ``wavefront_total_power``.

Output taps
~~~~~~~~~~~

``intensity``
    Stack PSF intensities from one or more focal planes channels-last.
    Fields: ``focal_planes`` (list of names).

``fiber_dual``
    Stack focal intensity + multi-mode-fiber-coupled intensity along
    axis 0. Fields: ``focal_plane_name``, ``fiber``
    (``{type: step_index, core_radius, NA, fiber_length, max_in_cache}``).

Post-processors
~~~~~~~~~~~~~~~

``max_intensity_norm``
    Divide each channel by the reference PSF peak.

``max_image_norm``
    Divide each channel by its own peak.

``per_sample_norm``
    Min-max normalize each channel to [0, 1].

``channels_first``
    Transpose ``(H, W, C) → (C, H, W)``.

``noisy_detector``
    Apply ``hcipy.NoisyDetector`` (read noise, dark current, flat-field,
    photon shot noise) to a single-channel intensity image.

    Fields:

    - ``int_phot_flux`` (float | null) — photon flux in photons/m². When
      set, rescales the input image so total accumulated charge =
      ``int_phot_flux * aperture_area`` (legacy ``_addNoiseToObservation``
      contract). ``None`` integrates at the natural input scale (useful
      when an upstream ``convolve_image`` has imposed absolute
      brightness).
    - ``detector`` (mapping) — forwarded to ``hcipy.NoisyDetector``:
      ``read_noise``, ``dark_current_rate``, ``flat_field``,
      ``include_photon_noise``, ``subsampling``.
    - ``clamp_nonnegative`` (bool, default ``True``) — apply ``np.abs``
      to read-out (legacy workaround for Gaussian read noise pulling
      pixels negative). Disable for science-faithful analyses.

    Single-focal-plane only: the processor needs one focal grid. For
    multi-filter noisy outputs, declare one output (with its own
    ``noisy_detector``) per filter.

    Per-sample override: ``output_overrides={"<output_name>":
    {"int_phot_flux": <flux>}}``.

``convolve_image``
    Convolve the focal-plane PSF (as a kernel, normalized by the at-rest
    reference PSF sum) with a caller-supplied scene. Equivalent to legacy
    ``fftconvolve(scene, psf / reference_psf_sum, mode='same')``.

    Fields:

    - ``image`` (array-like | null) — default 2D scene to convolve with.
      ``None`` means no default; the caller must supply ``convolve_image``
      via ``output_overrides`` per sample.

    Single-focal-plane only (kernel normalization needs a single
    ``reference_psf_sum``). For multi-filter convolution, split into one
    output (with its own ``convolve_image``) per filter.

    Per-sample override: ``output_overrides={"<output_name>":
    {"convolve_image": <2D array>}}``.

    Composes cleanly with ``noisy_detector`` for extended-source noisy
    imaging — the recommended order is
    ``intensity → convolve_image → noisy_detector``.

Examples
--------

See ``fixtures/configs/`` in the repo for working configs that reproduce
each of the legacy fixtures.
