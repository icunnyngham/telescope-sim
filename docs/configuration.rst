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

    strehl_core_rad: <float | null>

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

Examples
--------

See ``fixtures/configs/`` in the repo for working configs that reproduce
each of the legacy fixtures.
