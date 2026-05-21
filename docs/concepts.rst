Architecture & concepts
========================

The pipeline
------------

A ``TelescopeSim`` is a linear chain of pupil-plane stages followed by a
controlled fan-out at the pupil → focal boundary::

    [aperture (+ spider)]
            |
            v
    [correctors: c1 → c2 → ... → cN]
            |
            v
    [coronagraph (optional)]
            |
            v
       pupil_post_coro
            |
       +----+----+ ... +----+
       v    v         v
     fp_1 fp_2  ...  fp_K        ← named focal planes
       |    |         |
       v    v         v
     taps taps  ...  taps         ← OutputTap(s)
       |    |         |
       v    v         v
     post post  ...  post         ← ordered PostProcessor list per output

The chain is **strictly linear through the pupil-plane stages**. The only
fan-out is at the pupil → focal boundary, where multiple named focal
planes can consume the same pupil-plane wavefront. This subsumes the
per-filter loop, mixed angular/physical detector layouts (e.g. a science
detector + a fiber-coupling plane), and pre-coro / post-coro diagnostics.

Stages and the registry
-----------------------

Each pipeline stage type has an abstract base class
(:mod:`telescope_sim.abc`) and a small registry of concrete
implementations:

==================  ===================================
Kind                ABC
==================  ===================================
``aperture``        :class:`~telescope_sim.abc.Aperture`
``corrector``       :class:`~telescope_sim.abc.Corrector`
``coronagraph``     :class:`~telescope_sim.abc.Coronagraph`
``focal_plane``     :class:`~telescope_sim.abc.FocalPlane`
``output_tap``      :class:`~telescope_sim.abc.OutputTap`
``post_processor``  :class:`~telescope_sim.abc.PostProcessor`
==================  ===================================

To add a new implementation::

    from telescope_sim.abc import Aperture
    from telescope_sim.registry import register

    @register("aperture", "my_custom")
    class MyCustomAperture(Aperture):
        def build(self, pupil_grid):
            ...

Users can register their own implementations from their own packages
without modifying ``telescope-sim``.

The corrector role system
-------------------------

Each corrector has two orthogonal "role" attributes that control how it
participates in a sample:

``wavefront_role``
    Controls how actuator state gets set when ``sample()`` runs.

    - ``"actuate"`` — set by the ML model (caller's ``actuations`` dict).
    - ``"impose"`` — set by the caller via a separate channel (e.g. an
      environment-supplied aberration).
    - ``"fit"`` — lstsq-fit to another corrector's surface or to a named
      wavefront's phase. Requires ``fit_source``.

``target_strategy``
    Controls what this corrector contributes to the Y output dict
    returned by ``sample()``.

    - ``"none"`` — not in Y.
    - ``"actuators"`` — echo back current actuator state.
    - ``"actuators_plus_residual_fit"`` —
      ``actuators + fit_surface(cumulative_phase_pre_self)``. Useful for
      ML training where the "ideal" actuation is the model's command
      plus the residual disturbance.
    - ``"residual_fit_only"`` — just the residual fit.

This two-axis design covers the patterns observed across the legacy
variants (PTT-imposed-with-DM-approximation, cumulative residual fits
that include external atmosphere, stacked imposers with a single
fitting DM, …) without bespoke logic per case.

Reference PSF
-------------

Each focal plane precomputes its at-rest "reference PSF" once at
construction time: every corrector flattened, the coronagraph (if any)
bypassed, and the resulting summed intensity used for Strehl
normalization. Post-processors that depend on the reference (e.g.
``max_intensity_norm``) read it from the
:class:`~telescope_sim.abc.PipelineContext` passed into them.

Atmosphere
----------

Atmospheres are **not** correctors. They enter the simulation as a
per-sample input on :meth:`~telescope_sim.pipeline.TelescopeSim.sample`::

    out = sim.sample(atmos=my_atmosphere, ...)

The contract for ``atmos`` is loose by design: any wf→wf callable
(typically a ``hcipy.InfiniteAtmosphericLayer`` or
``hcipy.MultiLayerAtmosphere``, but anything that maps a
``hcipy.Wavefront`` to a modified one works) is applied at the front of
the chain, before any corrector. The caller owns time evolution — v2
holds no atmosphere state.

If the atmosphere also exposes ``phase_for(lam)`` returning
HCIPy-convention phase (``2π·OPD/lam``), its OPD is seeded into the
per-corrector cumulative-OPD stream. A downstream ``fit``-role corrector
with ``fit_source="cumulative_phase_pre_self"`` will then naturally fit
to (and, applied with the role's automatic sign-flip, cancel) the
atmosphere — no bespoke "atmosphere corrector" needed. Atmospheres
without ``phase_for`` still modify the wavefront, but only the wavefront
sees them; fit-role correctors see only the corrector chain.

The reference PSF used for Strehl normalization is never atmospheric:
it is cached once at sim-build with ``atmos=None`` implicit.
