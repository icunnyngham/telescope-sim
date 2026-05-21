API reference
==============

Top-level
---------

.. automodule:: telescope_sim
   :members: TelescopeSim, register, registry

Pipeline
--------

.. automodule:: telescope_sim.pipeline
   :members:

Abstract base classes
---------------------

.. automodule:: telescope_sim.abc
   :members:

.. automodule:: telescope_sim.abc.aperture
   :members:

.. automodule:: telescope_sim.abc.corrector
   :members:

.. automodule:: telescope_sim.abc.coronagraph
   :members:

.. automodule:: telescope_sim.abc.focal_plane
   :members:

.. automodule:: telescope_sim.abc.output_tap
   :members:

.. automodule:: telescope_sim.abc.post_processor
   :members:

Configuration
-------------

.. automodule:: telescope_sim.config.schema
   :members:

.. automodule:: telescope_sim.config.loader
   :members:

Concrete implementations
------------------------

.. automodule:: telescope_sim.apertures.segmented_circular
   :members:

.. automodule:: telescope_sim.apertures.external_pupil
   :members:

.. automodule:: telescope_sim.correctors.segmented_ptt
   :members:

.. automodule:: telescope_sim.correctors.zernike
   :members:

.. automodule:: telescope_sim.coronagraphs.standard
   :members:

.. automodule:: telescope_sim.focal_planes.angular
   :members:

.. automodule:: telescope_sim.focal_planes.physical
   :members:

.. automodule:: telescope_sim.outputs.intensity
   :members:

.. automodule:: telescope_sim.outputs.fiber_dual
   :members:

.. automodule:: telescope_sim.post.normalization
   :members:

.. automodule:: telescope_sim.post.noisy_detector
   :members:

.. automodule:: telescope_sim.post.convolve
   :members:

.. automodule:: telescope_sim.strehl
   :members:

Registry
--------

.. automodule:: telescope_sim.registry
   :members:
