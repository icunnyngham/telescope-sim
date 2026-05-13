telescope-sim
==============

Composable, config-driven simulation of multi-aperture telescope PSFs,
deformable mirrors, coronagraphs, and fiber coupling — built on
`HCIPy <https://hcipy.org>`_.

The package exposes a small set of abstract base classes
(:class:`~telescope_sim.abc.Aperture`,
:class:`~telescope_sim.abc.Corrector`,
:class:`~telescope_sim.abc.Coronagraph`,
:class:`~telescope_sim.abc.FocalPlane`,
:class:`~telescope_sim.abc.OutputTap`,
:class:`~telescope_sim.abc.PostProcessor`) plus a registry of stock
implementations. Optical pipelines are described in YAML; the package
validates the config with pydantic v2 and instantiates a
:class:`~telescope_sim.TelescopeSim` ready to ``sample()``.


.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   getting_started


.. toctree::
   :maxdepth: 2
   :caption: User guide

   concepts
   configuration


.. toctree::
   :maxdepth: 2
   :caption: Tutorials

   tutorials/01_canonical_mini_elf
   tutorials/02_coronagraph_vortex
   tutorials/03_vampires_zernike
   tutorials/04_fiber_mmf


.. toctree::
   :maxdepth: 2
   :caption: API reference

   api/index


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
