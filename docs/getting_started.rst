Getting started
================

Installation
------------

From PyPI::

    pip install telescope-sim

From source::

    git clone https://github.com/icunnyngham/telescope-sim.git
    cd telescope-sim
    conda env create -f envs/env-dev.yaml
    conda activate telescope-sim-dev
    pip install -e ".[dev,doc]"

Quick start
-----------

The simplest way to generate a PSF is to load a packaged preset and call
``sample()``::

    from telescope_sim import TelescopeSim
    import numpy as np

    sim = TelescopeSim.from_preset("elf_15seg")

    out = sim.sample(meas_strehl=True)
    psf = out["images"]["psf"]        # (H, W, n_filters) channels-last
    strehls = out["strehls"]          # dict {filter_name: strehl_ratio}

To apply actuator values to a corrector, pass an ``actuations`` dict keyed
by corrector name (matching the names in your config)::

    ptt_errors = np.random.normal(scale=0.1, size=(15, 3))
    out = sim.sample(actuations={"segments": ptt_errors}, meas_strehl=True)

For a custom setup, write a YAML config and load it directly::

    sim = TelescopeSim.from_yaml("path/to/my_config.yaml")

See :doc:`configuration` for the full schema.

Bundled presets
---------------

The package ships with a small set of presets:

- ``elf_15seg`` — the sELF (small-ELF) array: 15 segments of 0.5 m on a
  1.5 m ring (3.5 m outer diameter), J-band filter (1.25 µm, 5%
  bandwidth) with broadband sampling at 5 wavelengths.

Listing available presets::

    from telescope_sim import TelescopeSim
    # An invalid preset name reports what is available
    TelescopeSim.from_preset("__list__")   # raises with a helpful message
