"""JAX compute backend (optional; ``pip install telescope-sim[jax]``).

Select with ``backend: jax`` in a config YAML (or ``backend="jax"`` on
:meth:`TelescopeSim.from_yaml` / :meth:`~TelescopeSim.from_preset`). The
pipeline, config schema, apertures, correctors, output taps, and
post-processors are shared with the default hcipy backend; only wavefront
propagation runs on JAX, with kernels built from the hcipy grid geometry
so the two backends agree to float64 round-off. Correctors compose as
summed pupil-plane OPD (thin phase screens commute), so results match the
hcipy backend's sequential ``apply()`` chain exactly for all
mirror-surface correctors.

Because the propagation core is a pure JAX function over static arrays,
models built on this backend are the substrate for exporting
dLux/Zodiax-compatible differentiable models (planned; not yet built).

Importing this package enables 64-bit JAX globally (parity-first default;
a float32 speed knob can come later) and registers the backend's focal
planes in the registry overlay.
"""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

from telescope_sim.backends.jax import focal_planes  # noqa: F401, E402  (registry side effect)

__all__ = ["focal_planes"]
