"""dLux/JAX compute backend (optional; ``pip install telescope-sim[dlux]``).

Select with ``backend: dlux`` in a config YAML (or ``backend="dlux"`` on
:meth:`TelescopeSim.from_yaml` / :meth:`~TelescopeSim.from_preset`). The
pipeline, config schema, apertures, correctors, output taps, and
post-processors are shared with the default hcipy backend; only wavefront
propagation runs on JAX. Correctors compose as summed pupil-plane OPD
(thin phase screens commute), so results match the hcipy backend's
sequential ``apply()`` chain exactly for all mirror-surface correctors.

Importing this package enables 64-bit JAX globally (parity-first default;
a float32 speed knob can come later) and registers the backend's focal
planes in the registry overlay.
"""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

# The dLux dependency anchors this backend's optional-install contract and
# the planned differentiable-model export; propagation itself is a direct
# JAX port of the Fraunhofer MFT.
import dLux  # noqa: F401, E402

from telescope_sim.backends.dlux import focal_planes  # noqa: F401, E402  (registry side effect)

__all__ = ["focal_planes"]
