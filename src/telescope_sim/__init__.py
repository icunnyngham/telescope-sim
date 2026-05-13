"""telescope-sim — composable, config-driven simulation of multi-aperture telescope PSFs.

Built on HCIPy. The package exposes a pluggable pipeline of optical stages
(aperture, correctors, coronagraph, focal plane, output taps, post-processing)
driven by YAML configuration. Currently in alpha; concrete implementations are
under active development.
"""

from telescope_sim.pipeline import TelescopeSim
from telescope_sim.registry import register, registry

try:
    from telescope_sim._version import __version__
except ImportError:
    __version__ = "0.0.0+unknown"

__all__ = [
    "TelescopeSim",
    "register",
    "registry",
    "__version__",
]
