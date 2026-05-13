"""Abstract base classes for pluggable pipeline stages.

Each ABC defines the minimal interface a concrete implementation must satisfy
to be registered and consumed by the pipeline. See individual module docstrings
for semantics and the role/relationship system documented on :class:`Corrector`.
"""

from telescope_sim.abc.aperture import Aperture, ApertureResult
from telescope_sim.abc.corrector import Corrector
from telescope_sim.abc.coronagraph import Coronagraph
from telescope_sim.abc.focal_plane import FocalPlane
from telescope_sim.abc.output_tap import OutputTap
from telescope_sim.abc.post_processor import PipelineContext, PostProcessor

__all__ = [
    "Aperture",
    "ApertureResult",
    "Corrector",
    "Coronagraph",
    "FocalPlane",
    "OutputTap",
    "PostProcessor",
    "PipelineContext",
]
