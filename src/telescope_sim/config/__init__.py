"""Configuration schema and YAML loader.

Pydantic v2 models define the full v2.0 config surface; a thin loader converts
YAML to validated pydantic models and then instantiates registered pipeline
stages. The validated config object is the canonical in-memory representation
of a simulation setup.
"""

from telescope_sim.config.loader import load_yaml
from telescope_sim.config.schema import SimConfig

__all__ = ["SimConfig", "load_yaml"]
