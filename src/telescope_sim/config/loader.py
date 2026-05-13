"""YAML loader — parses a config file into a validated :class:`SimConfig`.

Currently a thin ``yaml.safe_load`` + ``SimConfig.model_validate``. Preset
resolution and registered-implementation lookup will be added once the full
schema lands.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from telescope_sim.config.schema import SimConfig


def load_yaml(path: str | Path) -> SimConfig:
    """Load and validate a YAML configuration file."""
    with open(path) as f:
        data = yaml.safe_load(f)
    return SimConfig.model_validate(data)


__all__ = ["load_yaml"]
