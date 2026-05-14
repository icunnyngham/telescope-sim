"""Shared pytest configuration.

Adds two opt-in flags:

- ``--runslow``: enables the canonical fixture regression suite (~14 min).
- ``--runfiber``: additionally enables the fiber/MMF fixture, which
  dominates the slow-suite runtime and isn't run by GitHub CI.
"""

from __future__ import annotations

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--runslow",
        action="store_true",
        default=False,
        help="Run tests marked @pytest.mark.slow (fixture regression suite).",
    )
    parser.addoption(
        "--runfiber",
        action="store_true",
        default=False,
        help="Run tests marked @pytest.mark.fiber (long fiber/MMF fixture).",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    runslow = config.getoption("--runslow")
    runfiber = config.getoption("--runfiber")
    skip_slow = pytest.mark.skip(reason="needs --runslow to run")
    skip_fiber = pytest.mark.skip(reason="needs --runfiber to run")
    for item in items:
        # Fiber check comes first: fiber tests are also slow, but
        # --runslow alone shouldn't enable them.
        if "fiber" in item.keywords and not runfiber:
            item.add_marker(skip_fiber)
        elif "slow" in item.keywords and not runslow:
            item.add_marker(skip_slow)
