"""Shared pytest configuration.

Adds the ``--runslow`` flag (skips slow tests by default; opt-in for CI nightly).
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


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if config.getoption("--runslow"):
        return
    skip_slow = pytest.mark.skip(reason="needs --runslow to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)
