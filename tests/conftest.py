"""Pytest configuration and shared fixtures."""

from __future__ import annotations

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-api-tests",
        action="store_true",
        default=False,
        help="Run tests that make real API calls (will charge your account).",
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    if config.getoption("--run-api-tests"):
        return  # user explicitly opted in — run everything
    skip = pytest.mark.skip(reason="Real API calls skipped. Use --run-api-tests to run.")
    for item in items:
        if item.get_closest_marker("api"):
            item.add_marker(skip)
