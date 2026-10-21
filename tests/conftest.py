"""Portable pytest defaults; real Metal execution is always an explicit choice."""
import pytest


def pytest_addoption(parser):
    parser.addoption("--run-metal", action="store_true", help="Execute explicit real-Metal tests")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-metal"):
        return
    skip = pytest.mark.skip(reason="Real Metal test; opt in with --run-metal -m metal")
    for item in items:
        if "metal" in item.keywords:
            item.add_marker(skip)
