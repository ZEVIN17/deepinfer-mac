#!/bin/sh
set -eu
uv sync --frozen --extra mlx --extra dev
uv run ruff check src tests scripts benchmarks/experiments benchmarks/quality
uv run pytest -q -m 'not metal'
uv run python site/build.py
uv run python site/check.py
# Device checks remain explicit: uv run pytest -q -m metal --run-metal
uv build --out-dir "${1:-dist/current}"
uv run python scripts/verify_artifacts.py --dist "${1:-dist/current}"
