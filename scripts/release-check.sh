#!/bin/sh
set -eu
uv sync --frozen --extra mlx --extra dev
uv run pytest -q -m 'not metal'
python3 site/build.py
python3 site/check.py
uv build
python3 scripts/verify_artifacts.py
