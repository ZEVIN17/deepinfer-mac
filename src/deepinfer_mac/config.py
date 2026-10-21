"""User data paths. Model weights remain in the standard Hugging Face cache."""
from __future__ import annotations

import os
from pathlib import Path


def data_dir() -> Path:
    override = os.getenv("DEEPINFER_MAC_HOME")
    return Path(override).expanduser() if override else Path.home() / "Library/Application Support/DeepInfer-Mac"


def ensure_data_dir() -> Path:
    root = data_dir()
    root.mkdir(parents=True, exist_ok=True)
    return root
