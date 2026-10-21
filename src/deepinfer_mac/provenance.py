"""Portable source hashing without recording private checkout paths."""
from __future__ import annotations

import hashlib
import json
from importlib.resources import files
from pathlib import Path


def source_manifest() -> dict:
    manifest = {}

    def visit(directory, prefix):
        for item in sorted(directory.iterdir(), key=lambda value: value.name):
            if item.name == '__pycache__':
                continue
            name = f'{prefix}/{item.name}'
            if item.is_dir():
                visit(item, name)
            elif item.name.endswith(('.py', '.json', '.js', '.html', '.css')):
                manifest[name] = hashlib.sha256(item.read_bytes()).hexdigest()

    package = files('deepinfer_mac')
    visit(package, 'src/deepinfer_mac')
    checkout = Path(__file__).resolve().parents[2]
    if (checkout / 'src' / 'deepinfer_mac').is_dir():
        for name in ('pyproject.toml', 'uv.lock', 'models.lock.json', 'upstream.lock.json'):
            path = checkout / name
            if path.is_file():
                manifest[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    canonical = json.dumps(manifest, sort_keys=True, separators=(',', ':')).encode()
    return {'algorithm': 'sha256', 'source_tree_sha256': hashlib.sha256(canonical).hexdigest(),
            'files': manifest, 'scope': 'runtime sources and available checkout locks; paths are repository relative'}
