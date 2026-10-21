"""Environment diagnostics that work before MLX is installed."""
from __future__ import annotations

import importlib.metadata
import json
import platform
import shutil
import sys

import psutil

from .config import data_dir


def diagnose() -> dict:
    checks = []
    supported = platform.system() == "Darwin" and platform.machine() == "arm64"
    checks.append({"name": "Apple Silicon", "ok": supported,
                   "detail": f"{platform.system()} {platform.machine()}",
                   "fix": "Use native arm64 Python on an Apple Silicon Mac." if not supported else None})
    checks.append({"name": "Python", "ok": sys.version_info >= (3, 12), "detail": platform.python_version(),
                   "fix": "uv python install 3.12" if sys.version_info < (3, 12) else None})
    versions = {}
    for package in ("deepinfer-mac", "mlx", "mlx-lm", "fastapi"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    metal_ok = False
    metal_detail = "MLX is not installed. Run: uv sync --extra mlx --extra dev"
    if versions["mlx"] and supported:
        try:
            import mlx.core as mx
            metal_ok = mx.metal.is_available()
            metal_detail = f"Metal available: {metal_ok}"
        except (ImportError, RuntimeError, OSError) as exc:
            metal_detail = str(exc)
    checks.append({"name": "Metal", "ok": metal_ok, "detail": metal_detail,
                   "fix": None if metal_ok else "Install the mlx extra and use a supported macOS version."})
    parent = data_dir()
    while not parent.exists():
        parent = parent.parent
    disk = shutil.disk_usage(parent)
    memory = psutil.virtual_memory()
    checks.append({"name": "Free disk", "ok": disk.free >= 8 * 1024**3,
                   "detail": f"{disk.free / 1024**3:.1f} GiB free; allow at least 8 GiB for weights and runtime"})
    return {"ok": all(c["ok"] for c in checks), "checks": checks, "versions": versions,
            "memory_total_bytes": memory.total, "memory_available_bytes": memory.available,
            "data_dir": str(data_dir()), "platform": platform.platform()}


def print_diagnostics(as_json: bool = False) -> bool:
    report = diagnose()
    if as_json:
        print(json.dumps(report, indent=2))
    else:
        for check in report["checks"]:
            print(f"{'OK' if check['ok'] else 'FAIL'}  {check['name']}: {check['detail']}")
            if check.get("fix"):
                print(f"      {check['fix']}")
        print(f"Memory: {report['memory_total_bytes'] / 1024**3:.1f} GiB total, "
              f"{report['memory_available_bytes'] / 1024**3:.1f} GiB currently available")
        print(f"Data: {report['data_dir']}")
    return report["ok"]
