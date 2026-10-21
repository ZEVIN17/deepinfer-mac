"""Freeze DS experiment runtime, exact harnesses and independent source oracle.

Never overwrites a previous archive or extraction. Run after coordinated source
edits stop, then execute from --extract with its src on PYTHONPATH. This helper
freezes code only; it does not run inference or create benchmark measurements.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
import tarfile
from pathlib import Path


def freeze(paths, output, extract):
    if output.exists() or extract.exists():
        raise FileExistsError("Use fresh archive/extraction paths; historical evidence is immutable")
    contents = {str(p): p.read_bytes() for p in sorted(paths)}
    hashes = {name: hashlib.sha256(blob).hexdigest() for name, blob in contents.items()}
    manifest = {
        "files": hashes,
        "tree_sha256": hashlib.sha256(json.dumps(hashes, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "scope": "DS runtime, locks, experiment harnesses, independent unmodified C++ oracle and licenses",
        "performance_measurements": False,
    }
    for name, blob in contents.items():
        if Path(name).read_bytes() != blob:
            raise RuntimeError(f"Source changed during freeze: {name}")
    contents["ds-source-manifest.json"] = (json.dumps(manifest, indent=2) + "\n").encode()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as gz, tarfile.open(fileobj=gz, mode="w") as archive:
        for name, blob in contents.items():
            info = tarfile.TarInfo(name)
            info.size, info.mode, info.mtime = len(blob), 0o644, 0
            archive.addfile(info, io.BytesIO(blob))
    archive_sha = hashlib.sha256(output.read_bytes()).hexdigest()
    for name, blob in contents.items():
        path = extract / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(blob)
    # Binding lives only in the local extracted manifest, never in the tar:
    # the archive cannot contain its own hash. The original internal manifest
    # remains exactly the content written before archive creation.
    bound_manifest = {**manifest, "archive_binding": {
        "path": os.path.relpath(output.resolve(), extract.resolve()),
        "sha256": archive_sha,
        "internal_manifest_sha256": hashlib.sha256(contents["ds-source-manifest.json"]).hexdigest(),
    }}
    (extract / "ds-source-manifest.json").write_text(json.dumps(bound_manifest, indent=2) + "\n")
    return {"archive": str(output), "archive_sha256": archive_sha,
            "tree_sha256": manifest["tree_sha256"], "files": len(hashes),
            "manifest": str(extract / "ds-source-manifest.json"), "extract": str(extract)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--extract", type=Path, required=True)
    args = parser.parse_args()
    paths = [p for p in Path("src/deepinfer_mac").rglob("*")
             if p.is_file() and "__pycache__" not in p.parts and p.suffix in (".py", ".json", ".html", ".js", ".css")]
    paths += [p for p in Path("tests/reference").rglob("*") if p.is_file()]
    paths += [p for p in Path("third_party/licenses").glob("*") if p.is_file()]
    paths += [Path(name) for name in (
        "LICENSE", "NOTICE", "pyproject.toml", "uv.lock", "models.lock.json", "upstream.lock.json",
        "benchmarks/experiments/ds_budget_ablation.py", "benchmarks/experiments/deepinfer_service_server.py",
        "benchmarks/experiments/prompt_token_audit.py", "benchmarks/experiments/hybrid_lagged_probe.py",
        "benchmarks/experiments/verify_lagged_trace.py", "scripts/freeze_ds_source.py",
    )]
    print(json.dumps(freeze(paths, args.output, args.extract), indent=2))


if __name__ == "__main__":
    main()
