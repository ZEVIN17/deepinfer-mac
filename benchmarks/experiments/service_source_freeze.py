"""Freeze complete local service code and audit actual interpreter environments.

Only standard-library metadata/file reads; environment inspection never imports
MLX or loads a model. Private filesystem paths stay in excluded local config.
"""

import gzip
import hashlib
import importlib.metadata
import io
import json
import subprocess
import sys
import tarfile
from pathlib import Path


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def source_files(root):
    result = {}
    for prefix in ("src/deepinfer_mac", "benchmarks/experiments"):
        for path in sorted((root / prefix).rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc":
                result[path.relative_to(root).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    for name in ("pyproject.toml", "uv.lock", "models.lock.json", "upstream.lock.json"):
        if (root / name).is_file():
            result[name] = hashlib.sha256((root / name).read_bytes()).hexdigest()
    return result


def freeze(workspace, destination, manifest):
    expected = source_files(workspace)
    relative = manifest.relative_to(workspace).as_posix()
    if relative not in expected:
        raise ValueError("Manifest must be inside the workspace benchmarks/experiments source tree")
    if destination.exists():
        if source_files(destination) != expected:
            raise ValueError("Existing source extraction differs; preserve it and use a fresh isolated root")
    else:
        destination.mkdir(parents=True)
        for name, wanted in expected.items():
            path = destination / name
            path.parent.mkdir(parents=True, exist_ok=True)
            data = (workspace / name).read_bytes()
            if hashlib.sha256(data).hexdigest() != wanted:
                raise ValueError("Source changed during extraction; use a fresh root")
            path.write_bytes(data)
    archive = destination.parent / "service-source.tar.gz"
    if not archive.exists():
        with (archive.open("wb") as output,
              gzip.GzipFile(fileobj=output, mode="wb", mtime=0) as compressed,
              tarfile.open(fileobj=compressed, mode="w") as tar):
            for name in sorted(expected):
                data = (destination / name).read_bytes()
                member = tarfile.TarInfo(name)
                member.size, member.mode, member.mtime = len(data), 0o444, 0
                tar.addfile(member, io.BytesIO(data))
    with tarfile.open(archive, "r:gz") as tar:
        members = tar.getmembers()
        archived = {member.name: hashlib.sha256(tar.extractfile(member).read()).hexdigest()
                    for member in members if member.isfile()}
        if archived != expected or len(members) != len(expected):
            raise ValueError("Existing source archive differs; use a fresh isolated root")
    actual = source_files(destination)
    if actual != expected or source_files(workspace) != expected:
        raise ValueError("Source changed while freezing")
    return {"root": str(destination.resolve()), "files": expected, "sha256": digest(expected),
            "archive": str(archive), "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
            "manifest_relative_path": relative}


def environment_current():
    files, versions = {}, {}
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata["Name"]
        versions[name] = distribution.version
        for item in distribution.files or ():
            if "__pycache__" in item.parts or item.suffix == ".pyc":
                continue
            path = distribution.locate_file(item)
            if path.is_file():
                files[f"{name}/{item}"] = hashlib.sha256(path.read_bytes()).hexdigest()
    details = {"versions": versions, "files": files, "python_version": sys.version,
               "executable_sha256": hashlib.sha256(Path(sys.executable).read_bytes()).hexdigest()}
    return details | {"sha256": digest(details)}


def environment(python, helper):
    process = subprocess.run([str(python), str(helper), "--environment"], check=True, capture_output=True, text=True)
    return json.loads(process.stdout)


def check(configuration, *, bind=False):
    frozen = configuration.get("frozen_source")
    if not frozen:
        return {"verified": False, "reason": "No frozen source/environment evidence"}
    root = Path(frozen["root"])
    errors = []
    alias = configuration.get("model_alias")
    if alias:
        path = Path(alias["path"])
        if not path.is_symlink() or path.resolve() != Path(alias["snapshot"]).resolve():
            errors.append("pinned model alias changed")
    actual = source_files(root)
    if actual != frozen["files"] or digest(actual) != frozen["sha256"]:
        errors.append("source bytes/inventory drift")
    archive = Path(frozen["archive"])
    if not archive.is_file() or hashlib.sha256(archive.read_bytes()).hexdigest() != frozen["archive_sha256"]:
        errors.append("source archive drift")
    if bind:
        expected_runner = root / "benchmarks/experiments/rotate_service_compare.py"
        expected_manifest = root / frozen["manifest_relative_path"]
        if Path(configuration.get("frozen_manifest", "")).resolve() != expected_manifest:
            errors.append("manifest path is not bound to frozen extraction")
        if Path(sys.argv[0]).resolve() != expected_runner:
            errors.append("runner must execute the frozen script path")
        # Every service argv points into this extraction; its explicit PYTHONPATH
        # binds deepinfer imports independently of an editable live installation.
        for engine in configuration["engines"]:
            command = engine["command"]
            if len(command) < 2 or Path(command[1]).parent.resolve() != expected_runner.parent:
                errors.append("service command is not bound to frozen scripts")
            if command[0] not in configuration.get("interpreter_environments", {}):
                errors.append("service interpreter has no captured environment")
            if engine.get("environment", {}).get("PYTHONPATH") != str(root / "src"):
                errors.append("service PYTHONPATH is not frozen")
            if engine.get("cwd") != str(root):
                errors.append("service cwd is not frozen")
        if Path(__file__).resolve().parent != expected_runner.parent:
            errors.append("source checker helper was imported from outside frozen scripts")
        for name in ("http_service_compare", "public_service_report", "service_inventory", "service_resource_selection"):
            module = sys.modules.get(name)
            if module is not None and Path(module.__file__).resolve() != expected_runner.parent / f"{name}.py":
                errors.append(f"{name} helper import is not frozen")
        sys.path.insert(0, str(root / "src"))
        from deepinfer_mac import provenance

        if Path(provenance.__file__).resolve() != root / "src/deepinfer_mac/provenance.py":
            errors.append("runtime import is not frozen")
    environments = {}
    for python, expected in configuration.get("interpreter_environments", {}).items():
        try:
            observed = environment(python, root / "benchmarks/experiments/service_source_freeze.py")
            environments[python] = {"sha256": observed["sha256"], "expected_sha256": expected["sha256"]}
            if observed != expected:
                errors.append("installed interpreter environment bytes drift")
        except (OSError, ValueError, subprocess.SubprocessError) as error:
            errors.append(f"environment audit failed: {type(error).__name__}")
    if not environments:
        errors.append("missing installed environment audit")
    return {"verified": not errors, "errors": errors, "source_sha256": digest(actual),
            "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest() if archive.is_file() else None,
            "execution_binding_verified": bind and not errors, "environments": environments}


if __name__ == "__main__":
    if sys.argv[1:] != ["--environment"]:
        raise SystemExit("Only --environment is supported")
    print(json.dumps(environment_current(), sort_keys=True))
