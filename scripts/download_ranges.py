"""Resumable public Hugging Face HTTP ranges with strict range/SHA verification.

Use when snapshot_download's single-stream transfer is too slow. Immutable pinned
snapshots only. Downloads one weight file at a time; no authentication is needed
for the project's public models. Existing HF metadata is downloaded separately.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import os
import re
import time
from pathlib import Path

import httpx


def download(repo, revision, filename, size, sha256, chunks_dir, workers=16, chunk_bytes=4 * 1024**2):
    if not re.fullmatch(r"[0-9a-f]{40}", revision) or not re.fullmatch(r"[0-9a-f]{64}", sha256):
        raise ValueError("Use a full immutable revision and SHA256")
    hub = Path(os.environ.get("HF_HOME", str(Path.home() / ".cache/huggingface"))) / "hub"
    cache = hub / ("models--" + repo.replace("/", "--"))
    target = cache / "blobs" / sha256
    target.parent.mkdir(parents=True, exist_ok=True)
    chunks = Path(chunks_dir)
    chunks.mkdir(parents=True, exist_ok=True)
    if (
        target.exists()
        and target.stat().st_size == size
        and hashlib.file_digest(target.open("rb"), "sha256").hexdigest() == sha256
    ):
        print("verified existing", filename, flush=True)
        return target
    url = f"https://huggingface.co/{repo}/resolve/{revision}/{filename}"
    timeout = httpx.Timeout(120, read=45)
    with httpx.Client(
        follow_redirects=True, timeout=timeout, limits=httpx.Limits(max_connections=workers)
    ) as client:

        def fetch(start):
            end = min(size - 1, start + chunk_bytes - 1)
            chunk = chunks / str(start)
            temp = chunks / (str(start) + ".tmp")
            if chunk.exists() and chunk.stat().st_size == end - start + 1:
                return start
            for attempt in range(16):
                existing = temp.stat().st_size if temp.exists() else 0
                if existing > end - start + 1:
                    raise ValueError(f"Oversized partial chunk {temp}; inspect before removing")
                if existing == end - start + 1:
                    temp.replace(chunk)
                    return start
                offset = start + existing
                try:
                    with client.stream(
                        "GET", url, headers={"Range": f"bytes={offset}-{end}", "Cache-Control": "no-cache"}
                    ) as response:
                        response.raise_for_status()
                        expected = f"bytes {offset}-{end}/{size}"
                        if response.status_code != 206 or response.headers.get("content-range") != expected:
                            raise ValueError(
                                f"Range mismatch {response.status_code}: {response.headers.get('content-range')}, expected {expected}"
                            )
                        with temp.open("ab") as output:
                            for block in response.iter_bytes(64 * 1024):
                                output.write(block)
                    if temp.stat().st_size == end - start + 1:
                        temp.replace(chunk)
                        return start
                except (httpx.HTTPError, OSError) as exc:
                    print(
                        f"retry chunk={start} attempt={attempt + 1} {type(exc).__name__}: {exc}", flush=True
                    )
                    time.sleep(min(attempt + 1, 10))
            raise RuntimeError(f"Failed resumable chunk {start}")

        starts = list(range(0, size, chunk_bytes))
        begin = time.monotonic()
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(fetch, start) for start in starts]
            for done, future in enumerate(concurrent.futures.as_completed(futures), 1):
                future.result()
                if done % 8 == 0 or done == len(starts):
                    print(
                        f"{filename} {done}/{len(starts)} elapsed={time.monotonic() - begin:.1f}s", flush=True
                    )
    digest = hashlib.sha256()
    assembled = target.with_suffix(".assembled")
    with assembled.open("wb") as output:
        for start in starts:
            data = (chunks / str(start)).read_bytes()
            digest.update(data)
            output.write(data)
    if digest.hexdigest() != sha256:
        raise ValueError(f"Final SHA256 mismatch: {digest.hexdigest()} expected {sha256}; chunks retained")
    assembled.replace(target)
    snapshot = cache / "snapshots" / revision / filename
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    if not snapshot.exists():
        snapshot.symlink_to(os.path.relpath(target, snapshot.parent))
    print(f"verified {filename} {sha256}", flush=True)
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("repo", "revision", "filename", "sha256", "chunks-dir"):
        parser.add_argument("--" + field, required=True)
    parser.add_argument("--size", type=int, required=True)
    parser.add_argument("--workers", type=int, default=16, choices=range(1, 65))
    args = parser.parse_args()
    download(args.repo, args.revision, args.filename, args.size, args.sha256, args.chunks_dir, args.workers)


if __name__ == "__main__":
    main()
