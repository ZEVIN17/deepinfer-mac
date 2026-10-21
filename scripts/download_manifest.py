"""Download a pinned public model manifest using resumable SHA-verified ranges."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from download_ranges import download
from huggingface_hub import snapshot_download


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--chunks-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=32, choices=range(1, 65))
    parser.add_argument("--metadata-only", action="store_true")
    args = parser.parse_args()
    model = json.loads(args.manifest.read_text())
    snapshot_download(
        model["repo"],
        revision=model["revision"],
        allow_patterns=["*.json", "*.jinja", "README*", "LICENSE*"],
        max_workers=4,
    )
    if args.metadata_only:
        return
    for item in model["files"]:
        if item["path"].endswith(".safetensors"):
            download(
                model["repo"],
                model["revision"],
                item["path"],
                item["size_bytes"],
                item["sha256"],
                args.chunks_dir / item["path"],
                workers=args.workers,
            )


if __name__ == "__main__":
    main()
