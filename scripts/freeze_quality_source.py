"""Freeze a reviewable quality runtime; never include private/unclear-license data."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import tarfile
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--extract', type=Path, required=True)
    args = ap.parse_args()
    roots = [Path('src/deepinfer_mac'), Path('benchmarks/quality'), Path('third_party/licenses')]
    paths = [p for root in roots for p in root.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc']
    paths += [Path(p) for p in ('LICENSE','NOTICE','pyproject.toml','uv.lock','models.lock.json','upstream.lock.json',
                              'benchmarks/experiments/phase2_model_quality.py','benchmarks/prompts/phase2-quality-smoke.json',
                              'docs/PHASE2_QUALITY_PROTOCOL.md')]
    contents = {str(p): p.read_bytes() for p in sorted(paths)}
    for name, data in contents.items():
        if Path(name).read_bytes() != data:
            raise RuntimeError(f'Source changed during freeze: {name}; retry after coordinated edits')
    hashes = {name: hashlib.sha256(data).hexdigest() for name, data in contents.items()}
    manifest = {'files': hashes, 'tree_sha256': hashlib.sha256(json.dumps(hashes,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
                'scope': 'runtime + locks + quality runner/scorers + local smoke; public third-party data materialized separately'}
    contents['quality-source-manifest.json'] = (json.dumps(manifest,indent=2)+'\n').encode()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('wb') as raw, gzip.GzipFile(fileobj=raw,mode='wb',mtime=0) as gz, tarfile.open(fileobj=gz,mode='w') as archive:
        for name, data in contents.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            info.mode = 0o644
            archive.addfile(info,io.BytesIO(data))
    for name, data in contents.items():
        dest = args.extract/name
        dest.parent.mkdir(parents=True,exist_ok=True)
        dest.write_bytes(data)
    print(json.dumps({'archive':str(args.output),'archive_sha256':hashlib.sha256(args.output.read_bytes()).hexdigest(),
                      'tree_sha256':manifest['tree_sha256'],'files':len(hashes),'extract':str(args.extract)},indent=2))


if __name__ == '__main__':
    main()
