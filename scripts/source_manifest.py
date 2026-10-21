"""Create a public source identity before an initial Git commit exists."""
from __future__ import annotations

import argparse
import io
import json
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))


def main():
    from deepinfer_mac.provenance import source_manifest

    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path('docs/evidence/source-manifest.json'))
    parser.add_argument('--archive', type=Path, help='Also save the hashed public source files and required license notices')
    args = parser.parse_args()
    result = source_manifest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    if args.archive:
        args.archive.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(args.archive, 'w:gz') as archive:
            names = [*result['files'], 'LICENSE', 'NOTICE',
                     *(path.relative_to(ROOT).as_posix() for path in (ROOT / 'third_party/licenses').glob('*'))]
            for name in names:
                data = (ROOT / name).read_bytes()
                member = tarfile.TarInfo(name)
                member.size = len(data)
                member.mode = 0o644
                archive.addfile(member, io.BytesIO(data))
    print(result['source_tree_sha256'])


if __name__ == '__main__':
    main()
