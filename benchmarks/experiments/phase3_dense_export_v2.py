"""Export v2 development and prospectively selected internal configuration."""
import argparse
import hashlib
import json
from pathlib import Path

from phase3_dense_export import digest, export


def export_v2(raw):
    public = export(raw)
    for field in ('resident_pair_load_ms', 'warmup_ms', 'input_token_counts'):
        if field in raw:
            public[field] = raw[field]
    public['failures'] = [{'type': str(f.get('type', 'unknown')),
                           'detail_sha256': hashlib.sha256(str(f.get('detail', '')).encode()).hexdigest()}
                          for f in raw.get('failures', [])]
    public['numerical_gate'] = raw['numerical_gate']
    public['output_and_boundary_checks_pass'] = raw['output_and_boundary_checks_pass']
    native = raw['native_batch']
    public['native_batch'] = {k: v for k, v in native.items() if k != 'outputs'}
    public['native_batch']['outputs'] = [{'count': len(out), 'sha256': digest(out)} for out in native['outputs']]
    return public


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    raw = json.loads(args.input.read_text())
    public = export_v2(raw)
    public['raw_report_sha256'] = hashlib.sha256(args.input.read_bytes()).hexdigest()
    args.output.write_text(json.dumps(public, indent=2) + '\n')
