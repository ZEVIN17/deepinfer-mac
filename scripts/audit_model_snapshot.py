"""Audit pinned local safetensors headers and file SHA without loading model arrays."""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--snapshot', type=Path, required=True)
    ap.add_argument('--api', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    api = json.loads(args.api.read_text())
    index_file = args.snapshot/'model.safetensors.index.json'
    if index_file.exists():
        index = json.loads(index_file.read_text())
    else:
        with (args.snapshot/'model.safetensors').open('rb') as handle:
            length = struct.unpack('<Q', handle.read(8))[0]
            single_header = json.loads(handle.read(length))
        index = {'weight_map': {key: 'model.safetensors' for key in single_header if key != '__metadata__'}}
    config = json.loads((args.snapshot/'config.json').read_text())
    files, keys, tensor_bytes = [], set(), {'text': 0, 'vision': 0, 'mtp': 0, 'other': 0}
    for name in sorted(set(index['weight_map'].values())):
        path = args.snapshot/name
        with path.open('rb') as handle:
            header_size = struct.unpack('<Q', handle.read(8))[0]
            header = json.loads(handle.read(header_size))
            handle.seek(0)
            digest = hashlib.file_digest(handle, 'sha256').hexdigest()
        expected = next(x for x in api['siblings'] if x['rfilename'] == name)
        assert path.stat().st_size == expected['size']
        assert digest == expected['lfs']['sha256']
        category_counts = {key: 0 for key in tensor_bytes}
        category_bytes = dict(category_counts)
        for key, info in header.items():
            if key == '__metadata__':
                continue
            assert key not in keys and index['weight_map'][key] == name
            keys.add(key)
            category = 'mtp' if 'mtp' in key else 'vision' if any(v in key for v in ('vision_tower','visual')) else 'text' if key.startswith(('language_model.', 'model.', 'lm_head.')) else 'other'
            size = info['data_offsets'][1]-info['data_offsets'][0]
            tensor_bytes[category] += size
            category_counts[category] += 1
            category_bytes[category] += size
        files.append({'path': name, 'size_bytes': path.stat().st_size, 'sha256': digest,
                      'tensor_count_by_category': category_counts, 'tensor_bytes_by_category': category_bytes})
    assert keys == set(index['weight_map'])
    out = {'repo': api['id'], 'revision': api['sha'], 'quantization': config.get('quantization'),
           'files': files, 'index_present': index_file.exists(), 'tensor_count': len(keys), 'tensor_bytes_by_category': tensor_bytes,
           'weight_files_sha256_verified': True, 'weights_loaded': False,
           'mtp_keys': sorted(k for k in keys if 'mtp' in k),
           'metadata_sha256': {p: hashlib.sha256((args.snapshot/p).read_bytes()).hexdigest() for p in ('config.json','tokenizer.json','tokenizer_config.json','chat_template.jinja') if (args.snapshot/p).exists()}}
    args.output.write_text(json.dumps(out, indent=2)+'\n')
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
