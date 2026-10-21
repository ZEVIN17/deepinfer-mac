"""Verify distributable resources and exclusion of private/runtime artifacts.

Use --dist for a temporary candidate build; published artifacts stay immutable.
"""
import argparse
import json
import re
import tarfile
import zipfile
from pathlib import Path

# Match an actual home component; prose naming the generic /Users/ root is not private.
PRIVATE_HOME = re.compile(b'/' + b'Users/' + rb'[^\s/,\"\'\\<>]+')

root = Path(__file__).resolve().parent.parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--dist', type=Path, default=root / 'dist')
args = parser.parse_args()
wheels = list(args.dist.glob('*.whl'))
sdists = list(args.dist.glob('*.tar.gz'))
assert len(wheels) == len(sdists) == 1, 'dist must contain exactly one current wheel and sdist'
wheel, sdist = wheels[0], sdists[0]


def check_release_json(name, data):
    """Only IDs/results/config may accompany unclear-license public tasks.

    Local project-authored smoke reports remain distributable. Public-subset
    raw prompts, scoring fixtures and generations belong in results/quality-cache.
    """
    if not name.endswith('.json'):
        return
    try:
        value = json.loads(data)
    except (ValueError, UnicodeError):
        return
    pending = [value]
    while pending:
        value = pending.pop()
        if isinstance(value, dict):
            task_id = value.get('task_id', value.get('id', ''))
            if isinstance(task_id, str) and task_id.startswith(('lcb:', 'ifeval:', 'mmlu:')):
                raw_keys = set(value) & {'messages', 'output', 'prompt', 'scoring', 'public_tests',
                                         'private_tests', 'question_content', 'question_text'}
                assert not raw_keys, f'raw public benchmark content in release: {name} ({task_id})'
            pending.extend(value.values())
        elif isinstance(value, list):
            pending.extend(value)


with zipfile.ZipFile(wheel) as archive:
    files = archive.namelist()
    for suffix in ('deepinfer_mac/cli.py', 'deepinfer_mac/engine.py', 'deepinfer_mac/web/index.html',
                   'deepinfer_mac/web/app.js', 'deepinfer_mac/web/style.css',
                   'deepinfer_mac/benchmark/quick.json', 'deepinfer_mac/models.lock.json'):
        assert suffix in files, f'missing {suffix}'
    for name in files:
        check_release_json(name, archive.read(name))
        if name.endswith(('.py', '.md', '.json', '.toml', '.js', '.html', '.css', '.txt')):
            assert not PRIVATE_HOME.search(archive.read(name)), f'private home path in wheel: {name}'
    for suffix in ('licenses/LICENSE', 'licenses/NOTICE', 'third_party/licenses/MLX-LM.txt'):
        assert any(name.endswith(suffix) for name in files), f'missing license: {suffix}'
with tarfile.open(sdist) as archive:
    sources = archive.getnames()
    for member in archive.getmembers():
        if member.isfile() and member.name.endswith('.json'):
            check_release_json(member.name, archive.extractfile(member).read())
        if member.isfile() and member.name.endswith(('.py', '.md', '.json', '.toml', '.js', '.html', '.css', '.txt')):
            assert not PRIVATE_HOME.search(archive.extractfile(member).read()), f'private home path in sdist: {member.name}'
    for suffix in ('/README.md', '/README.zh-CN.md', '/uv.lock', '/models.lock.json', '/upstream.lock.json',
                   '/site/build.py', '/.github/workflows/pages.yml', '/CONTRIBUTING.md'):
        assert any(name.endswith(suffix) for name in sources), f'missing sdist source {suffix}'
for name in files + sources:
    parts = Path(name).parts
    assert not any(part in ('.venv', '.git', '.env', '__pycache__', '.pytest_cache', 'quality-cache', 'results') for part in parts), name
    assert not name.endswith(('.safetensors', '.gguf', '.pyc', '.pem', '.key', '.log')), name
print(f'PASS wheel ({len(files)} entries) + source distribution ({len(sources)} entries): resources, licenses, clean source')
