"""Verify distributable resources and exclusion of private/runtime artifacts."""
import tarfile
import zipfile
from pathlib import Path

root = Path(__file__).resolve().parent.parent
wheels = list((root / 'dist').glob('*.whl'))
sdists = list((root / 'dist').glob('*.tar.gz'))
assert len(wheels) == len(sdists) == 1, 'dist must contain exactly one current wheel and sdist'
wheel, sdist = wheels[0], sdists[0]
with zipfile.ZipFile(wheel) as archive:
    files = archive.namelist()
    for suffix in ('deepinfer_mac/cli.py', 'deepinfer_mac/engine.py', 'deepinfer_mac/web/index.html',
                   'deepinfer_mac/web/app.js', 'deepinfer_mac/web/style.css',
                   'deepinfer_mac/benchmark/quick.json', 'deepinfer_mac/models.lock.json'):
        assert suffix in files, f'missing {suffix}'
    for name in files:
        if name.endswith(('.py', '.md', '.json', '.toml', '.js', '.html', '.css', '.txt')):
            assert (b'/' + b'Users' + b'/') not in archive.read(name), f'private home path in wheel: {name}'
    for suffix in ('licenses/LICENSE', 'licenses/NOTICE', 'third_party/licenses/MLX-LM.txt'):
        assert any(name.endswith(suffix) for name in files), f'missing license: {suffix}'
with tarfile.open(sdist) as archive:
    sources = archive.getnames()
    for member in archive.getmembers():
        if member.isfile() and member.name.endswith(('.py', '.md', '.json', '.toml', '.js', '.html', '.css', '.txt')):
            assert (b'/' + b'Users' + b'/') not in archive.extractfile(member).read(), f'private home path in sdist: {member.name}'
    for suffix in ('/README.md', '/README.zh-CN.md', '/uv.lock', '/models.lock.json', '/upstream.lock.json',
                   '/site/build.py', '/.github/workflows/pages.yml', '/CONTRIBUTING.md'):
        assert any(name.endswith(suffix) for name in sources), f'missing sdist source {suffix}'
for name in files + sources:
    parts = Path(name).parts
    assert not any(part in ('.venv', '.git', '.env', '__pycache__', '.pytest_cache') for part in parts), name
    assert not name.endswith(('.safetensors', '.pyc', '.pem', '.key', '.log')), name
print(f'PASS wheel ({len(files)} entries) + source distribution ({len(sources)} entries): resources, licenses, clean source')
