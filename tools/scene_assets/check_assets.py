"""Check curated scene dependencies without reading image/mesh bytes by default.
--integrity also hashes original Poly Haven files against saved provider MD5s.
Run from repository root; this command never changes files or downloads assets.
"""
import argparse
import hashlib
import json
import re
from pathlib import Path

SCENES = ('the_kunstkammer', 'the_gregorian', 'the_peacock_lamp', 'the_invisible_window')


def dependencies(root):
    pending = [root / 'scenes' / (name + '.owe') for name in SCENES]
    seen = set()
    while pending:
        p = pending.pop().resolve()
        if p in seen:
            continue
        seen.add(p)
        if not p.is_file():
            continue
        if p.suffix == '.owe':
            pending.extend(p.parent / ref for ref in re.findall(r'"(\.\./[^"\n]+)"', p.read_text()))
        elif p.suffix == '.gltf':
            data = json.loads(p.read_text())
            pending.extend(p.parent / item['uri'] for key in ('buffers', 'images') for item in data.get(key, []) if 'uri' in item and not item['uri'].startswith('data:'))
    return sorted(seen)


def provider_hashes(root):
    result = {}
    def walk(value):
        if isinstance(value, dict):
            if 'url' in value and 'md5' in value:
                result[value['url']] = value['md5']
            for child in value.values(): walk(child)
        elif isinstance(value, list):
            for child in value: walk(child)
    for p in (root / 'docs/assets/polyhaven').glob('*/files.json'):
        walk(json.loads(p.read_text()))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path.cwd())
    parser.add_argument('--integrity', action='store_true')
    args = parser.parse_args()
    root = args.root.resolve()
    paths = dependencies(root)
    missing = [p for p in paths if not p.is_file()]
    for p in missing: print('missing:', p.relative_to(root) if p.is_relative_to(root) else p)
    failures = 0
    if args.integrity:
        checked = 0
        hashes = provider_hashes(root)
        manifest = json.loads((root / 'docs/assets/polyhaven.json').read_text())
        for entry in manifest:
            for record in entry['files']:
                p = root / record['path']; expected = hashes.get(record['url'])
                if not p.is_file() or not expected: continue
                checked += 1
                digest = hashlib.md5()
                with p.open('rb') as f:
                    for chunk in iter(lambda: f.read(1024 * 1024), b''): digest.update(chunk)
                if digest.hexdigest() != expected:
                    print('MD5 mismatch:', record['path']); failures += 1
        print(f'{checked} original files checked against provider MD5s')
        print('Provider MD5 checks cover recorded original Poly Haven files only; scans and derivatives use gallery capture hashes.')
    print(f'{len(paths)} dependencies; {len(missing)} missing; {failures} integrity mismatches')
    return 1 if missing or failures else 0


if __name__ == '__main__':
    raise SystemExit(main())
