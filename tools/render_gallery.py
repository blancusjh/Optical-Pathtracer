#!/usr/bin/env python3
"""Render gallery candidates into raw archives, then explicitly promote reviewed images."""
import argparse
import datetime
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / 'docs/gallery/manifest.json'
EXTENSIONS = ('.png', '.json', '.pfm', '.stdout.log', '.stderr.log')
# Each region inconsistency stops one path (its light is dropped): a gallery capture may stop at most
# one path in a million, and leak none.
MAX_INCONSISTENCY_RATE = 1e-6


def load_entries(path=MANIFEST):
    entries = json.loads(Path(path).read_text())['entries']
    names = [e['name'] for e in entries]
    if len(set(names)) != len(names) or any(not re.fullmatch(r'[a-z][a-z0-9_]*', n) for n in names):
        raise ValueError('gallery names must be unique safe filenames')
    return {e['name']: e for e in entries}


def select_entries(entries, names, scans=False):
    unknown = set(names) - entries.keys()
    if unknown:
        raise ValueError('unknown gallery entry: ' + ', '.join(sorted(unknown)))
    return [entries[n] for n in names] if names else [e for e in entries.values() if scans or not e['opt_in']]


def arguments(entry, args):
    result = entry['arguments'].copy()
    for option, value in (('--backend', args.backend), ('--device', args.device)):
        if value is not None:
            if option in result:
                result[result.index(option) + 1] = str(value)
            else:
                result += [option, str(value)]
    return result


def identity(entry, options, binary, asset_cache=None):
    from run_benchmarks import asset_identity, code_identity, digest, environment_identity, file_digest
    assets = asset_identity(ROOT / entry['scene'], asset_cache)
    missing = [p for p, h in assets.items() if h == 'missing']
    if missing:
        raise ValueError('missing scene assets: ' + ', '.join(missing))
    return {'schema': 1, 'scene': entry['scene'], 'arguments': options, 'assets': assets,
            'code': code_identity(binary), 'host': environment_identity(binary),
            'gallery_tool_sha256': file_digest(__file__), 'entry_sha256': digest(entry)}


def verified_archive(prefix, entry, *, expected=None):
    from run_benchmarks import asset_identity, digest, file_digest
    data = json.loads(Path(str(prefix) + '.gallery.json').read_text())
    ident = data['identity']
    if data.get('schema') != 1 or data.get('name') != entry['name']:
        raise ValueError('archive has wrong name/schema')
    if ident.get('entry_sha256') != digest(entry):
        raise ValueError('gallery entry changed since rendering; rerender this entry')
    if ident['scene'] != entry['scene'] or ident['assets'] != asset_identity(ROOT / entry['scene']):
        raise ValueError('scene/assets changed since rendering; rerender this entry')
    if expected is not None and ident != expected:
        raise ValueError('render identity changed; reuse refused')
    for ext in EXTENSIONS:
        if file_digest(str(prefix) + ext) != data['artifacts'][ext]:
            raise ValueError('archive artifact changed: ' + ext)
    return data


def validate_native(record, options):
    limitations = []
    for flag, field, convert in (('--spp', 'samples_per_pixel', int), ('--seed', 'seed', int),
                                 ('--passes', 'passes', int), ('--integrator', 'integrator', str),
                                 ('--max-depth', 'max_depth', int)):
        if flag in options and record[field] != convert(options[options.index(flag) + 1]):
            raise ValueError('renderer did not honor ' + flag + '; possible integrator fallback')
    if '--detector' in options and record['detector']['name'] != options[options.index('--detector') + 1]:
        raise ValueError('renderer used a different detector')
    if '--resolution' in options:
        w, h = map(int, options[options.index('--resolution') + 1].split('x'))
        pose = record['observer_pose']
        if (pose['width'], pose['height']) != (w, h):
            raise ValueError('renderer used a different resolution')
    if '--photons' in options:
        requested = float(options[options.index('--photons') + 1])
        photons = record.get('sampling', {}).get('photons')
        if photons is None and record['integrator'] in ('vcm', 'bdpt'):
            limitations.append('Native bidirectional records omit the photon budget; the exact requested --photons value is retained in the command, not independently verified from native metadata.')
        elif photons is None or photons['per_pixel_and_iteration'] != requested:
            raise ValueError('renderer used a different photon budget')
    return limitations


def render_entry(prefix, entry, options, binary, reuse, asset_cache):
    from run_benchmarks import asset_identity, file_digest
    ident = identity(entry, options, binary, asset_cache)
    if reuse:
        try:
            verified_archive(prefix, entry, expected=ident)
            print(entry['name'] + ': verified archive reused', flush=True)
            return
        except (OSError, ValueError, KeyError):
            pass
    if any(Path(str(prefix) + ext).exists() for ext in (*EXTENSIONS, '.gallery.json')):
        raise ValueError('archive already exists; choose a fresh --archive or matching --reuse')
    prefix.parent.mkdir(parents=True, exist_ok=True)
    cmd = [str(binary), 'render', entry['scene'], *options, '--out', str(prefix)]
    print(shlex.join(cmd), flush=True)
    t0 = time.perf_counter()
    with Path(str(prefix) + '.stdout.log').open('w') as out, Path(str(prefix) + '.stderr.log').open('w') as err:
        result = subprocess.run(cmd, cwd=ROOT, stdout=out, stderr=err)
    wall = time.perf_counter() - t0
    if result.returncode:
        raise ValueError(f'render exited {result.returncode}; logs preserved at {prefix}')
    record = json.loads(Path(str(prefix) + '.json').read_text())
    limitations = validate_native(record, options)
    if asset_identity(ROOT / entry['scene']) != ident['assets'] or file_digest(binary) != ident['code']['binary_sha256']:
        raise ValueError('scene/assets/binary changed during rendering; archive cannot be certified')
    data = {'schema': 1, 'name': entry['name'], 'identity': ident, 'command': cmd,
            'validation_limitations': limitations,
            'wall_seconds': wall, 'render_seconds': record['render_seconds'], 'review': None,
            'artifacts': {ext: file_digest(str(prefix) + ext) for ext in EXTENSIONS}}
    Path(str(prefix) + '.gallery.json').write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    print(f"{entry['name']}: staged; {record['render_seconds']:.3f}s transport, {wall:.3f}s wall", flush=True)


def promotion_data(prefix, entry, review_note):
    if not review_note.strip():
        raise ValueError('promotion requires --review-note describing image/physics review')
    if entry.get('rights') != 'cleared':
        raise ValueError('promotion requires cleared source rights; retain this image in its local archive')
    data = verified_archive(prefix, entry)
    record = json.loads(Path(str(prefix) + '.json').read_text())
    stats = record.get('statistics', {})
    if any(type(stats.get(field)) is not int for field in ('paths', 'region_inconsistencies', 'leaks')):
        raise ValueError('promotion requires recorded path, inconsistency and leak counts; retain this capture locally')
    if stats['leaks'] != 0:
        raise ValueError('promotion requires recorded zero leaks; retain this diagnostic capture locally')
    if stats['region_inconsistencies'] > MAX_INCONSISTENCY_RATE * stats['paths']:
        raise ValueError(f'promotion requires at most {MAX_INCONSISTENCY_RATE:g} region inconsistencies per path; '
                         'retain this diagnostic capture locally')
    return data


def promote_entry(prefix, entry, destination, review_note):
    data = promotion_data(prefix, entry, review_note)
    data['source_rights'] = entry['rights']
    data['review'] = {'note': review_note, 'utc': datetime.datetime.now(datetime.timezone.utc).isoformat()}
    destination.mkdir(parents=True, exist_ok=True)
    for ext in ('.png', '.json'):
        shutil.copy2(str(prefix) + ext, destination / (entry['name'] + ext))
    (destination / (entry['name'] + '.gallery.json')).write_text(json.dumps(data, indent=2) + '\n')
    print(entry['name'] + ': promoted; raw PFM retained at ' + str(prefix), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('names', nargs='*', help='explicit entries; default is the legacy gallery')
    parser.add_argument('--scans', action='store_true', help='include opt-in scan views')
    parser.add_argument('--manifest', type=Path, default=MANIFEST)
    parser.add_argument('--binary', type=Path, default=Path(os.environ.get('OWE', str(ROOT / 'build/owe'))))
    parser.add_argument('--archive', type=Path, help='raw staging directory')
    parser.add_argument('--destination', type=Path, default=ROOT / 'docs/gallery')
    parser.add_argument('--backend', choices=('cpu', 'gpu'))
    parser.add_argument('--device', type=int)
    parser.add_argument('--reuse', action='store_true')
    parser.add_argument('--review-note', default='')
    group = parser.add_mutually_exclusive_group()
    for flag in ('list', 'dry-run', 'check', 'promote'):
        group.add_argument('--' + flag, action='store_true')
    args = parser.parse_args(argv)
    try:
        selected = select_entries(load_entries(args.manifest), args.names, args.scans)
        if args.list:
            for e in selected:
                print(f"{e['name']} scene={e['scene']} opt_in={e['opt_in']} rights={e['rights']} {shlex.join(e['arguments'])}")
            return 0
        if args.promote and not args.names:
            raise ValueError('promotion requires an explicit reviewed set of entry names')
        if (args.check or args.promote or args.reuse) and args.archive is None:
            raise ValueError('--archive is required for check, promotion, or reuse')
        stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
        archive = (args.archive or ROOT / 'out/gallery' / stamp).resolve()
        binary = args.binary.resolve()
        if args.promote:
            for e in selected:
                promotion_data(archive / e['name'], e, args.review_note)
        cache = {}
        for e in selected:
            prefix = archive / e['name']
            opts = arguments(e, args)
            if args.dry_run:
                print(shlex.join([str(binary), 'render', e['scene'], *opts, '--out', str(prefix)]))
            elif args.check:
                verified_archive(prefix, e)
                print(e['name'] + ': archive verified against current scene/assets')
            elif args.promote:
                promote_entry(prefix, e, args.destination.resolve(), args.review_note)
            else:
                render_entry(prefix, e, opts, binary, args.reuse, cache)
        return 0
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))


if __name__ == '__main__':
    sys.exit(main())
