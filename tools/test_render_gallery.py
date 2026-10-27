#!/usr/bin/env python3
"""Focused archive integrity and publication checks, with no renderer/GPU invocation."""
import argparse
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import render_gallery as gallery
import run_benchmarks as benchmark


class GalleryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.entry = {'name': 'fixture', 'scene': 'scene.owe', 'arguments': ['--spp', '4'],
                      'rights': 'cleared', 'opt_in': False}
        (self.root / 'scene.owe').write_text('material White { reflectance = 0.8 }\n')
        self.prefix = self.root / 'archive/fixture'
        self.prefix.parent.mkdir()
        for ext in gallery.EXTENSIONS:
            Path(str(self.prefix) + ext).write_text('fixture' + ext)
        Path(str(self.prefix) + '.json').write_text(json.dumps({
            'statistics': {'paths': 10**7, 'region_inconsistencies': 0, 'leaks': 0}}))
        with mock.patch.object(benchmark, 'ROOT', self.root):
            assets = benchmark.asset_identity(self.root / 'scene.owe')
        self.data = {'schema': 1, 'name': 'fixture', 'identity': {'scene': 'scene.owe', 'assets': assets,
                     'entry_sha256': benchmark.digest(self.entry)},
                     'artifacts': {ext: benchmark.file_digest(str(self.prefix) + ext) for ext in gallery.EXTENSIONS}}
        Path(str(self.prefix) + '.gallery.json').write_text(json.dumps(self.data))
        for module in (gallery, benchmark):
            patcher = mock.patch.object(module, 'ROOT', self.root)
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_changed_scene_and_changed_pfm_are_rejected(self):
        gallery.verified_archive(self.prefix, self.entry)
        (self.root / 'scene.owe').write_text('changed physical material')
        with self.assertRaisesRegex(ValueError, 'scene/assets changed'):
            gallery.verified_archive(self.prefix, self.entry)
        (self.root / 'scene.owe').write_text('material White { reflectance = 0.8 }\n')
        Path(str(self.prefix) + '.pfm').write_text('replaced image')
        with self.assertRaisesRegex(ValueError, 'artifact changed'):
            gallery.verified_archive(self.prefix, self.entry)

    def test_reuse_rejects_identity_change(self):
        expected = {**self.data['identity'], 'arguments': ['--spp', '8']}
        with self.assertRaisesRegex(ValueError, 'reuse refused'):
            gallery.verified_archive(self.prefix, self.entry, expected=expected)

    def test_promotion_requires_review_and_cleared_rights(self):
        destination = self.root / 'public'
        with self.assertRaisesRegex(ValueError, 'review-note'):
            gallery.promote_entry(self.prefix, self.entry, destination, '')
        self.assertFalse(destination.exists())
        self.entry['rights'] = 'unresolved'
        self.data['identity']['entry_sha256'] = benchmark.digest(self.entry)
        Path(str(self.prefix) + '.gallery.json').write_text(json.dumps(self.data))
        with self.assertRaisesRegex(ValueError, 'cleared source rights'):
            gallery.promote_entry(self.prefix, self.entry, destination, 'reviewed')
        self.assertFalse(destination.exists())
        self.entry['rights'] = 'cleared'
        self.data['identity']['entry_sha256'] = benchmark.digest(self.entry)
        Path(str(self.prefix) + '.gallery.json').write_text(json.dumps(self.data))
        gallery.promote_entry(self.prefix, self.entry, destination, 'contacts reviewed')
        self.assertTrue(Path(str(self.prefix) + '.pfm').exists())
        self.assertFalse((destination / 'fixture.pfm').exists())
        self.assertEqual((destination / 'fixture.png').read_bytes(), Path(str(self.prefix) + '.png').read_bytes())

    def test_promotion_rejects_bad_or_missing_transport_diagnostics(self):
        destination = self.root / 'public'
        for stats in ({'paths': 10**7, 'region_inconsistencies': 11, 'leaks': 0},
                      {'paths': 10**7, 'region_inconsistencies': 0, 'leaks': 1}, {},
                      {'region_inconsistencies': 0, 'leaks': 0},
                      {'paths': 10**7, 'region_inconsistencies': False, 'leaks': 0}):
            with self.subTest(stats=stats):
                self.record(stats)
                with self.assertRaisesRegex(ValueError, 'promotion requires'):
                    gallery.promote_entry(self.prefix, self.entry, destination, 'reviewed')
                self.assertFalse(destination.exists())

    def test_promotion_accepts_one_inconsistency_per_million_paths(self):
        self.record({'paths': 10**7, 'region_inconsistencies': 10, 'leaks': 0})
        gallery.promote_entry(self.prefix, self.entry, self.root / 'public', 'reviewed')
        self.assertTrue((self.root / 'public/fixture.png').exists())

    def record(self, stats):
        Path(str(self.prefix) + '.json').write_text(json.dumps({'statistics': stats}))
        self.data['artifacts']['.json'] = benchmark.file_digest(str(self.prefix) + '.json')
        Path(str(self.prefix) + '.gallery.json').write_text(json.dumps(self.data))

    def test_batch_preflight_rejects_all_before_copying(self):
        manifest = self.root / 'manifest.json'
        second = {**self.entry, 'name': 'blocked', 'rights': 'unresolved'}
        manifest.write_text(json.dumps({'entries': [self.entry, second]}))
        destination = self.root / 'public'
        with mock.patch.object(gallery, 'verified_archive', return_value=self.data), \
                mock.patch.object(gallery, 'promote_entry') as promote, \
                contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                gallery.main(['fixture', 'blocked', '--manifest', str(manifest),
                              '--archive', str(self.prefix.parent), '--destination', str(destination),
                              '--promote', '--review-note', 'reviewed'])
            promote.assert_not_called()
        self.assertFalse(destination.exists())

    def test_detector_change_rejects_old_archive_for_check_and_promotion(self):
        self.entry['arguments'] += ['--detector', 'Room']
        self.data['identity']['entry_sha256'] = benchmark.digest(self.entry)
        Path(str(self.prefix) + '.gallery.json').write_text(json.dumps(self.data))
        gallery.verified_archive(self.prefix, self.entry)
        changed = {**self.entry, 'arguments': ['--spp', '4', '--detector', 'Shade']}
        with self.assertRaisesRegex(ValueError, 'gallery entry changed'):
            gallery.verified_archive(self.prefix, changed)
        destination = self.root / 'public'
        with self.assertRaisesRegex(ValueError, 'gallery entry changed'):
            gallery.promote_entry(self.prefix, changed, destination, 'reviewed Shade')
        self.assertFalse(destination.exists())

    def test_unknown_and_unsafe_names_fail_before_render(self):
        with self.assertRaisesRegex(ValueError, 'unknown gallery'):
            gallery.select_entries({'fixture': self.entry}, ['misspelled'])
        manifest = self.root / 'manifest.json'
        manifest.write_text(json.dumps({'entries': [{**self.entry, 'name': '../outside'}]}))
        with self.assertRaisesRegex(ValueError, 'safe filenames'):
            gallery.load_entries(manifest)
        manifest.write_text(json.dumps({'entries': [self.entry]}))
        with mock.patch.object(gallery, 'render_entry') as render, contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                gallery.main(['unknown', '--manifest', str(manifest)])
            render.assert_not_called()

    def test_manifest_scan_selection_and_renderer_fallback(self):
        entries = {'fixture': self.entry, 'scan': {**self.entry, 'name': 'scan', 'opt_in': True}}
        self.assertEqual(len(gallery.select_entries(entries, [])), 1)
        self.assertEqual(len(gallery.select_entries(entries, [], scans=True)), 2)
        self.assertEqual(gallery.select_entries(entries, ['scan'])[0]['name'], 'scan')
        with self.assertRaisesRegex(ValueError, 'fallback'):
            gallery.validate_native({'integrator': 'path'}, ['--integrator', 'vcm'])

    def test_bidirectional_photon_metadata_omission_is_explicit(self):
        limitations = gallery.validate_native({'integrator': 'vcm', 'sampling': {'bidirectional': {}}},
                                              ['--integrator', 'vcm', '--photons', '1'])
        self.assertEqual(len(limitations), 1)
        self.assertIn('not independently verified', limitations[0])
        with self.assertRaisesRegex(ValueError, 'photon budget'):
            gallery.validate_native({'integrator': 'sppm', 'sampling': {}}, ['--photons', '16'])
        with self.assertRaisesRegex(ValueError, 'photon budget'):
            gallery.validate_native({'integrator': 'sppm', 'sampling': {'photons': {'per_pixel_and_iteration': 1}}},
                                    ['--photons', '16'])


if __name__ == '__main__':
    unittest.main()
