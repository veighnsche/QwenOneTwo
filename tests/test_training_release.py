"""Model-free release ingestion regressions using explicit schema-only fixtures."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from qwen_one_two.training.inputs import canonical, digest, validate_record_manifest
from qwen_one_two.training.release import load_release, decode
from release_fixtures import row, pilot_rows, write_release, approve, replace_record


class ReleaseTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_bound_bytes_and_immutable_snapshot(self):
        source = b'UTF8 schema fixture: \xce\xb1'
        r = row(sources=[dict(id='s', version=1, path='sources/s.txt', sha256=digest(source))])
        write_release(self.root, [r], assets={'sources/s.txt':source})
        release = load_release(self.root)
        self.assertEqual(release.summary['selected_records'], 1)
        self.assertFalse(release.summary['semantic_authentication'])
        release.records['fixture']['human_input']['prompt'] = 'mutation'
        self.assertIn('SCHEMA', release.records['fixture']['human_input']['prompt'])
        (self.root/'sources/s.txt').write_bytes(b'changed')
        self.assertEqual(release.asset('sources/s.txt'), source)
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            load_release(self.root)

    def test_pinned_manifest_and_current_review_binding(self):
        manifest = write_release(self.root)
        with self.assertRaisesRegex(ValueError, 'manifest hash'):
            load_release(self.root, expected_hash='0'*64)
        replace_record(self.root, manifest, 'fixture', lambda r:r['labels'].update(status='changed after review'))
        with self.assertRaisesRegex(ValueError, 'current annotations'):
            load_release(self.root)

    def test_missing_or_changed_review_and_release_approval(self):
        manifest = write_release(self.root)
        (self.root/'reviews/fixture-0.json').unlink()
        with self.assertRaisesRegex(ValueError, 'missing release artifact'):
            load_release(self.root)
        manifest = write_release(self.root)
        manifest['version'] = 2
        (self.root/'manifest.json').write_bytes(canonical(manifest))
        with self.assertRaisesRegex(ValueError, 'record/release binding'):
            load_release(self.root)
        manifest = write_release(self.root)
        manifest['extra_note'] = 'changed after release approval'
        (self.root/'manifest.json').write_bytes(canonical(manifest))
        with self.assertRaisesRegex(ValueError, 'approval binding'):
            load_release(self.root)

    def test_self_review_model_review_and_unbound_judgments_reject(self):
        for change in (lambda e:e.update(reviewer='TEST-ONLY-author'),
                       lambda e:e.update(kind='model'),
                       lambda e:e.update(judgments={}),
                       lambda e:e['record'].update(annotation_sha256='0'*64)):
            manifest = write_release(self.root, mutate_review=change)
            with self.assertRaises(ValueError):
                load_release(self.root)

    def test_locked_single_review_rejected(self):
        r = row(split='locked')
        write_release(self.root, [r], purpose='locked', review_counts={'fixture':1})
        with self.assertRaisesRegex(ValueError, 'independent reviews'):
            load_release(self.root)
        r['reviews'] = [dict(reviewer='one', kind='human', semantics_blind=True,
              outcome_independent=True, decision='approved', artifact_hash='a'*64, record_version=1)]
        with self.assertRaisesRegex(ValueError, 'independent reviews'):
            validate_record_manifest(r, {}, for_release=True)

    def test_pilot_exact_independent_families_and_domain_counts(self):
        rows = pilot_rows()
        write_release(self.root, rows, purpose='pilot')
        release = load_release(self.root)
        self.assertEqual(release.summary['selected_records'], 120)
        self.assertEqual(set(release.summary['selected_domains'].values()), {20})
        write_release(self.root, rows[:-1], purpose='pilot')
        with self.assertRaisesRegex(ValueError, '120 independent'):
            load_release(self.root)
        rows[0]['domain'] = rows[-1]['domain']
        write_release(self.root, rows, purpose='pilot')
        with self.assertRaisesRegex(ValueError, 'domain allocation'):
            load_release(self.root)
        rows = pilot_rows(); rows[0]['human_input'] = rows[1]['human_input']
        write_release(self.root, rows, purpose='pilot')
        with self.assertRaisesRegex(ValueError, 'identical human inputs'):
            load_release(self.root)

    def test_design_exposed_or_retired_locked_material_rejected(self):
        for flag in ('design_exposed', 'test_driven_tuning'):
            write_release(self.root, [row(split='locked', **{flag:True})], purpose='locked')
            with self.assertRaisesRegex(ValueError, 'exposed or retired'):
                load_release(self.root)

    def test_lineage_cycles_missing_ancestors_and_cross_split_reject(self):
        a = row('a', parents=['b']); b = row('b', family=a['family'], parents=['a'])
        write_release(self.root, [a,b])
        with self.assertRaisesRegex(ValueError, 'cyclic'):
            load_release(self.root)
        write_release(self.root, [a])
        with self.assertRaisesRegex(ValueError, 'missing ancestor'):
            load_release(self.root)
        b.update(parents=[], split='locked')
        write_release(self.root, [a,b], selected={'a'})
        with self.assertRaisesRegex(ValueError, 'split'):
            load_release(self.root)

    def test_excluded_ancestor_and_quarantined_batch_block_descendants(self):
        a = row('a'); b = row('b', family=a['family'], parents=['a'])
        write_release(self.root, [a,b], selected={'b'}, exclusions={'records':['a'], 'families':[], 'batches':[]})
        with self.assertRaisesRegex(ValueError, 'quarantined lineage'):
            load_release(self.root)
        a.update(quarantined=True, batch='batch'); b.update(parents=[], batch='batch')
        write_release(self.root, [a,b], selected={'b'},
                      batches={'batch':{'members':['a','b'], 'second_review':['a','b']}})
        with self.assertRaisesRegex(ValueError, 'quarantined lineage'):
            load_release(self.root)

    def test_source_identity_collision_and_duplicate_cross_split_tasks(self):
        first = b'first'; second = b'second'
        a = row('a', sources=[dict(id='same', version=1, path='a.txt', sha256=digest(first))])
        b = row('b', sources=[dict(id='same', version=1, path='b.txt', sha256=digest(second))])
        write_release(self.root, [a,b], assets={'a.txt':first, 'b.txt':second})
        with self.assertRaisesRegex(ValueError, 'identity/version collision'):
            load_release(self.root)
        a = row('a'); b = row('b', split='locked', human_input=a['human_input'])
        write_release(self.root, [a,b], selected={'a'})
        with self.assertRaisesRegex(ValueError, 'duplicate human input'):
            load_release(self.root)

    def test_second_review_audit_counts_and_strata(self):
        rows = [row(str(i), batch='b') for i in range(21)]
        counts = {str(i):2 if i < 20 else 1 for i in range(21)}
        batch = {'b':{'members':[str(i) for i in range(21)], 'second_review':[str(i) for i in range(20)]}}
        write_release(self.root, rows, batches=batch, review_counts=counts)
        load_release(self.root)
        batch['b']['second_review'] = ['0']
        write_release(self.root, rows, batches=batch, review_counts=counts)
        with self.assertRaisesRegex(ValueError, 'insufficient second-review audit'):
            load_release(self.root)
        batch['b']['second_review'] = [str(i) for i in range(20)]
        rows[-1]['domain'] = 'new-schema-stratum'
        write_release(self.root, rows, batches=batch, review_counts=counts)
        with self.assertRaisesRegex(ValueError, 'uncovered batch audit stratum'):
            load_release(self.root)

    def test_json_ambiguity_and_unsafe_paths(self):
        for data in (b'{"id":1,"id":2}', b'{"n":NaN}', b'{"n":1e999}'):
            with self.assertRaises(ValueError):
                decode(data)
        manifest = write_release(self.root)
        for name in ('../outside', '/tmp/absolute', 'records/../fixture.json'):
            manifest['records'][0]['path'] = name; approve(self.root, manifest)
            with self.assertRaisesRegex(ValueError, 'unsafe artifact path'):
                load_release(self.root)
        manifest = write_release(self.root)
        (self.root/'escape').symlink_to(self.root.parent)
        manifest['records'][0]['path'] = 'escape/outside'; approve(self.root, manifest)
        with self.assertRaisesRegex(ValueError, 'escapes release directory'):
            load_release(self.root)

    def test_cli_reports_only_mechanical_scope_and_draft_rejection(self):
        manifest = write_release(self.root)
        result = subprocess.run([sys.executable, '-m', 'qwen_one_two.training.release', str(self.root)],
                                text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(json.loads(result.stdout)['semantic_authentication'])
        manifest['state'] = 'draft'; approve(self.root, manifest)
        result = subprocess.run([sys.executable, '-m', 'qwen_one_two.training.release', str(self.root)],
                                text=True, capture_output=True)
        self.assertEqual(result.returncode, 1)
        self.assertIn('REJECTED', result.stderr)


if __name__ == '__main__':
    unittest.main()
