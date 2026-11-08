"""RD-01 identity, selection, ambiguity and leakage checks; no model imports."""
from copy import deepcopy
import importlib.abc
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

from qwen_one_two.training.inputs import canonical, digest


class BlockModels(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'numpy', 'safetensors', 'transformers', 'peft'}:
            raise AssertionError('model library forbidden: '+fullname)


class RedesignCoverageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.block = BlockModels(); sys.meta_path.insert(0, cls.block)
        from qwen_one_two.training.release import load_release
        from qwen_one_two.training.architecture_audit import redesign_manifest
        from scripts.fit_reference import TRAIN_SHA, DEV_SHA
        root = Path(__file__).resolve().parents[1]/'data/training-collections-20261105/releases'
        cls.train = load_release(root/'train', expected_hash=TRAIN_SHA)
        cls.dev = load_release(root/'development', expected_hash=DEV_SHA)
        cls.manifest = redesign_manifest(cls.train, cls.dev)
        cls.index = {e['id']: e for e in cls.manifest['inventory']}

    @classmethod
    def tearDownClass(cls):
        sys.meta_path.remove(cls.block)

    def test_full_prepare_is_deterministic_with_model_imports_blocked(self):
        from qwen_one_two.training.architecture_audit import redesign_manifest
        self.assertEqual(canonical(self.manifest), canonical(redesign_manifest(self.train, self.dev)))

    def test_every_selected_original_row_and_receiver_accounted(self):
        self.assertEqual(len(self.index), 9000)
        self.assertEqual(self.manifest['receiver_rows_accounted'], 1500)
        self.assertEqual(set(self.manifest['panels']['train_receiver']),
                         {k for k in self.train.selected_ids if self.train.record(k)['training_category'] == 'receiver'})
        self.assertEqual(len(self.manifest['panels']['development_E']), 130)
        self.assertEqual(len(self.manifest['panels']['development_X']), 180)
        self.assertEqual(len(self.manifest['separate_reader_X_ids']), 20)
        self.assertEqual(self.manifest['native_supervision']['training']['language_replay']['operation_status'], {})
        self.assertEqual(self.manifest['native_supervision']['training']['reference']['operation_status'], {})

    def test_family_overlap_rejected(self):
        from qwen_one_two.training.redesign_coverage import assert_disjoint
        with self.assertRaisesRegex(ValueError, 'family overlap'):
            assert_disjoint([{'family': 'same'}], [{'family': 'same'}])

    def test_locked_rejected_before_reading(self):
        from qwen_one_two.training.architecture_audit import redesign_manifest
        with self.assertRaisesRegex(ValueError, 'no locked'):
            redesign_manifest(SimpleNamespace(purpose='training'), SimpleNamespace(purpose='locked'))

    def test_stale_release_record_review_source_and_family_rejected(self):
        from qwen_one_two.training.redesign_coverage import validate_identity
        original = self.index[self.manifest['panels']['train_receiver'][0]]
        for field, value in [('release_sha256', '0'*64), ('record_sha256', '0'*64),
                             ('family', 'foreign'), ('split', 'development'), ('reviews', [])]:
            with self.subTest(field=field):
                entry = deepcopy(original); entry[field] = value
                with self.assertRaises(ValueError): validate_identity(self.train, entry)
        entry = deepcopy(original); entry['sources'][0]['sha256'] = '0'*64
        with self.assertRaisesRegex(ValueError, 'source identity'): validate_identity(self.train, entry)

    def test_unclassified_prose_is_not_a_supported_operator(self):
        from qwen_one_two.training.redesign_coverage import reviewed_profile
        key = next(k for k in self.manifest['panels']['train_receiver'] if self.index[k]['operation'] == 'Score' and self.index[k]['status'] == 'Assessed')
        row = self.train.record(key)
        row['semantics']['rubric']['conditions'] = 'Use some unspecified nonlinear rule'
        result = reviewed_profile(row)
        self.assertEqual(result['recipe'], 'unclassified')
        self.assertEqual(result['classification'], 'ambiguous_or_unsupported')
        row['supervision']['result']['status'] = 'Insufficient'
        self.assertEqual(reviewed_profile(row)['classification'], 'not_applicable')

    def test_training_gaps_and_concurrent_shifts_remain_explicit(self):
        support = self.manifest['training_support']
        self.assertEqual(support['Noul/positive_and']['polarity'], {'false': 168, 'true': 88})
        self.assertEqual(support['Noul/negated_and']['assessed_training_records'], 0)
        self.assertEqual(support['Noul/positive_or']['assessed_training_records'], 0)
        self.assertEqual(support['Score/completion']['denominators'], {'100': 170, '20': 84})
        score = [self.index[k] for k in self.manifest['panels']['development_X'] if self.index[k]['operation'] == 'Score']
        self.assertTrue(all(e['axes']['familiar_numeric_range'] is False for e in score))
        self.assertTrue(all(e['axes']['changed_wording_or_binding'] for e in score))
        self.assertTrue(all(self.index[k]['status'] == 'Assessed'
                            for k in self.manifest['panels']['development_E/unclassified']))

    def test_hidden_outcomes_and_review_columns_do_not_enter_receiver_view(self):
        from qwen_one_two.training.component_producer import component_view
        key = self.manifest['panels']['train_receiver'][0]
        row = self.train.record(key)
        before, p, _ = component_view(self.train, row)
        row['labels'] = {'hidden': 'poison'}
        row['supervision'] = {'hidden': 'poison'}
        row['guidance'] = {'hidden': 'poison'}
        row['reviews'] = [{'hidden': 'poison'}]
        wrapper = SimpleNamespace(record=lambda k: deepcopy(row), purpose='training',
                                  selected_ids={key}, asset=self.train.asset)
        after, q, _ = component_view(wrapper, row)
        self.assertEqual(before.materialize(p), after.materialize(q))
        self.assertTrue(all(r.role != 'reason' for r in after.references))

    def test_native_annotations_do_not_become_unconditional_execution_targets(self):
        entries = [e for e in self.index.values() if e['category'] == 'interleaved']
        self.assertEqual(len(entries), 2000)
        self.assertTrue(all(not e['native']['execution_qualified'] for e in entries))
        calls = [c for e in entries for c in e['native']['calls']]
        self.assertTrue(any('missing_call_reason' in c['gaps'] for c in calls))
        self.assertTrue(any(c['choice_target_unique'] is False for c in calls))
        self.assertTrue(any('ambiguous_source_occurrences' in c['gaps'] for c in calls))
        readers = [e for e in self.index.values() if e['category'] == 'reader']
        self.assertTrue(all(e['native']['metadata_targets_present'] for e in readers))
        self.assertTrue(all(not e['native']['calls'] for e in readers))

    def test_subgroup_report_preserves_failures_and_full_denominator(self):
        from qwen_one_two.training.learning_report import summarize_coverage_learning
        ids = self.manifest['panels']['train_receiver']
        cases = [{'record': key, 'record_sha256': self.index[key]['record_sha256'],
                  'head': {}, 'failure': 'disclosed failed fixture'} for key in ids]
        report = summarize_coverage_learning(self.train, cases, self.manifest, panel='train_receiver', kind='heads')
        self.assertEqual(report['full']['scheduled'], 1500)
        self.assertEqual(sum(op['counts']['failed'] for op in report['full']['operations'].values()), 1500)
        self.assertIn('train_receiver/Score/interior', report['subgroups'])
        with self.assertRaisesRegex(ValueError, 'coverage'):
            summarize_coverage_learning(self.train, cases[:-1], self.manifest, panel='train_receiver', kind='heads')
        stale = deepcopy(self.manifest); stale['releases']['training'] = '0'*64
        with self.assertRaisesRegex(ValueError, 'stale coverage release'):
            summarize_coverage_learning(self.train, cases, stale, panel='train_receiver', kind='heads')
        changed = deepcopy(self.manifest); changed['panels']['train_receiver'] = ids[:-1]
        with self.assertRaisesRegex(ValueError, 'changed full coverage panel'):
            summarize_coverage_learning(self.train, cases[:-1], changed, panel='train_receiver', kind='heads')


if __name__ == '__main__':
    unittest.main()
