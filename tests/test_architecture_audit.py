"""Metadata statistics, not model/probe execution."""
import unittest
from types import SimpleNamespace
from qwen_one_two.training.architecture_audit import summarize_rows, audit


class ArchitectureAuditTests(unittest.TestCase):
    def test_statuses_do_not_inflate_value_coverage(self):
        def row(op, status, payload=None):
            return {'operation': op, 'family': 'one', 'domain': 'scheduling',
                    'semantics': {'rubric': {'direction': 'higher'}},
                    'supervision': {'result': {'operation': op, 'status': status, 'payload': payload}}}
        report = summarize_rows([row('Noul', 'Assessed', {'p_true': 0}),
                                 row('Noul', 'Insufficient'),
                                 row('Score', 'Assessed', {'score_interval': [.4, .4]}),
                                 row('Score', 'Assessed', {'score_interval': [.1, .3]})])
        self.assertEqual(report['records'], 4)
        self.assertEqual(report['families'], 1)
        self.assertEqual(report['noul_truth'], {'false': 1})
        self.assertEqual(report['score_point_targets'], 1)
        self.assertEqual(report['score_directions'], {'higher': 2})

    def test_locked_rejected_before_reading_records(self):
        with self.assertRaisesRegex(ValueError, 'no locked'):
            audit(SimpleNamespace(purpose='training'), SimpleNamespace(purpose='locked'))


if __name__ == '__main__':
    unittest.main()
