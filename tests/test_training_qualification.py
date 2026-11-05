"""Model-free numeric evidence binding tests; fixture pass flags are schema only."""
from pathlib import Path
import tempfile
import unittest
from qwen_one_two.training.inputs import canonical, digest
from qwen_one_two.training.qualification import load_numeric_evidence
from qwen_one_two.training.release import decode
from qualification_fixtures import write_numeric_fixture


class QualificationEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)

    def test_complete_byte_binding_and_snapshot(self):
        identity=write_numeric_fixture(self.root)
        evidence=load_numeric_evidence(self.root,expected_hash=identity)
        self.assertEqual(evidence.summary['cases'],108)
        evidence.summary['cases']=0
        self.assertEqual(evidence.summary['cases'],108)
        (self.root/'report.json').write_bytes(b'changed')
        self.assertEqual(evidence.identity,identity)
        with self.assertRaisesRegex(ValueError,'hash mismatch'):
            load_numeric_evidence(self.root,expected_hash=identity)

    def test_child_bytes_and_archived_sources_cannot_change(self):
        identity=write_numeric_fixture(self.root)
        (self.root/'1024-port1/report.json').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'hash mismatch'):
            load_numeric_evidence(self.root,expected_hash=identity)
        identity=write_numeric_fixture(self.root)
        (self.root/'source-snapshot/qwen_one_two/codec/schema.py').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'hash mismatch'):
            load_numeric_evidence(self.root,expected_hash=identity)

    def test_failed_partial_duplicate_or_foreign_matrix_rejects(self):
        for change in (lambda r:r.update(protocol='v3-512'), lambda r:r.update(status='FAILED'),
                       lambda r:r['cases'].pop(), lambda r:r['cases'].append(r['cases'][0]),
                       lambda r:r.update(gpu_uuid='wrong-device')):
            identity=write_numeric_fixture(self.root)
            report=decode((self.root/'report.json').read_bytes());change(report)
            raw=canonical(report);(self.root/'report.json').write_bytes(raw)
            with self.assertRaises(ValueError):
                load_numeric_evidence(self.root,expected_hash=digest(raw))

    def test_rehashed_broken_lineage_or_accounting_rejects(self):
        for field,value in (('prefills',2),('lineage',2),('total',1023),('held_positions',1),('consumed',0)):
            def mutate(report,child,port):
                child['cases'][0]['summary'][field]=value
            identity=write_numeric_fixture(self.root,mutate=mutate)
            with self.assertRaises(ValueError):
                load_numeric_evidence(self.root,expected_hash=identity)

    def test_rehashed_child_integrity_or_status_coverage_rejects(self):
        for change in (lambda c:c['frozen_base'].update(base_tied_storage=False),
                       lambda c:c.update(all_live_parameters_unchanged=False),
                       lambda c:c.update(source_hashes={'schema.py':'0'*64}),
                       lambda c:[r.update(comparison={'pass':'yes'}) for r in c['cases']],
                       lambda c:[r.update(calls=[{'status':1}]*len(r['calls'])) for r in c['cases']]):
            identity=write_numeric_fixture(self.root,mutate=lambda r,c,p:change(c))
            with self.assertRaises(ValueError):
                load_numeric_evidence(self.root,expected_hash=identity)

    def test_rehashed_aggregate_cannot_disagree_with_child_traces(self):
        write_numeric_fixture(self.root)
        report=decode((self.root/'report.json').read_bytes())
        report['cases'][0]['summary']['prefills']=2
        raw=canonical(report);(self.root/'report.json').write_bytes(raw)
        with self.assertRaisesRegex(ValueError,'aggregate/child trace mismatch'):
            load_numeric_evidence(self.root,expected_hash=digest(raw))


if __name__ == '__main__':
    unittest.main()
