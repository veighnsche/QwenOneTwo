"""Model-free driver-boundary simulation; no Session/model/optimizer is run.

Release objects use explicit TEST-ONLY schema markers, not real human evidence.
Actual one-cache Session validation remains covered by the earlier CUDA tests.
"""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from qwen_one_two.training.driver import Driver
from qwen_one_two.training.release import load_release
from qwen_one_two.training.sampler import Sampler
from release_fixtures import row, pilot_rows, write_release


class WindowFixture:
    """Orchestration counters only; never fits or updates anything."""
    def __init__(self, phase='guided_joint'):
        self.phase, self.scheduled_windows, self.examples = phase, 0, 0
        self.aborts = 0

    def abort(self):
        self.examples = 0; self.aborts += 1

    def step(self):
        assert self.examples == 8
        self.scheduled_windows += 1; self.examples = 0
        return {'stepped':False, 'simulation_only':True}


class DriverTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temp.cleanup)
        root = Path(cls.temp.name)
        write_release(root/'pilot', pilot_rows(), purpose='pilot')
        cls.pilot = load_release(root/'pilot')
        records = [row(category, split='train', training_category=category)
                   for category in ('interleaved','direct','language_replay','decision_replay')]
        write_release(root/'training', records, purpose='training', pilot_hash=cls.pilot.identity)
        cls.release = load_release(root/'training')
        cls.prerequisites = {'g1_core':True, 'numeric_1024':{'protocol':'v3-1024-v1',
            'protocol_complete':True, 'status':'PASSED', 'report_hash':'b'*64}}

    def driver(self, *, release=None, pilot=None, sampler=None, prerequisites=None):
        release = self.release if release is None else release
        return Driver(WindowFixture(), sampler or Sampler.from_release(self.release,'guided_joint'),
                      release, self.pilot if pilot is None else pilot,
                      self.prerequisites if prerequisites is None else prerequisites)

    def test_unverified_strings_and_wrong_release_purpose_reject(self):
        for release in ({'pilot_review_hash':'anything'}, self.pilot):
            with self.assertRaises(ValueError):
                self.driver(release=release)
        with self.assertRaises(ValueError):
            self.driver(pilot='arbitrary-hash')

    def test_manifest_identity_and_full_catalogue_must_match(self):
        sampler = Sampler.from_release(self.release,'guided_joint')
        sampler.release_hash = '0'*64
        with self.assertRaisesRegex(ValueError,'identity mismatch'):
            self.driver(sampler=sampler)
        sampler = Sampler.from_release(self.release,'guided_joint')
        sampler.rows['direct']['category'] = 'interleaved'
        with self.assertRaisesRegex(ValueError,'catalogue mismatch'):
            self.driver(sampler=sampler)

    def test_pilot_hash_and_numeric_prerequisite_reject_before_producer(self):
        with tempfile.TemporaryDirectory() as temp:
            records = [row('x', split='train', training_category='direct')]
            write_release(temp, records, purpose='training', pilot_hash='0'*64)
            release = load_release(temp)
            with self.assertRaisesRegex(ValueError,'pilot release binding'):
                self.driver(release=release,sampler=Sampler.from_release(release,'guided_joint'))
        bad = deepcopy(self.prerequisites); bad['numeric_1024']['protocol_complete'] = False
        with self.assertRaisesRegex(ValueError,'1024 qualification'):
            self.driver(prerequisites=bad)

    def test_repeated_exposures_receive_original_bytes_and_bound_report(self):
        driver = self.driver()
        calls = []
        def producer(record,phase,window):
            self.assertTrue(record['human_input']['prompt'].startswith('SCHEMA FIXTURE'))
            calls.append((record['id'],phase,window))
            record['human_input']['prompt'] = 'producer mutation'
            return None, {}, None
        def collect(window,*args):
            window.examples += 1
            return {'simulation_only':True}
        with patch('qwen_one_two.training.driver.collect_session',collect):
            report = driver.run_window(producer)
        self.assertEqual(len(calls),8)
        self.assertEqual(report['release_sha256'], self.release.identity)
        self.assertEqual(report['pilot_release_sha256'], self.pilot.identity)
        self.assertEqual(report['sampler']['windows'],1)
        self.assertFalse(report['window']['stepped'])

    def test_producer_failure_keeps_failed_ids_and_requires_explicit_resume(self):
        driver = self.driver(); calls = []
        def producer(record,*args):
            calls.append(record['id'])
            raise RuntimeError('schema simulation failure')
        with self.assertRaisesRegex(RuntimeError,'simulation failure'):
            driver.run_window(producer)
        self.assertEqual(len(calls),1)
        self.assertEqual(driver.sampler.windows,1)
        self.assertEqual(driver.window.scheduled_windows,0)
        self.assertEqual(driver.window.aborts,1)
        self.assertEqual(driver.reports[-1]['status'],'ABORTED')
        with self.assertRaisesRegex(ValueError,'explicit checkpoint resume'):
            driver.run_window(producer)

    def test_mutation_after_initial_binding_rejects_without_exposure(self):
        driver = self.driver()
        driver.sampler.rows['direct']['category'] = 'interleaved'
        with self.assertRaisesRegex(ValueError,'catalogue mismatch'):
            driver.run_window(lambda *args:self.fail('must reject before producer'))
        self.assertEqual(driver.sampler.windows,0)

    def test_forged_receiver_catalogue_labels_reject(self):
        with tempfile.TemporaryDirectory() as temp:
            record = row('receiver',split='train',training_category='receiver',operation='Choice',
                         labels={'schema_version':'test-only-v1','status':'Assessed'})
            write_release(temp,[record],purpose='training',pilot_hash=self.pilot.identity)
            release = load_release(temp)
            sampler = Sampler.from_release(release,'components')
            sampler.rows['receiver']['status'] = 'Insufficient'
            with self.assertRaisesRegex(ValueError,'catalogue mismatch'):
                Driver(WindowFixture('components'),sampler,release,self.pilot,self.prerequisites)

    def test_pilot_families_and_exact_inputs_cannot_leak_into_training(self):
        pilot_record = next(iter(self.pilot.records.values()))
        for field in ('family','human_input'):
            with tempfile.TemporaryDirectory() as temp:
                record = row('leak',split='train',training_category='direct',**{field:pilot_record[field]})
                write_release(temp,[record],purpose='training',pilot_hash=self.pilot.identity)
                release = load_release(temp)
                with self.assertRaisesRegex(ValueError,'leakage'):
                    self.driver(release=release,sampler=Sampler.from_release(release,'guided_joint'))


if __name__ == '__main__':
    unittest.main()
