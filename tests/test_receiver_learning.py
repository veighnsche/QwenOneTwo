"""Existing-release schedule/answer-blindness checks; no model execution."""
import unittest
from types import SimpleNamespace
from collections import Counter
from qwen_one_two.training.release import load_release
from qwen_one_two.training.receiver_learning import training_order,ReceiverProducer
from qwen_one_two.training.component_producer import component_view


class ReceiverLearningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train=load_release('data/training-collections-20261105/releases/train',
            expected_hash='c70d4f44135a34c00e7fa0348833b1ec56613def15f5753231589492ed35478f')
        cls.dev=load_release('data/training-collections-20261105/releases/development',
            expected_hash='5b8f4cffdc4f33e255d3a34306cac03a68c50dcd9dab905ae8d8adcf89c81e77')

    def test_complete_pass_and_explicit_real_record_wrap(self):
        order=training_order(self.train);counts=Counter(order)
        self.assertEqual(len(order),1504);self.assertEqual(len(counts),1500)
        self.assertEqual(order[-4:],order[:4]);self.assertEqual(Counter(counts.values()),{1:1496,2:4})
        self.assertEqual(order,training_order(self.train))
        self.assertTrue(all(self.train.record(k)['training_category']=='receiver' for k in order))

    def test_development_is_never_a_training_source(self):
        with self.assertRaisesRegex(ValueError,'training release'):training_order(self.dev)
        with self.assertRaisesRegex(ValueError,'never train on development'):
            ReceiverProducer(None,None,None,self.dev)

    def test_all_E_views_materialize_without_outcomes_or_reason(self):
        rows=[self.dev.record(e['id']) for e in self.dev.manifest['records'] if e['selected']
              and self.dev.record(e['id']).get('evaluation_slice')=='E']
        self.assertEqual(len(rows),130)
        for row in rows:
            view,p,_=component_view(self.dev,row,intervention=True)
            _,body=view.materialize(p)
            self.assertNotIn('reason',[r.role for r in view.references])
            self.assertNotIn(b'"supervision"',body);self.assertNotIn(b'"p_true"',body)
            self.assertNotIn(b'"acceptable_selection_ids"',body)
            self.assertNotIn(b'"score_interval"',body)


if __name__=='__main__':unittest.main()
