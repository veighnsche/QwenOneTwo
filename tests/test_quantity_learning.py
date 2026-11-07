"""Model-free coverage, held-out admission and reporting checks."""
import subprocess
import sys
import unittest
from qwen_one_two.training.quantity_learning import training_panel,quantity_summary,PASSES
from qwen_one_two.training.release import load_release
from qwen_one_two.training.two_gpu import ROOT
from scripts.fit_reference import TRAIN_SHA,DEV_SHA


class QuantityLearningTests(unittest.TestCase):
    def test_complete_training_only_panel(self):
        train=load_release(ROOT/'data/training-collections-20261105/releases/train',expected_hash=TRAIN_SHA)
        dev=load_release(ROOT/'data/training-collections-20261105/releases/development',expected_hash=DEV_SHA)
        panel=training_panel(train)
        self.assertEqual(len(panel),254);self.assertEqual(len({x['family'] for x in panel}),96)
        self.assertEqual({x['counts'][0] for x in panel},{20,100})
        self.assertEqual(len(panel)*PASSES,1016);self.assertEqual(len(panel)*PASSES%8,0)
        self.assertEqual(sum(x['target'] in (0.,1.) for x in panel),170)
        self.assertEqual(panel,training_panel(train))
        with self.assertRaisesRegex(ValueError,'training release'):training_panel(dev)
        self.assertFalse({x['family'] for x in panel}&{dev.record(e['id'])['family'] for e in dev.manifest['records'] if e['selected']})

    def test_counts_have_separate_exact_and_rounded_metrics_with_missing_denominator(self):
        rows=[{'quantities':{'predicted':[19.8,4.1],'target':[20,4]}},
              {'quantities':{'predicted':[20.,5.9],'target':[20,5]}},{}]
        result=quantity_summary(rows)
        self.assertEqual(result['scheduled'],3);self.assertEqual(result['measured'],2);self.assertEqual(result['missing'],1)
        self.assertEqual(result['required']['exact_correct'],1)
        self.assertEqual(result['required']['nearest_integer_correct'],2)
        self.assertEqual(result['both_nearest_integer_correct'],1)
        self.assertAlmostEqual(result['completed']['mae'],.5)
        with self.assertRaises(ValueError):quantity_summary([{'quantities':{'predicted':[float('nan'),0],'target':[20,0]}}])

    def test_metadata_import_never_imports_models(self):
        subprocess.run([sys.executable,'-c',"""
import sys
class Block:
 def find_spec(self,name,*args):
  if name.split('.')[0] in {'torch','numpy','safetensors','transformers'}:raise RuntimeError(name)
sys.meta_path.insert(0,Block())
import scripts.train_quantity_comparison
"""],check=True)


if __name__=='__main__':unittest.main()
