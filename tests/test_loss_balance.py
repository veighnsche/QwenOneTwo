"""Source/data/control identity checks; no model-runtime imports."""
from copy import deepcopy
from pathlib import Path
import unittest
from scripts.train_receiver_probe import loss_balance_baseline,BALANCE_OVERRIDES
from qwen_one_two.training.release import decode
from qwen_one_two.training.two_gpu import source_hashes,ROOT
from qwen_one_two.training.manifest import file_hash


class LossBalanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.path=ROOT/'data/receiver-probe-20261107/run'
        cls.panel=decode((cls.path/'plan.json').read_bytes())['panel']
        cls.sources=source_hashes()
        cls.sources['scripts/train_receiver_probe.py']=file_hash(ROOT/'scripts/train_receiver_probe.py')

    def test_completed_control_binds_without_loading_weights(self):
        binding,report=loss_balance_baseline(self.path,self.panel,self.sources)
        self.assertEqual(binding['overrides'],{'status':.1,'feasibility':.05})
        self.assertEqual(report['optimizer_updates'],32)
        self.assertEqual(binding['changed_sources'],['scripts/train_receiver_probe.py'])

    def test_changed_order_or_model_source_is_rejected(self):
        with self.assertRaisesRegex(ValueError,'panel/order'):
            loss_balance_baseline(self.path,list(reversed(self.panel)),self.sources)
        changed=dict(self.sources);changed['qwen_one_two/training/engine.py']='0'*64
        with self.assertRaisesRegex(ValueError,'unrelated comparison source'):
            loss_balance_baseline(self.path,self.panel,changed)

    def test_changed_source_set_or_wrong_control_is_rejected(self):
        changed=dict(self.sources);changed.pop('qwen_one_two/training/engine.py')
        with self.assertRaisesRegex(ValueError,'source set'):
            loss_balance_baseline(self.path,self.panel,changed)
        with self.assertRaisesRegex(ValueError,'baseline belongs'):
            loss_balance_baseline('/tmp',self.panel,self.sources)


if __name__=='__main__':unittest.main()
