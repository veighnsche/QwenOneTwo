import os
from types import SimpleNamespace
import unittest

from qwen_one_two.g1_accounting import Ledger
from qwen_one_two.g1_proof import Episode
from qwen_one_two.g1_backbone import GPU_UUID


class GateBoundaryTest(unittest.TestCase):
    def test_supervision_reads_pre_request_counters(self):
        ep = Episode.__new__(Episode)
        ep.ledger = Ledger(33, b=44, attempts=1, writer=3, records=4, consumed=True)
        ep.codec = object()
        ep.discrete = []
        original = vars(ep.ledger).copy()

        class Observed(Exception):
            pass

        def inspect(boundary, target):
            self.assertEqual((boundary, target), ('request', 1))
            self.assertEqual(vars(ep.ledger), original)
            raise Observed()

        ep.gate_logits = inspect
        with self.assertRaises(Observed):
            ep.call(1)
        self.assertEqual(vars(ep.ledger), original)


@unittest.skipUnless(os.environ.get('CUDA_VISIBLE_DEVICES') == GPU_UUID,
                     'Explicit RTX 3090 CUDA selection required')
class GateCudaTest(unittest.TestCase):
    def test_choice_is_not_masked_or_replaced(self):
        import torch
        self.assertIn('RTX 3090', torch.cuda.get_device_name(0))
        ep = Episode.__new__(Episode)
        ep.ledger = Ledger(0)
        ep.state = SimpleNamespace(position=0)
        ep.hidden = torch.zeros(1, 1, 1024, device='cuda')
        ep.gate_observations = []
        ep.codec = SimpleNamespace(
            feature=lambda h, counters: h,
            gate=lambda f: torch.tensor([-2., 0., 3., 1.], device='cuda'))
        with self.assertRaisesRegex(RuntimeError, 'selected Choice'):
            ep.release_generation()
        g = ep.gate_observations[-1]
        self.assertEqual((g['raw_action'], g['executed_action']), (2, 2))
        self.assertNotIn('resource_mask', g)


if __name__ == '__main__':
    unittest.main()
