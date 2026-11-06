"""CUDA-only block application/restore checks, no fitting or backward calls."""
import unittest
import torch
from qwen_one_two.g1_proof import configure
from qwen_one_two.training.diagnostics import BlockBank,BLOCKS,same_bytes


class BlockReplacementGPU(unittest.TestCase):
    @classmethod
    def setUpClass(cls):configure()

    def sources(self):
        baseline={f'pooler.{branch}.{block}'+('' if block=='queries' else '.weight'):
                  torch.randn(3,4,device='cuda') for branch in ('shared','target') for block in BLOCKS}
        baseline['T.0']=torch.randn(2,3,device='cuda')
        donor={n:v.clone()+(1 if n.startswith('pooler.') else 0) for n,v in baseline.items()}
        return baseline,donor

    def test_exhaustive_out_of_order_apply_restore_and_source_isolation(self):
        a,b=self.sources();bank=BlockBank(a,b);current={n:v.clone().requires_grad_() for n,v in a.items()}
        for mask in (*reversed(range(32)),31,0):
            result=bank.apply(current,mask)
            self.assertEqual(len(result['changed_tensors']),2*mask.bit_count())
            for n,v in current.items():
                expected=b[n] if n in bank.groups and mask&(1<<bank.groups[n]) else a[n]
                self.assertTrue(same_bytes(v,expected));self.assertIsNone(v.grad)
        self.assertTrue(all(same_bytes(bank.baseline[n],a[n]) and same_bytes(bank.donor[n],b[n]) for n in a))
        a[next(iter(a))].zero_();self.assertFalse(same_bytes(bank.baseline[next(iter(a))],a[next(iter(a))]))

    def test_invalid_sources_destination_and_masks_reject(self):
        a,b=self.sources();b['T.0'].add_(1)
        with self.assertRaisesRegex(ValueError,'decoder/codec'):BlockBank(a,b)
        a,b=self.sources();bank=BlockBank(a,b);current={n:v.clone() for n,v in a.items()}
        for mask in (-1,32,True):
            with self.assertRaisesRegex(ValueError,'mask'):bank.apply(current,mask)
        current['T.0'].add_(1)
        with self.assertRaisesRegex(ValueError,'nonpooler'):bank.apply(current,1)
        a,b=self.sources();b['pooler.shared.queries']=b['pooler.shared.queries'][:1]
        with self.assertRaisesRegex(ValueError,'shape/dtype'):BlockBank(a,b)
