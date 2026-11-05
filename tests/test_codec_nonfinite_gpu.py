"""Reject invalid neural values before clamp/argmax; no capability fitting."""
import unittest
import torch
from torch.nn import functional as F
from qwen_one_two.g1_proof import configure
from qwen_one_two.g1_backbone import EffectiveTable
from qwen_one_two.codec.neural import Codec, norm
from qwen_one_two.codec.schema import Port


class NonfiniteGPU(unittest.TestCase):
    def setUp(self):
        configure()
        with torch.device('cuda'):
            embedding=torch.randn(4000,1024)
            self.codec=Codec(torch.tensor(1.),embedding,{'q.weight':torch.randn(256,1024),'k.weight':torch.randn(256,1024)})
            self.codec.table=EffectiveTable(embedding,self.codec.budget_row,self.codec.score_row)

    def test_score_overflow_rejected_before_clamp(self):
        with torch.no_grad():self.codec.score_row.fill_(1e38)
        self.assertTrue(bool(self.codec.score_row.isfinite().all()))
        with self.assertRaisesRegex(ValueError,'nonfinite'):
            self.codec.evaluate(torch.ones(1024,device='cuda'),Port.SCORE)

    def test_gate_and_pointer_overflow_rejected_before_argmax(self):
        with torch.no_grad():self.codec.gate.weight.fill_(1e38)
        with self.assertRaisesRegex(ValueError,'nonfinite'):
            self.codec.gate(torch.ones(256,device='cuda'))
        with torch.no_grad():
            self.codec.pointers['source'].weight.fill_(1e18)
            self.codec.pointer_key.weight.fill_(1e18)
        with self.assertRaisesRegex(ValueError,'nonfinite'):
            self.codec.pointer('source',torch.ones(256,device='cuda'),torch.ones(2,1024,device='cuda'))

    def test_normalization_overflow_rejected(self):
        with self.assertRaisesRegex(ValueError,'nonfinite'):
            norm(torch.full((1024,),1e30,device='cuda'))

    def test_finite_head_preserves_values_and_gradients(self):
        x=torch.randn(256,device='cuda',requires_grad=True)
        actual=self.codec.gate(x)
        expected=F.linear(x,self.codec.gate.weight,self.codec.gate.bias)
        self.assertTrue(torch.equal(actual,expected))
        ga=torch.autograd.grad(actual.square().sum(),x,retain_graph=True)[0]
        ge=torch.autograd.grad(expected.square().sum(),x)[0]
        self.assertTrue(torch.equal(ga,ge))

    def test_evaluator_products_reject_nonfinite_before_selection(self):
        with torch.no_grad():
            self.codec.qev_q.weight.fill_(1e18)
            self.codec.qev_k.weight.fill_(1e18)
            self.codec.prototypes.fill_(1)
        for port in (Port.NOUL,Port.CHOICE):
            with self.assertRaisesRegex(ValueError,'nonfinite'):
                self.codec.evaluate(torch.ones(1024,device='cuda'),port,
                                    torch.ones(2,1024,device='cuda') if port==Port.CHOICE else None)

    def test_invalid_native_gate_terminates_without_dispatch(self):
        from types import SimpleNamespace
        from qwen_one_two.codec.runtime import Runtime
        from qwen_one_two.codec.session import Session
        rt=Runtime('numeric-guard-fixture',0,100,1);rt.startup()
        # Mechanical boundary fixture; no claim of a model prefill from this stub.
        backbone=SimpleNamespace(initial=lambda:SimpleNamespace(layers=(),position=0,lineage=1))
        session=Session(backbone,self.codec,rt)
        session.hidden=torch.ones(1,1,1024,device='cuda')
        with torch.no_grad():self.codec.gate.bias.fill_(float('nan'))
        with self.assertRaisesRegex(ValueError,'nonfinite'):session.native_step()
        self.assertEqual(rt.terminal,'Failed');self.assertEqual(rt.dispatched,0)
