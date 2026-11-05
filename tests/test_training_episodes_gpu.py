"""Packet-binding and actual Session guard engineering tests on CUDA."""
import unittest
import torch
from qwen_one_two.g1_proof import configure
from qwen_one_two.codec.fixtures import base, slots
from qwen_one_two.codec.runtime import Runtime
from qwen_one_two.codec.schema import Port, Status, Kind
from qwen_one_two.codec.records import Semantic
from qwen_one_two.training.episodes import DecisionLabels, decision_losses


class PacketLossGPU(unittest.TestCase):
    @classmethod
    def setUpClass(cls):configure()

    def test_assessed_raw_loss_survives_abstention_and_divergence_masks(self):
        rt=Runtime('fixture',10,512,128);rt.startup()
        packet,*_=base(rt,Port.SCORE,evidence=False)
        raw=torch.tensor(1.4,device='cuda',requires_grad=True)
        evaluated={'raw':raw,'status_logits':torch.zeros(5,device='cuda',requires_grad=True),
                   'reason_slots':slots(4,True)}
        labels=DecisionLabels(packet.snapshot,Status.ASSESSED,score_point=.4)
        losses,report=decision_losses(rt.registry,packet.snapshot,Status.INSUFFICIENT,evaluated,labels)
        self.assertEqual(set(losses),{'status','score'})
        self.assertGreater(torch.autograd.grad(losses['score'],raw)[0].item(),0)
        losses,report=decision_losses(rt.registry,packet.snapshot,Status.ASSESSED,evaluated,
                                    DecisionLabels(None,Status.ASSESSED,score_point=.4))
        self.assertEqual(losses,{});self.assertFalse(report['supported_packet'])
        losses,_=decision_losses(rt.registry,packet.snapshot,Status.ASSESSED,evaluated,
                                DecisionLabels(packet.snapshot,Status.INSUFFICIENT,reason_slots=slots(4)))
        self.assertEqual(set(losses),{'status'})

    def test_choice_uses_bound_stable_refs_and_masks_missing_labels(self):
        rt=Runtime('fixture',10,512,128);rt.startup()
        packet,_,_,candidates,*_=base(rt,Port.CHOICE,evidence=False)
        evaluated={'pointer_logits':torch.zeros(2,device='cuda',requires_grad=True),
                   'status_logits':torch.zeros(5,device='cuda',requires_grad=True),
                   'feasibility_logits':torch.zeros(2,4,device='cuda',requires_grad=True)}
        labels=DecisionLabels(packet.snapshot,Status.ASSESSED,choice=candidates,
                              feasibility=((candidates[0],0),))
        losses,_=decision_losses(rt.registry,packet.snapshot,Status.ASSESSED,evaluated,labels)
        self.assertAlmostEqual(losses['choice'].item(),0.)
        grads=torch.autograd.grad(losses['feasibility'],evaluated['feasibility_logits'])[0]
        self.assertEqual(grads[1].abs().sum().item(),0.)
        foreign=rt.registry.publish(Kind.ALTERNATIVE,Semantic(slots(4)))
        with self.assertRaises(ValueError):decision_losses(rt.registry,packet.snapshot,Status.ASSESSED,evaluated,
                DecisionLabels(packet.snapshot,Status.ASSESSED,choice=(foreign,)))
