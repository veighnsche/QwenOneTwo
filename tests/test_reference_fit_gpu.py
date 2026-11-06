"""Real-model reference objective parity/ownership, before authorized fitting."""
import os
import unittest
import torch
from torch.nn.attention import sdpa_kernel,SDPBackend
from qwen_one_two.g1_proof import configure
from qwen_one_two.codec.qualify import load
from qwen_one_two.training.reference import Reference
from qwen_one_two.g1_checks import grad_error


class ReferenceFitGPU(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        configure();cls.math=sdpa_kernel(SDPBackend.MATH);cls.math.__enter__()
        cls.b,cls.c,_=load(os.environ['QOT_QUALIFY_SNAPSHOT'],'data/g1-proof/gate-repair-v2-20261105/diagnostic.safetensors')
        cls.reference=Reference(cls.b,cls.c)

    @classmethod
    def tearDownClass(cls):
        del cls.reference,cls.c,cls.b
        cls.math.__exit__(None,None,None);torch.cuda.empty_cache()

    def test_causal_teacher_forcing_matches_serial_loss_and_gradients(self):
        text='{"fields":{"criterion":"completed units divided by required units under the supplied normalization","proposition":"The work completion assessment for the supplied project"},"role":"question"}'
        targets=self.b.tokenizer.encode(text,add_special_tokens=False)+[248046]
        prefix=torch.randn(8,1024,device='cuda',requires_grad=True)
        params=[prefix,self.reference.T[1]]
        expected=self.reference.reconstruct(prefix,targets)
        eg=torch.autograd.grad(expected,params)
        actual=self.reference.reconstruct_teacher_forced(prefix,targets)
        ag=torch.autograd.grad(actual,params)
        self.assertLess(abs(float(expected.detach()-actual.detach())),1e-4)
        for left,right in zip(ag,eg):
            result=grad_error(left,right,.05,.995)
            self.assertTrue(result['pass'],result)
        self.assertTrue(all(p.grad is None for p in self.b.parameters()))
        print({'targets':len(targets),'serial_loss':float(expected.detach()),'parallel_loss':float(actual.detach()),
               'gradient_comparisons':[grad_error(a,b,.05,.995) for a,b in zip(ag,eg)]},flush=True)

    def test_future_targets_cannot_change_first_prediction(self):
        prefix=torch.randn(2,1024,device='cuda')
        # The first prediction's hidden state must remain invariant to suffixes.
        with torch.no_grad():
            a,_=self.reference(self.b.initial(),'T',torch.cat((prefix,self.b.embedding([10,11])[0])))
            b,_=self.reference(self.b.initial(),'T',torch.cat((prefix,self.b.embedding([12,13])[0])))
        self.assertTrue(torch.equal(a[0,1],b[0,1]))

    def test_target_history_ablation_matches_serial_loss_and_gradients(self):
        targets=self.b.tokenizer.encode('Property meaning in the existing reference record.',add_special_tokens=False)+[248046]
        prefix=torch.randn(2,1024,device='cuda',requires_grad=True)
        params=(prefix,self.reference.T[1])
        expected=self.reference.reconstruct(prefix,targets,target_history=False)
        eg=torch.autograd.grad(expected,params)
        actual=self.reference.reconstruct_teacher_forced(prefix,targets,target_history=False)
        ag=torch.autograd.grad(actual,params)
        self.assertLess(abs(float(actual.detach()-expected.detach())),1e-4)
        for a,e in zip(ag,eg):self.assertTrue(grad_error(a,e,.05,.995)['pass'])
        self.assertGreater(float(ag[0].norm()),1e-10)

    def test_ablated_history_never_embeds_target_tokens(self):
        prefix=torch.randn(2,1024,device='cuda',requires_grad=True)
        original=self.b.embedding
        def reject(ids):raise AssertionError('target content entered ablated decoder input')
        self.b.embedding=reject
        try:
            loss=self.reference.reconstruct_teacher_forced(prefix,[10,11,248046],target_history=False)
            gradient=torch.autograd.grad(loss,prefix)[0]
        finally:self.b.embedding=original
        self.assertTrue(bool(gradient.isfinite().all()));self.assertGreater(float(gradient.norm()),1e-10)

    def test_stationary_decoder_owns_only_pooler_and_passes_its_gradients(self):
        from qwen_one_two.training.engine import reference_groups
        from qwen_one_two.training.binding import wire_payload
        ref=Reference(self.b,self.c)
        before=ref.decoder_fingerprint();ref.train_pooler_only()
        owned={id(p) for group in reference_groups(ref) for _,p in group['named_parameters']}
        self.assertEqual(owned,{id(p) for p in ref.pooler.parameters()})
        states=torch.randn(20,1024,device='cuda')
        payload=ref.pooler(states,8,2)
        loss=ref.reconstruct_teacher_forced(wire_payload(ref,payload,'target'),[10,11,248046])
        loss.backward()
        self.assertTrue(all(p.grad is not None and bool(p.grad.isfinite().all()) for p in ref.pooler.parameters()))
        self.assertGreater(sum(float(p.grad.norm()) for p in ref.pooler.parameters()),1e-10)
        self.assertTrue(all(p.grad is None for p in list(ref.T.parameters())+list(ref.codec.parameters())+list(self.b.parameters())+list(self.c.parameters())))
        self.assertEqual(before,ref.decoder_fingerprint())


if __name__=='__main__':unittest.main()
