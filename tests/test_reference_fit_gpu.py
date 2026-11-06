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

    def test_target_partition_preserves_outputs_isolates_gradients_and_exports(self):
        import tempfile
        from pathlib import Path
        from safetensors.torch import load_file
        from scripts.fit_reference import export_reference
        from qwen_one_two.training.inputs import SLOTS
        from qwen_one_two.training.binding import wire_payload
        ref=Reference(self.b,self.c);states=torch.randn(20,1024,device='cuda')
        with torch.no_grad():original={role:ref.pooler(states,*slots) for role,slots in SLOTS.items()}
        decoder=ref.decoder_fingerprint();ref.split_pooler()
        self.assertEqual(ref.pooler_kind,'target-v2')
        self.assertEqual(sum(m.queries.shape[0] for m in (ref.pooler.shared,ref.pooler.target)),30)
        with torch.no_grad():
            for role,slots in SLOTS.items():self.assertTrue(torch.equal(original[role],ref.pooler(states,*slots)),role)
        pointers=[p.untyped_storage().data_ptr() for p in ref.pooler.parameters()]
        self.assertEqual(len(pointers),len(set(pointers)))
        ref.train_pooler_only()
        target=wire_payload(ref,ref.pooler(states,*SLOTS['target']),'target')
        loss=ref.reconstruct_teacher_forced(target,[10,11,248046])
        named=list(ref.pooler.named_parameters())
        grads=torch.autograd.grad(loss,[p for _,p in named],allow_unused=True)
        for (name,_),gradient in zip(named,grads):
            if name.startswith('target.'):
                self.assertIsNotNone(gradient);self.assertTrue(bool(gradient.isfinite().all()))
            else:self.assertIsNone(gradient,name)
        self.assertGreater(sum(float(g.norm()) for g in grads if g is not None),1e-10)
        self.assertEqual(ref.decoder_fingerprint(),decoder)
        ref.freeze()
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'reference';export_reference(path,ref,{'scope':'GPU schema fixture; never semantic qualification'})
            restored=Reference(self.b,self.c,pooler_kind='target-v2')
            restored.load_state_dict({k.removeprefix('reference.'):v for k,v in load_file(str(path/'weights.safetensors'),device='cuda').items()})
            restored.freeze(ref.artifact_id)
            with torch.no_grad():
                for role,slots in SLOTS.items():self.assertTrue(torch.equal(restored.pooler(states,*slots),ref.pooler(states,*slots)),role)
        self.assertTrue(all(p.grad is None for p in list(self.b.parameters())+list(self.c.parameters())))


if __name__=='__main__':unittest.main()
