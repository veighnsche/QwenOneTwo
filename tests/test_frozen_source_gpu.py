"""RD-03 assertions owned by RD-07. No model imports during collection."""
import os
from pathlib import Path
import unittest

from qwen_one_two.training.frozen_source import GPU_CHECKS


class FrozenSourceGPU(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if os.environ.get('QOT_RD03_GPU_QUALIFY')!='1':
            raise unittest.SkipTest('RD-03 GPU assertions are pending explicit RD-07 authorization')
        from qwen_one_two.g1_device import PRIMARY
        if os.environ.get('CUDA_VISIBLE_DEVICES')!=PRIMARY:
            raise ValueError('RD-03 qualification requires the exact 3090 UUID before model imports')
        import torch
        from torch.nn.attention import sdpa_kernel,SDPBackend
        from qwen_one_two.g1_proof import configure
        from qwen_one_two.codec.qualify import load
        from qwen_one_two.training.reference import Reference
        from qwen_one_two.training.receiver_learning import groups
        from qwen_one_two.training.checkpoint import restore_model
        from qwen_one_two.training.receiver_plan import initial_state,TRAIN_SHA
        from qwen_one_two.training.release import load_release
        from qwen_one_two.training.two_gpu import ROOT,CHECKPOINT
        configure();cls.math=sdpa_kernel(SDPBackend.MATH);cls.math.__enter__()
        cls.addClassCleanup(cls.math.__exit__,None,None,None)
        cls.snapshot=os.environ['QOT_QUALIFY_SNAPSHOT']
        cls.b,cls.c,_=load(cls.snapshot,ROOT/'data/g1-proof/gate-repair-v2-20261105/diagnostic.safetensors')
        cls.ref=Reference(cls.b,cls.c,pooler_kind='target-v2');cls.ref.freeze()
        groups(cls.b,cls.c,cls.ref)
        saved,_=initial_state();restore_model(ROOT/CHECKPOINT,cls.b,cls.c,cls.ref,saved['identities'])
        cls.train=load_release(ROOT/'data/training-collections-20261105/releases/train',expected_hash=TRAIN_SHA)
        cls.row=next(cls.train.record(k) for k in sorted(cls.train.selected_ids)
                     if cls.train.record(k)['training_category']=='receiver' and cls.train.record(k)['operation']=='Noul'
                     and cls.train.record(k)['supervision']['result']['status']=='Assessed')
        from qwen_one_two.training.component_producer import component_view
        cls.view,cls.provenance,_=component_view(cls.train,cls.row)

    def encoder(self,cache):
        from qwen_one_two.training.clean import CleanEncoder
        from qwen_one_two.training.frozen_source import FROZEN_SOURCE
        from qwen_one_two.training.two_gpu import CHECKPOINT_SHA
        from qwen_one_two.training.inputs import digest
        return CleanEncoder(self.b,self.ref,checkpoint_hash=CHECKPOINT_SHA,profile_hash=digest(b'RD03-GPU-v1'),
                            source_policy=FROZEN_SOURCE,cache_sources=cache,cache_references=True)

    def states(self,encoded):
        return [encoded['registry'].get(ref).states for ref in encoded['sources']]

    def test_frozen_features_ignore_D_updates(self):
        import torch
        from qwen_one_two.g1_backbone import MODE
        encoder=self.encoder(False)
        before=encoder.encode(self.view,self.provenance,run='rd03-D-before')
        selected=next(iter(self.b.adapter_parameters('D')));original=selected.detach().clone()
        ownership=[(id(p),p.requires_grad) for p in self.b.parameters()]
        try:
            with torch.no_grad():selected.flatten()[0].add_(.25)
            self.assertFalse(torch.equal(selected,original))
            after=encoder.encode(self.view,self.provenance,run='rd03-D-after')
            self.assertTrue(all(torch.equal(a,b) for a,b in zip(self.states(before),self.states(after))))
            self.assertTrue(all(torch.equal(a,b) for a,b in zip(before['slots'],after['slots'])))
            self.assertEqual(before['input_hash'],after['input_hash'])
            self.assertEqual(ownership,[(id(p),p.requires_grad) for p in self.b.parameters()])
            self.assertEqual(MODE.get(),'R')
        finally:
            with torch.no_grad():selected.copy_(original)

    def test_cache_matches_uncached_and_preserves_bindings(self):
        import torch
        uncached=self.encoder(False);cached=self.encoder(True)
        a=uncached.encode(self.view,self.provenance,run='rd03-uncached')
        b=cached.encode(self.view,self.provenance,run='rd03-cache-miss')
        c=cached.encode(self.view,self.provenance,run='rd03-cache-hit')
        self.assertEqual(cached.source_cache_hits,len(cached.source_cache))
        self.assertEqual(a['audit'],b['audit']);self.assertEqual(b['audit'],c['audit'])
        from qwen_one_two.training.clean import CleanEncoder
        from qwen_one_two.training.two_gpu import CHECKPOINT_SHA
        from qwen_one_two.training.inputs import digest
        legacy=CleanEncoder(self.b,self.ref,checkpoint_hash=CHECKPOINT_SHA,profile_hash=digest(b'RD03-GPU-v1'))
        self.assertTrue(all(torch.equal(legacy.reference_payload(ref),slot)
                            for ref,slot in zip(self.view.references,a['slots'])))
        self.assertTrue(all(torch.equal(x,y) and torch.equal(y,z)
                            for x,y,z in zip(self.states(a),self.states(b),self.states(c))))
        self.assertEqual([a['registry'].get(r).offsets for r in a['sources']],
                         [c['registry'].get(r).offsets for r in c['sources']])
        self.assertEqual([a['registry'].get(r).token_ids for r in a['sources']],
                         [c['registry'].get(r).token_ids for r in c['sources']])
        with torch.no_grad():self.states(b)[0].add_(1)
        d=cached.encode(self.view,self.provenance,run='rd03-cache-clone')
        self.assertTrue(all(torch.equal(x,y) for x,y in zip(self.states(a),self.states(d))))
        with self.assertRaises(ValueError):a['registry'].get(c['sources'][0])

    def test_consumer_gradients_and_frozen_W(self):
        import torch
        from qwen_one_two.training.receiver_learning import ReceiverProducer
        from qwen_one_two.training.frozen_source import bind_frozen_source
        from qwen_one_two.g1_evidence import frozen_check
        producer=ReceiverProducer(self.b,self.c,self.ref,self.train,
                                  input_profile='frozen-w-pooled-role-slots-v1',cache_sources=True)
        original_reference=self.ref.fingerprint()
        session,losses,_=producer(self.row)
        self.assertNotIn('failure',session.episode_report)
        d=list(self.b.adapter_parameters('D'));source=list(self.c.source.parameters())
        gradients=torch.autograd.grad(losses['status'],d+source,allow_unused=True)
        self.assertTrue(any(g is not None and g.norm().item()>0 for g in gradients[:len(d)]))
        self.assertTrue(any(g is not None and g.norm().item()>0 for g in gradients[len(d):]))
        self.assertTrue(all(g is None or bool(g.isfinite().all()) for g in gradients))
        self.assertEqual(session.summary()['prefills'],1);self.assertEqual(session.summary()['lineage'],1)
        frozen=bind_frozen_source(self.b)
        self.assertEqual(frozen['verified_W']['parameters_checked'],320)
        self.assertTrue(frozen_check(self.b,self.snapshot)['pass'])
        self.assertEqual(original_reference,self.ref.fingerprint())
        self.assertTrue(all(p.grad is None for p in self.b.parameters()))


if __name__=='__main__':unittest.main()
