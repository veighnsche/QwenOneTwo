"""RD-05 checks pending RD-07. No model imports or execution at collection."""
import os
from pathlib import Path
import random
import tempfile
import unittest


class JointReceiverGPU(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if os.environ.get('QOT_RD05_GPU_QUALIFY')!='1':raise unittest.SkipTest('requires submitted RD-07')
        from qwen_one_two.g1_device import PRIMARY
        if os.environ.get('CUDA_VISIBLE_DEVICES')!=PRIMARY:raise ValueError('exact 3090 UUID required before imports')
        import torch
        from torch.nn.attention import sdpa_kernel,SDPBackend
        from qwen_one_two.g1_proof import configure
        from qwen_one_two.codec.qualify import load
        from qwen_one_two.training.reference import Reference
        from qwen_one_two.training.receiver_plan import initial_state,TRAIN_SHA
        from qwen_one_two.training.release import load_release
        from qwen_one_two.training.two_gpu import ROOT
        configure();cls.math=sdpa_kernel(SDPBackend.MATH);cls.math.__enter__()
        cls.addClassCleanup(cls.math.__exit__,None,None,None)
        cls.snapshot=os.environ['QOT_QUALIFY_SNAPSHOT']
        cls.b,cls.c,_=load(cls.snapshot,ROOT/'data/g1-proof/gate-repair-v2-20261105/diagnostic.safetensors')
        cls.ref=Reference(cls.b,cls.c,pooler_kind='target-v2');cls.ref.freeze();cls.saved,_=initial_state()
        cls.train=load_release(ROOT/'data/training-collections-20261105/releases/train',expected_hash=TRAIN_SHA)
        cls.row=next(cls.train.record(k) for k in sorted(cls.train.selected_ids)
            if cls.train.record(k)['training_category']=='receiver' and cls.train.record(k)['operation']=='Noul'
            and cls.train.record(k)['supervision']['result']['status']=='Assessed')

    def setUp(self):
        from qwen_one_two.training.receiver_learning import groups
        from qwen_one_two.training.engine import Window
        from qwen_one_two.training.checkpoint import restore_model
        from qwen_one_two.training.two_gpu import ROOT,CHECKPOINT
        for name in ('direct_role_projection','direct_role_length'):
            if hasattr(self.c,name):delattr(self.c,name)
        self.c.requires_grad_(True)
        for mode in ('R','D'):
            for p in self.b.adapter_parameters(mode):p.requires_grad_(True)
        self.owned=groups(self.b,self.c,self.ref);self.w=Window(self.owned,'components')
        restore_model(ROOT/CHECKPOINT,self.b,self.c,self.ref,self.saved['identities'],window=self.w)
        self.reference_id=self.ref.fingerprint()

    def test_joint_VJPs_and_eligible_accumulation(self):
        import torch
        from qwen_one_two.training.receiver_learning import ReceiverProducer
        from qwen_one_two.training.episodes import collect_session
        from qwen_one_two.g1_evidence import frozen_check
        # Real existing reviewed frozen-W/pooled component. Direct corpus input
        # is inadmissible; its projection path has separate RD-04 engineering checks.
        producer=ReceiverProducer(self.b,self.c,self.ref,self.train,input_profile='frozen-w-pooled-role-slots-v1')
        for _ in range(8):
            session,losses,exposure=producer(self.row)
            self.assertNotIn('failure',session.episode_report)
            self.assertEqual(set(losses),{'status','noul'})
            self.assertFalse(session.episode_report['component_admission']['receiver_behavior_accepted'])
            collect_session(self.w,session,losses,exposure)
        self.assertEqual(dict(self.w.counts),{'status':8,'noul':8})
        from scripts.train_receiver_probe import gradient_report
        diagnostics=gradient_report(self.w)
        for name in ('status','noul'):
            self.assertGreater(diagnostics['components'][name]['owners']['D']['weighted_norm'],0)
            self.assertEqual(set(diagnostics['components'][name]['owners']),{g['name'] for g in self.w.ownership})
        d={i for i,n in enumerate(self.w.names) if any(n==name for group in self.owned if group['name']=='D'
                                                   for name,_ in group['named_parameters'])}
        for component in ('status','noul'):
            self.assertTrue(any(i in d and float(g.norm())>0 for i,g in self.w.buffers[component].items()))
            self.assertTrue(all(bool(g.isfinite().all()) for g in self.w.buffers[component].values()))
        expected={}
        for component,buffers in self.w.buffers.items():
            for i,g in buffers.items():
                weight=self.w.coefficients[component]/self.w.counts[component]
                if i in expected:expected[i].add_(g,alpha=weight)
                else:expected[i]=g.mul(weight)
        self.w.combine()
        self.assertTrue(all(torch.equal(self.w.parameters[i].grad,value) for i,value in expected.items()))
        self.assertTrue(all(not p.requires_grad and p.grad is None for p in self.b.adapter_parameters('R')))
        self.assertFalse(self.c.budget_row.requires_grad);self.assertIsNone(self.c.budget_row.grad)
        self.assertTrue(all(not p.requires_grad and p.grad is None for p in self.ref.parameters()))
        self.assertEqual(self.reference_id,self.ref.fingerprint());self.assertTrue(frozen_check(self.b,self.snapshot)['pass'])
        self.assertEqual(self.w.optimizer_updates,188);self.w.abort()  # no optimizer step

    def test_complete_window_resume_and_frozen_owners(self):
        import torch
        from qwen_one_two.training.receiver_learning import extend_profile_window
        from qwen_one_two.training.receiver_profiles import direct_projection_owners
        from qwen_one_two.training.checkpoint import save,bundle_tensors,resume
        from qwen_one_two.training.reference import state_fingerprint
        from qwen_one_two.training.inputs import digest
        from qwen_one_two.training.two_gpu import ROOT
        rng=torch.cuda.get_rng_state(0).clone();python_rng=random.getstate()
        parent_state=state_fingerprint({n:p for n,p in zip(self.w.names,self.w.parameters)})
        old_moments={n:{k:v.clone() for k,v in self.w.optimizer.state[p].items()}
                     for n,p in zip(self.w.names,self.w.parameters) if p in self.w.optimizer.state}
        window=extend_profile_window(self.w,self.owned,self.c,'direct-role-states-v1')
        self.assertTrue(torch.equal(rng,torch.cuda.get_rng_state(0)));self.assertEqual(python_rng,random.getstate())
        self.assertEqual(window.ownership[-2:],direct_projection_owners())
        self.assertFalse(self.c.direct_role_length.weight.requires_grad)
        self.assertTrue(all(p not in window.optimizer.state for p in self.c.direct_role_projection.parameters()))
        for n,p in zip(window.names,window.parameters):
            if n in old_moments:
                self.assertTrue(all(torch.equal(v,window.optimizer.state[p][k]) for k,v in old_moments[n].items()))
        self.assertEqual(parent_state,state_fingerprint({n:p for n,p in zip(self.w.names,self.w.parameters)}))
        identities=dict(self.saved['identities'],protocol=digest(b'RD05-declared-resume-engineering-fixture'))
        sampler={'scope':'engineering complete-window state; no fit or semantic qualification','cursor':0}
        with tempfile.TemporaryDirectory(dir=ROOT/'data') as temp:
            path=Path(temp)/'checkpoint';weights=bundle_tensors(self.b,self.c,self.ref)
            save(path,window,weights,identities,sampler,reference_frozen=True,reference_id=self.reference_id)
            before=state_fingerprint(weights);cuda_next=torch.rand(4,device='cuda');python_next=random.random()
            with torch.no_grad():window.parameters[0].flatten()[0].add_(.25)
            restored=resume(path,window,weights,identities)
            self.assertEqual(before,state_fingerprint(weights));self.assertEqual(restored['sampler'],sampler)
            self.assertTrue(torch.equal(cuda_next,torch.rand(4,device='cuda')));self.assertEqual(python_next,random.random())
            self.assertEqual(window.optimizer_updates,188);self.assertEqual(window.scheduled_windows,188)
            self.assertEqual(self.reference_id,self.ref.fingerprint())
        self.assertTrue(all(p.grad is None for p in window.parameters))


if __name__=='__main__':unittest.main()
