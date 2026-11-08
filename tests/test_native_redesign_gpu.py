"""RD-06 engineering traces/VJPs pending RD-07; no models at collection."""
import os
import unittest
from unittest.mock import patch


class NativeRedesignGPU(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if os.environ.get('QOT_RD06_GPU_QUALIFY')!='1':raise unittest.SkipTest('requires submitted RD-07')
        from qwen_one_two.g1_device import PRIMARY
        if os.environ.get('CUDA_VISIBLE_DEVICES')!=PRIMARY:raise ValueError('exact 3090 UUID required before imports')
        import torch
        from torch.nn.attention import sdpa_kernel,SDPBackend
        from qwen_one_two.g1_proof import configure
        from qwen_one_two.codec.qualify import load
        from qwen_one_two.training.reference import Reference
        from qwen_one_two.training.receiver_learning import groups
        from qwen_one_two.training.receiver_plan import initial_state,TRAIN_SHA
        from qwen_one_two.training.checkpoint import restore_model
        from qwen_one_two.training.release import load_release
        from qwen_one_two.training.two_gpu import ROOT,CHECKPOINT
        configure();cls.math=sdpa_kernel(SDPBackend.MATH);cls.math.__enter__();cls.addClassCleanup(cls.math.__exit__,None,None,None)
        cls.snapshot=os.environ['QOT_QUALIFY_SNAPSHOT']
        cls.b,cls.c,_=load(cls.snapshot,ROOT/'data/g1-proof/gate-repair-v2-20261105/diagnostic.safetensors')
        cls.ref=Reference(cls.b,cls.c,pooler_kind='target-v2');cls.ref.freeze();groups(cls.b,cls.c,cls.ref)
        saved,_=initial_state();restore_model(ROOT/CHECKPOINT,cls.b,cls.c,cls.ref,saved['identities'])
        cls.c.enable_direct_roles();cls.c.requires_grad_(False);cls.b.requires_grad_(False)
        for p in cls.b.adapter_parameters('R'):p.requires_grad_(True)
        # All receiver/codec weights stay frozen; R and continuous writer states carry the VJP.
        cls.reference_id=cls.ref.fingerprint()
        cls.train=load_release(ROOT/'data/training-collections-20261105/releases/train',expected_hash=TRAIN_SHA)
        cls.row=next(cls.train.record(k) for k in sorted(cls.train.selected_ids)
                    if cls.train.record(k)['training_category']=='writer' and cls.train.record(k)['operation']=='Noul')

    def assessed(self):
        import torch
        logits=torch.full((5,),-30.,device='cuda');logits[0]=30.
        return patch.object(self.c.status,'forward',return_value=logits)

    def test_actual_native_return_and_prefill_source(self):
        for operation in ('Noul','Choice','Score'):
            row=next(self.train.record(k) for k in sorted(self.train.selected_ids)
                     if self.train.record(k)['training_category']=='writer' and self.train.record(k)['operation']==operation)
            self.assertTrue(self.native_return(row))

    def native_return(self,row):
        import torch
        from qwen_one_two.training.producer import EpisodeProducer
        from qwen_one_two.codec.session import Session
        from qwen_one_two.codec.layout import DIRECT
        from qwen_one_two.codec.schema import Kind
        from qwen_one_two.g1_evidence import frozen_check
        token=self.b.tokenizer.encode(row['supervision']['continuations'][0],add_special_tokens=False)[0]
        # Engineering stop: one actual native token after the real return. It
        # is not a semantically approved continuation/training example.
        def continuation(s):
            if s.runtime.phase==0:s.close()
            s.native([token]);s.runtime.terminate('Completed',emit_end=False)
            return token
        producer=EpisodeProducer(self.b,self.c,self.train,packet_profile=DIRECT)
        with torch.no_grad(),self.assessed(),patch.object(Session,'native_step',continuation):
            s,losses,exposure=producer(row,'components',1)
        report=s.episode_report
        self.assertNotIn('failure',report);self.assertEqual(s.runtime.consumptions,1)
        self.assertEqual(report['offline_targets_prepared_before_session'],0)
        self.assertEqual(report['native_output_tokens'],[token])
        self.assertNotIn('native',losses);self.assertNotIn('noul',losses)
        self.assertFalse(report['calls'][0]['semantic_targets_applied']);self.assertFalse(exposure.gold_return)
        source=s.runtime.registry.get(s.native_admission.source,Kind.SOURCE)
        self.assertEqual(tuple(source.offsets),tuple(report['source_offsets']))
        self.assertEqual(tuple(source.token_ids),tuple(report['source_token_ids']))
        s.native_admission.validate(s,losses)
        self.assertEqual(s.prefills,1);self.assertEqual(s.state.lineage,1)
        self.assertEqual(s.runtime.position,s.state.position);self.assertEqual(s.summary()['held_positions'],0)
        self.assertTrue(frozen_check(self.b,self.snapshot)['pass']);self.assertEqual(self.reference_id,self.ref.fingerprint())
        print({'fixture':'existing writer; actual returned scalar, no gold return',
               'forcing':['guided operation/full-source/no-attributes controls','Assessed status','one existing continuation token','budget512'],
               'semantic_qualification':False,'optimizer_updates':0},flush=True)
        return True

    def test_native_loss_through_actual_return_receiver_and_writer(self):
        import torch
        from qwen_one_two.training.producer import EpisodeProducer,native_plan
        from qwen_one_two.training.clean import tokenize_excerpt
        from qwen_one_two.training.losses import ExampleLosses
        from qwen_one_two.codec.records import Source
        from qwen_one_two.codec.runtime import Runtime
        from qwen_one_two.codec.session import Session
        from qwen_one_two.codec.layout import DIRECT
        from qwen_one_two.codec.schema import Kind,Role
        from qwen_one_two.g1_evidence import frozen_check
        plan=native_plan(self.train,self.row,'components',1)
        ids,selected,offsets=tokenize_excerpt(self.b.tokenizer,plan.prompt.encode())
        rt=Runtime('RD06-connected',len(ids),0,128,profile=DIRECT);s=Session(self.b,self.c,rt)
        h=s.prefill(ids);source=rt.registry.publish(Kind.SOURCE,Source(plan.prompt.encode(),tuple(ids[i] for i in selected),offsets,h[0,selected]))
        s.start_budget(512);producer=EpisodeProducer(self.b,self.c,self.train,packet_profile=DIRECT)
        evaluated=[];inputs=[];emissions=[];evaluate=s.evaluate;emit=self.c.emit
        def observe_emit(feature,role,count,attribute=None):
            result=emit(feature,role,count,attribute)
            if role==Role.QUESTION:emissions.append(result)
            return result
        def observe(packet):
            result=evaluate(packet);inputs.append(result[0]);evaluated.append(result[1]);return result
        with self.assessed(),patch.object(s,'evaluate',side_effect=observe),patch.object(self.c,'emit',side_effect=observe_emit), \
             patch.object(self.b,'frozen_source',side_effect=AssertionError('offline encoding inside live Session')):
            self.assertTrue(producer._guided_call(s,source,plan.calls[0],ExampleLosses(),{'calls':[]},1024))
        s.close();token=self.b.tokenizer.encode(self.row['supervision']['continuations'][0],add_special_tokens=False)[0]
        losses,_=s.native([token],True);raw=evaluated[0]['raw']
        # Separate the return -> evaluator -> writer VJP from the direct causal
        # cache route; freezing D must preserve this input derivative.
        signal=torch.autograd.grad(losses[0],raw,retain_graph=True)[0]
        self.assertTrue(bool(signal.isfinite().all()));self.assertGreater(float(signal.abs().sum()),0)
        wire_gradient=torch.autograd.grad(raw,inputs[0],retain_graph=True)[0]
        self.assertGreater(float(wire_gradient.norm()),0)
        self.assertTrue(emissions)
        owners=[emissions[0]]+list(self.b.adapter_parameters('R'))
        gradients=torch.autograd.grad(raw,owners,grad_outputs=signal,allow_unused=True)
        self.assertTrue(gradients[0] is not None and bool(gradients[0].isfinite().all()) and float(gradients[0].norm())>0)
        self.assertTrue(any(g is not None and bool(g.isfinite().all()) and float(g.norm())>0 for g in gradients[1:]))
        self.assertTrue(all(not p.requires_grad and p.grad is None for p in self.b.adapter_parameters('D')))
        self.assertTrue(all(not p.requires_grad and p.grad is None for p in self.c.parameters()))
        self.assertTrue(frozen_check(self.b,self.snapshot)['pass']);self.assertEqual(self.reference_id,self.ref.fingerprint())
        rt.terminate('Completed',emit_end=False)
        self.assertEqual(s.prefills,1);self.assertEqual(s.state.lineage,1);self.assertEqual(rt.position,s.state.position)
        print({'fixture':'engineering return-specific VJP, not approved semantic continuation',
               'forcing':['guided controls','Assessed status','existing token target','budget512'],
               'receiver_frozen':True,'optimizer_updates':0},flush=True)


if __name__=='__main__':unittest.main()
