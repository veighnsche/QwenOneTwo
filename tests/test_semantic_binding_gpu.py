"""Engineering binding/gradient fixtures, not qualified reference semantics."""
import os
import unittest
from dataclasses import replace
from types import SimpleNamespace
import torch
from torch.nn.attention import sdpa_kernel,SDPBackend
from qwen_one_two.g1_proof import configure
from qwen_one_two.codec.qualify import load
from qwen_one_two.codec.schema import Kind,Port
from qwen_one_two.codec.records import Source,Semantic,Span
from qwen_one_two.codec.session import Session
from qwen_one_two.training.reference import Reference
from qwen_one_two.training.release import load_release
from qwen_one_two.training.producer import native_plan,EpisodeProducer,PrerequisiteError
from qwen_one_two.training.binding import request_targets,compare_request,bind_labels,formation_losses
from qwen_one_two.training.losses import ExampleLosses


class SemanticBindingGPU(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        configure();cls.math=sdpa_kernel(SDPBackend.MATH);cls.math.__enter__()
        cls.b,cls.c,_=load(os.environ['QOT_QUALIFY_SNAPSHOT'],'data/g1-proof/gate-repair-v2-20261105/diagnostic.safetensors')
        cls.ref=Reference(cls.b,cls.c);cls.ref.freeze()
        cls.release=load_release('data/training-collections-20261105/releases/train',
            expected_hash='c70d4f44135a34c00e7fa0348833b1ec56613def15f5753231589492ed35478f')

    @classmethod
    def tearDownClass(cls):
        del cls.ref,cls.b,cls.c,cls.release
        cls.math.__exit__(None,None,None);torch.cuda.empty_cache()

    def fixture(self,key='train-writer-0004'):
        from qwen_one_two.training.clean import tokenize_excerpt
        from qwen_one_two.codec.runtime import Runtime
        row=self.release.record(key)
        plan=native_plan(self.release,row,'components',1)
        ids,selected,offsets=tokenize_excerpt(self.b.tokenizer,plan.prompt.encode())
        rt=Runtime('binding-fixture',len(ids),0,128);s=Session(self.b,self.c,rt)
        h=s.prefill(ids)
        source=rt.registry.publish(Kind.SOURCE,Source(plan.prompt.encode(),tuple(ids[i] for i in selected),offsets,h[0,selected]))
        s.start_budget(512)
        targets=request_targets(self.ref,self.release,row,plan.prompt.encode(),source_ref=source,source_record=rt.registry.get(source))
        # Reference-equal semantic payloads are prescribed for comparison tests;
        # they are never substituted into the native producer's actual writer.
        leaf=targets.payloads[0].clone().requires_grad_()
        writer=s.begin_writer(Port.NOUL if row['operation']=='Noul' else Port.CHOICE)
        q=rt.publish(Kind.QUESTION,Semantic(leaf));t=rt.publish(Kind.TARGET,Semantic(targets.payloads[1]))
        candidates=tuple(rt.publish(Kind.ALTERNATIVE,Semantic(payload)) for role,payload in zip(targets.names,targets.payloads) if role=='alternative')
        n=len(selected);evidence=tuple(Span(source,i,min(32,n-i)) for i in range(0,n,32))
        writer.draft=replace(writer.draft,question=q,target=t,evidence=evidence,candidates=candidates)
        s.step(writer.tick_vector(s.hidden[0,-1])[None],'R','fixture_finish',False)
        packet=rt.seal(q,t,evidence,candidates=candidates)
        return s,packet,targets,leaf

    def test_exact_content_compares_but_unqualified_reference_never_mints_labels(self):
        with torch.no_grad():s,packet,targets,_=self.fixture()
        matches,reason=compare_request(s.runtime.registry,packet.snapshot,targets)
        self.assertTrue(matches,reason)
        labels,binding=bind_labels(self.ref,None,s.runtime.registry,packet.snapshot,targets,runtime=s.runtime)
        self.assertIsNone(labels);self.assertFalse(binding['supported'])
        self.assertIn('qualification',binding['reason'])

    def test_matching_ids_and_nearby_slots_are_not_semantic_approval(self):
        with torch.no_grad():s,packet,targets,_=self.fixture()
        reg=s.runtime.registry;spec=reg.get(packet.snapshot,Kind.SNAPSHOT)
        q=reg.get(spec.question);changed=q.slots.clone();changed[0,0]+=.001
        ref=reg.publish(Kind.QUESTION,replace(q,slots=changed),previous=spec.question)
        snap=reg.publish(Kind.SNAPSHOT,replace(spec,question=ref))
        self.assertFalse(compare_request(reg,snap,targets)[0])
        # Same text with a different source identity/state lineage also rejects.
        source=reg.get(spec.evidence[0].source)
        other=reg.publish(Kind.SOURCE,source)
        snap=reg.publish(Kind.SNAPSHOT,replace(spec,evidence=tuple(replace(x,source=other) for x in spec.evidence)))
        self.assertFalse(compare_request(reg,snap,targets)[0])
        snap=reg.publish(Kind.SNAPSHOT,replace(spec,evidence=spec.evidence[:-1]))
        self.assertFalse(compare_request(reg,snap,targets)[0])

    def test_candidate_permutation_preserves_stable_identity_bindings(self):
        from qwen_one_two.codec.records import Bank
        from qwen_one_two.training.binding import candidate_mapping
        with torch.no_grad():s,packet,targets,_=self.fixture('train-writer-0000')
        reg=s.runtime.registry;spec=reg.get(packet.snapshot,Kind.SNAPSHOT)
        original=candidate_mapping(reg,packet.snapshot,targets)
        bank=reg.get(spec.bank,Kind.BANK)
        permuted=reg.publish(Kind.BANK,Bank(tuple(reversed(bank.candidates))))
        snap=reg.publish(Kind.SNAPSHOT,replace(spec,bank=permuted))
        matches,reason=compare_request(reg,snap,targets)
        self.assertTrue(matches,reason)
        self.assertEqual(candidate_mapping(reg,snap,targets),original)

    def test_target_tensors_cannot_be_edited_into_an_approval(self):
        with torch.no_grad():s,packet,targets,_=self.fixture();targets.payloads[0][0,0]+=.001
        with self.assertRaisesRegex(ValueError,'target mutation'):
            compare_request(s.runtime.registry,packet.snapshot,targets)

    def test_frozen_auxiliary_decoder_has_input_gradient_and_no_live_owner_gradient(self):
        s,packet,targets,leaf=self.fixture()
        book=ExampleLosses();self.b.checkpointing=True
        try:
            formation_losses(self.ref,s.runtime.registry,packet.snapshot,targets,book)
            values=book.finish()
            self.assertEqual(set(values),{'alignment','reconstruction'})
            params=list(self.b.adapter_parameters('R'))+list(self.b.adapter_parameters('D'))
            gradients=torch.autograd.grad(values['reconstruction'],[leaf]+params,allow_unused=True)
        finally:self.b.checkpointing=False
        self.assertTrue(bool(gradients[0].isfinite().all()));self.assertGreater(gradients[0].norm().item(),0)
        self.assertTrue(all(x is None for x in gradients[1:]))
        self.assertTrue(all(p.grad is None and not p.requires_grad for p in self.ref.parameters()))
        print({'fixture':'existing train-writer-0004; untrained frozen reference; prescribed exact content',
               'formation_components':sorted(values),'input_gradient_norm':gradients[0].norm().item(),
               'live_adapter_gradients':0,'optimizer_updates':0},flush=True)

    def test_unqualified_reference_and_diagnostic_accumulation_reject(self):
        from qwen_one_two.training.episodes import collect_session
        from qwen_one_two.training.inputs import Exposure
        with self.assertRaisesRegex(PrerequisiteError,'qualification'):
            EpisodeProducer(self.b,self.c,self.release,reference=self.ref)
        with self.assertRaisesRegex(ValueError,'qualification'):
            self.ref.initialize_live(self.c)
        with self.assertRaisesRegex(ValueError,'diagnostic reference'):
            collect_session(None,SimpleNamespace(episode_report={'diagnostic_reference':True}),{},Exposure('native_writer'))

    def test_continuation_needs_the_actual_value_selection_and_snapshot(self):
        """Typed mechanical return fixtures; not observed model judgments."""
        from qwen_one_two.codec.fixtures import base
        from qwen_one_two.codec.runtime import Runtime
        from qwen_one_two.codec.records import Result
        from qwen_one_two.codec.schema import Status
        from qwen_one_two.training.episodes import DecisionLabels
        from qwen_one_two.training.binding import compatible_return
        for port in (Port.NOUL,Port.SCORE,Port.CHOICE):
            rt=Runtime('return-fixture-'+str(port),10,512,128);rt.startup()
            packet,_,_,candidates,*_=base(rt,port,evidence=False)
            reg=rt.registry
            labels=DecisionLabels(packet.snapshot,Status.ASSESSED,noul=1.,score_interval=(.3,.3),
                                  choice=(candidates[0],) if candidates else ())
            def result(value):
                if port==Port.CHOICE:
                    return reg.publish(Kind.RESULT,Result(packet.snapshot,Status.ASSESSED,
                        selected=candidates[value],selected_index=value))
                return reg.publish(Kind.RESULT,Result(packet.snapshot,Status.ASSESSED,scalar=torch.tensor(value,device='cuda')))
            good=result(0 if port==Port.CHOICE else 1. if port==Port.NOUL else .3)
            bad=result(1 if port==Port.CHOICE else .9 if port==Port.NOUL else .31)
            self.assertTrue(compatible_return(reg,good,labels,None))
            self.assertFalse(compatible_return(reg,bad,labels,None))
            self.assertFalse(compatible_return(reg,good,replace(labels,snapshot=None),None))
            if candidates:
                self.assertFalse(compatible_return(reg,good,replace(labels,choice=candidates),None))


if __name__=='__main__':unittest.main()
