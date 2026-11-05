"""Real GPU episode/boundary proof using existing reviewed data, no fitting.

Diagnostic weights and prescribed control establish transport, not learned
invocation or correctness. No optimizer or data-generation service is imported.
"""
import os
import unittest
from unittest.mock import patch
import torch
from torch.nn.attention import sdpa_kernel,SDPBackend

from qwen_one_two.g1_proof import configure
from qwen_one_two.codec.qualify import load
from qwen_one_two.codec.session import Session
from qwen_one_two.codec.render import Packet
from qwen_one_two.training.release import load_release
from qwen_one_two.training.producer import EpisodeProducer


class ProducerGPU(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        configure()
        cls.math=sdpa_kernel(SDPBackend.MATH);cls.math.__enter__()
        cls.b,cls.c,_=load(os.environ['QOT_QUALIFY_SNAPSHOT'],
                          'data/g1-proof/gate-repair-v2-20261105/diagnostic.safetensors')
        cls.release=load_release('data/training-collections-20261105/releases/train',
            expected_hash='c70d4f44135a34c00e7fa0348833b1ec56613def15f5753231589492ed35478f')

    @classmethod
    def tearDownClass(cls):
        del cls.b,cls.c,cls.release
        cls.math.__exit__(None,None,None);torch.cuda.empty_cache()

    def test_text_and_serialized_annotations_reject_before_any_forward(self):
        session=Session(self.b,self.c,None)
        for value in ('Assess this question',{'question':'offline annotation'},b'{}'):
            with self.assertRaisesRegex(ValueError,'latent Packet'):
                session.evaluate(value)
        self.assertIsNone(session.hidden)

    def test_existing_data_invokes_Qev_with_connected_packet_in_same_cache(self):
        from qwen_one_two.training.losses import ExampleLosses
        from qwen_one_two.training.producer import native_plan
        from qwen_one_two.training.clean import tokenize_excerpt
        from qwen_one_two.codec.runtime import Runtime
        from qwen_one_two.codec.records import Source
        from qwen_one_two.codec.schema import Kind
        rows=self.release.sampler_rows()
        for operation in ('Noul','Choice','Score'):
            key=next(r['id'] for r in rows if r['category']=='writer' and r['operation']==operation)
            row=self.release.record(key)
            plan=native_plan(self.release,row,'components',1)
            ids,selected,offsets=tokenize_excerpt(self.b.tokenizer,plan.prompt.encode())
            rt=Runtime('proof-'+operation,len(ids),0,128)
            s=Session(self.b,self.c,rt)
            h=s.prefill(ids)
            source=rt.registry.publish(Kind.SOURCE,Source(plan.prompt.encode(),tuple(ids[i] for i in selected),offsets,h[0,selected]))
            s.start_budget(512)
            producer=EpisodeProducer(self.b,self.c,self.release)
            book=ExampleLosses();report={'calls':[]}
            before=[p.detach().clone() for p in self.b.adapter_parameters('D')]
            forwards=[];packet_inputs=[]
            forward=self.b.forward;encode=self.c.encode
            def track_forward(state,mode,embeds):
                forwards.append((state.position,mode,embeds))
                return forward(state,mode,embeds)
            def track_encode(packet):
                self.assertIsInstance(packet,Packet)
                value=encode(packet)
                if packet.snapshot is not None and packet.result is None:
                    packet_inputs.append(value)
                return value
            # Neither token embedding lookup nor text tokenization is permitted
            # while constructing/invoking/returning this actual latent call.
            # The preserved diagnostic checkpoint has an untrained attachment
            # decoder. Prescribe only child-count=0 for this positive transport
            # fixture; status/value/selection and all semantic slots remain the
            # actual live outputs. This is not a learned-Reason success claim.
            child_count=torch.full((9,),-30.,device='cuda');child_count[0]=30.
            with patch.object(self.c.heads['child_count'],'forward',return_value=child_count), \
                 patch.object(self.b,'forward',side_effect=track_forward), \
                 patch.object(self.c,'encode',side_effect=track_encode), \
                 patch.object(self.b,'embedding',side_effect=AssertionError('text lookup during latent call')), \
                 patch.object(self.b.tokenizer,'encode',side_effect=AssertionError('text tokenization during latent call')):
                delivered=producer._guided_call(s,source,plan.calls[0],book,report,512)
            self.assertTrue(delivered,report)
            self.assertEqual(len(report['calls']),1)
            self.assertEqual(rt.dispatched,1);self.assertEqual(rt.consumptions,1)
            self.assertEqual(s.prefills,1);self.assertEqual(s.state.lineage,1)
            decisions=[entry for entry in forwards if entry[1]=='D']
            self.assertEqual(len(decisions),1)
            actual=decisions[0][2]
            self.assertEqual(actual.shape[1],1024);self.assertEqual(actual.dtype,torch.float32)
            self.assertTrue(actual.is_cuda);self.assertTrue(actual.requires_grad)
            self.assertTrue(any(actual is value for value in packet_inputs))
            self.assertEqual(forwards[-1][1],'R')
            self.assertEqual(s.state.position,rt.position)
            # The actual return stays in the graph. A native loss after it has
            # a connected derivative to the exact tensor supplied to W+D.
            s.close()
            token=self.b.tokenizer.encode('Done',add_special_tokens=False)[0]
            losses,_=s.native([token],True)
            gradient=torch.autograd.grad(losses[0],actual)[0]
            self.assertTrue(bool(gradient.isfinite().all()))
            self.assertGreater(gradient.norm().item(),0)
            rt.debit('H',1,'chat_end');s.step(self.b.embedding([248046])[0],'R','chat_end',False)
            rt.terminate('Completed',emit_end=False)
            summary=s.summary()
            self.assertEqual(summary['total'],sum(summary[k] for k in ('P','B','A','H')))
            self.assertEqual(summary['held_positions'],0)
            self.assertTrue(all(torch.equal(a,b) for a,b in zip(before,self.b.adapter_parameters('D'))))
            print({'record':key,'operation':operation,'proof':'actual latent transport; prescribed control',
                   'diagnostic_fixture':'Reason attachment count prescribed zero; no parameter updates',
                   'summary':{k:summary[k] for k in ('P','B','A','H','total','prefills','lineage','dispatched','consumed')},
                   'packet_gradient_norm':gradient.norm().item(),'calls':report['calls']},flush=True)
            del losses,gradient,s,h,actual,packet_inputs,forwards,before,book
            torch.cuda.empty_cache()

    def test_complete_direct_producer_returns_driver_contract_without_optimizer(self):
        key=next(r['id'] for r in self.release.sampler_rows() if r['category']=='direct')
        producer=EpisodeProducer(self.b,self.c,self.release)
        with torch.no_grad():
            session,losses,exposure=producer(self.release.record(key),'components',1)
        self.assertEqual(session.runtime.terminal,'Completed',producer.last_report)
        self.assertEqual(session.prefills,1)
        self.assertEqual(session.state.position,session.runtime.position)
        self.assertIn('native',losses)
        self.assertTrue(exposure.native_teacher_forcing)
        self.assertEqual(session.runtime.dispatched,0)
        self.assertEqual(session.runtime.remaining[2],64-session.runtime.h)

    def test_untrained_reason_failure_is_a_failed_exposure_without_gold_repair(self):
        key=next(r['id'] for r in self.release.sampler_rows() if r['category']=='writer' and r['operation']=='Noul')
        producer=EpisodeProducer(self.b,self.c,self.release)
        with torch.no_grad():
            session,losses,exposure=producer(self.release.record(key),'components',1)
        self.assertEqual(session.runtime.terminal,'Failed')
        self.assertEqual(losses,{})
        self.assertEqual(session.runtime.dispatched,1)
        self.assertEqual(session.runtime.consumptions,0)
        self.assertEqual(session.runtime.position,session.state.position)
        self.assertEqual(session.runtime.summary()['held_positions'],0)
        self.assertFalse(exposure.gold_return)
        self.assertIn('compatible pointer',producer.last_report['failure'])

    def test_clean_receiver_and_reader_use_existing_records_not_text_invocation(self):
        from pathlib import Path
        from qwen_one_two.training.reference import Reference
        from qwen_one_two.training.manifest import DIAGNOSTIC_SHA
        from qwen_one_two.training.inputs import digest
        reference=Reference(self.b,self.c);reference.freeze()
        producer=EpisodeProducer(self.b,self.c,self.release,reference=reference,
            checkpoint_hash=DIAGNOSTIC_SHA,profile_hash=digest(Path('docs/codec-feedback-precision-v3.md').read_bytes()))
        samples=[next(r['id'] for r in self.release.sampler_rows() if r['category']==category)
                 for category in ('receiver','reader','decision_replay')]
        samples += [next(r['id'] for r in self.release.sampler_rows() if r['category']=='reader'
                         and self.release.record(r['id'])['operation']==operation
                         and self.release.record(r['id'])['labels']['status']==status)
                    for operation,status in (('Noul','Insufficient'),('Choice','NoFeasibleChoice'))]
        for key in samples:
            row=self.release.record(key)
            category=row['training_category']
            child_count=torch.full((9,),-30.,device='cuda');child_count[0]=30.
            with torch.no_grad(),patch.object(self.c.heads['child_count'],'forward',return_value=child_count):
                session,losses,exposure=producer(row,'components',1)
            self.assertEqual(session.runtime.terminal,'Completed',producer.last_report)
            self.assertEqual(session.prefills,1);self.assertEqual(session.state.lineage,1)
            self.assertEqual(session.state.position,session.runtime.position)
            self.assertEqual(session.runtime.dispatched,1);self.assertEqual(session.runtime.consumptions,1)
            self.assertEqual(session.runtime.summary()['held_positions'],0)
            self.assertEqual(exposure.gold_return,category=='reader')
            self.assertEqual(producer.last_report['text_tokens_for_invocation'],0)
            self.assertIn('native' if category=='reader' else 'status',losses)
            print({'record':key,'category':category,'qualification':'untrained frozen reference construction fixture',
                   'total':session.runtime.position,'loss_components':sorted(losses),'gold_return':exposure.gold_return},flush=True)

    def test_overlength_discards_losses_before_an_unexecuted_position_is_charged(self):
        from dataclasses import replace
        from qwen_one_two.training.producer import native_plan
        from qwen_one_two.training.clean import tokenize_excerpt
        key=next(r['id'] for r in self.release.sampler_rows() if r['category']=='direct')
        row=self.release.record(key);plan=native_plan(self.release,row,'components',1)
        ids,_,_=tokenize_excerpt(self.b.tokenizer,plan.prompt.encode())
        plan=replace(plan,limit=len(ids)+7)
        producer=EpisodeProducer(self.b,self.c,self.release)
        with torch.no_grad(),patch('qwen_one_two.training.producer.native_plan',return_value=plan):
            s,losses,_=producer(row,'components',1)
        self.assertEqual(s.runtime.terminal,'Failed')
        self.assertEqual(losses,{})
        self.assertEqual(s.runtime.position,s.state.position)
        self.assertIn('ceiling',producer.last_report['failure'])

    def test_free_window_allows_isolated_replay_but_rejects_forced_native_policy(self):
        from qwen_one_two.training.episodes import collect_session
        from qwen_one_two.training.inputs import Exposure
        class ScheduledExposure:
            phase='free_joint'
            def add_episode(self,*args,**kwargs):self.collected=True
        ids=self.b.tokenizer.encode('Replay boundary engineering fixture.',add_special_tokens=False)
        with torch.no_grad():
            s=Session.open(self.b,self.c,ids,128)
            s.runtime.terminate('Failed',usable=False,emit_end=False)
        window=ScheduledExposure()
        report=collect_session(window,s,{},Exposure('isolated_receiver',packet_origin='reviewed_component',decision_forcing=True))
        self.assertFalse(report['accepted']);self.assertTrue(window.collected)
        for branch in ('native_writer','joint_receiver'):
            with self.assertRaisesRegex(ValueError,'forced free-control'):
                collect_session(window,s,{},Exposure(branch,decision_forcing=True))

    def test_set_valued_reader_annotation_does_not_supervise_single_selection_return(self):
        from pathlib import Path
        from qwen_one_two.training.reference import Reference
        from qwen_one_two.training.manifest import DIAGNOSTIC_SHA
        from qwen_one_two.training.inputs import digest
        key=next(r['id'] for r in self.release.sampler_rows() if r['category']=='reader'
                 and len((self.release.record(r['id'])['labels'].get('payload') or {}).get('acceptable_selection_ids',[]))>1)
        reference=Reference(self.b,self.c);reference.freeze()
        producer=EpisodeProducer(self.b,self.c,self.release,reference=reference,
            checkpoint_hash=DIAGNOSTIC_SHA,profile_hash=digest(Path('docs/codec-feedback-precision-v3.md').read_bytes()))
        with torch.no_grad():s,losses,exposure=producer(self.release.record(key),'components',1)
        self.assertEqual(s.runtime.terminal,'Completed',producer.last_report)
        self.assertEqual(losses,{})
        self.assertIn('single bound return',producer.last_report['native_continuation_mask'])
        self.assertTrue(exposure.gold_return)


if __name__=='__main__':unittest.main()
