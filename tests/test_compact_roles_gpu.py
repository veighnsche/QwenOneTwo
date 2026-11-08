"""Opt-in v2 changed-interface GPU qualification; no optimizer steps."""
import gc,json,os,unittest,random,tempfile
from pathlib import Path
from qwen_one_two.codec.layout import COMPACT_DIRECT,COMPACT_POOLED,is_direct


def record(value):
    print(json.dumps(value,allow_nan=False),flush=True)
    if os.environ.get('QOT_RD07_EVIDENCE_DIR'):
        with (Path(os.environ['QOT_RD07_EVIDENCE_DIR'])/'metrics.jsonl').open('a') as f:f.write(json.dumps(value,allow_nan=False)+'\n')


class CompactRolesGPU(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if os.environ.get('QOT_COMPACT_GPU_QUALIFY')!='1':raise unittest.SkipTest('requires explicit v2 qualification')
        from tests.test_direct_roles_gpu import DirectRolesGPU
        DirectRolesGPU.setUpClass.__func__(cls)
        cls.c.enable_direct_roles()

    def vectors(self,n):
        from tests.test_direct_roles_gpu import DirectRolesGPU
        return DirectRolesGPU.vectors(self,n)

    def ids(self):
        return self.b.tokenizer.encode(self.b.tokenizer.apply_chat_template([{'role':'user','content':self.prompt}],
            tokenize=False,add_generation_prompt=True,enable_thinking=True),add_special_tokens=False)

    def packet(self,rt,port,evidence=()):
        from qwen_one_two.codec.records import Semantic
        from qwen_one_two.codec.schema import Kind,Port
        def semantic(n):return Semantic(self.vectors(n),profile=self.wire,lengths=(n,) if is_direct(self.wire) else ())
        q=rt.publish(Kind.QUESTION,semantic(49 if is_direct(self.wire) else 8))
        t=rt.publish(Kind.TARGET,semantic(23 if is_direct(self.wire) else 2))
        cs=tuple(rt.publish(Kind.ALTERNATIVE,semantic(n if is_direct(self.wire) else 4)) for n in (41,40)) if port==Port.CHOICE else ()
        sizes=(11,24,16,16,0) if is_direct(self.wire) else ()
        rubric=rt.publish(Kind.RUBRIC,Semantic(self.vectors(67 if sizes else 10),direction=1,
            profile=self.wire,lengths=sizes)) if port==Port.SCORE else None
        return rt.seal(q,t,evidence=evidence,candidates=cs,rubric=rubric)

    def test_a_existing_reviewed_inputs_and_joint_VJPs(self):
        import torch
        from qwen_one_two.training.receiver_learning import ReceiverProducer
        from qwen_one_two.training.release import load_release
        from qwen_one_two.training.receiver_plan import TRAIN_SHA
        from qwen_one_two.training.two_gpu import ROOT
        train=load_release(ROOT/'data/training-collections-20261105/releases/train',expected_hash=TRAIN_SHA)
        for profile in ('frozen-w-pooled-compact-v2','direct-role-states-v2'):
            producer=ReceiverProducer(self.b,self.c,self.ref,train,input_profile=profile,cache_sources=True)
            seen=set()
            for op in ('Noul','Choice','Score'):
                row=next(train.record(k) for k in sorted(train.selected_ids) if train.record(k)['training_category']=='receiver'
                         and train.record(k)['operation']==op and train.record(k)['supervision']['result']['status']=='Assessed')
                s,losses,_=producer(row);self.assertNotIn('failure',s.episode_report);self.assertIn('status',losses)
                owner=list(self.b.adapter_parameters('D'))
                if profile=='direct-role-states-v2':owner+=list(self.c.direct_role_projection.parameters())
                gradients=torch.autograd.grad(sum(losses.values()),owner,allow_unused=True)
                self.assertTrue(any(g is not None and float(g.norm())>0 for g in gradients))
                self.assertTrue(all(g is None or bool(g.isfinite().all()) for g in gradients))
                self.assertEqual(s.prefills,1);self.assertEqual(s.state.position,s.runtime.position)
                s.component_admission.validate(s.runtime)
                with torch.no_grad():
                    measured,_,_=producer(row,deliver=True)
                self.assertIn('behavior',measured.episode_report)
                self.assertIn('head',measured.episode_report)
                self.assertEqual(measured.prefills,1)
                self.assertEqual(measured.state.position,measured.runtime.position)
                seen.add(op);del s,losses,gradients,measured
            self.assertEqual(len(seen),3)
            record({'check':'v2-real-input-VJPs','profile':profile,'operations':sorted(seen),'optimizer_updates':0})
        self.assertTrue(all(not p.requires_grad and p.grad is None for p in self.b.adapter_parameters('R')))
        self.assertTrue(all(p.grad is None for p in self.ref.parameters()))

    def test_b_all_statuses_maximum_reason_and_256_boundary(self):
        import torch
        from qwen_one_two.codec.runtime import Runtime
        from qwen_one_two.codec.records import Reason,Source,Span
        from qwen_one_two.codec.schema import Port,Status,Kind,Role,decode_atoms,ValueKind,Invalid
        for self.wire in (COMPACT_POOLED,COMPACT_DIRECT):
            count=0
            for port in (Port.NOUL,Port.CHOICE,Port.SCORE):
                for status in tuple(Status)[1:]:
                    if status==Status.NO_FEASIBLE_CHOICE and port!=Port.CHOICE:continue
                    rt=Runtime('v2-all-status',16,1024,64,profile=self.wire);rt.startup();rt.begin(port)
                    p=self.packet(rt,port);wire=self.c.encode(p);self.assertEqual(len(wire),len(p))
                    rt.semantic_positions(len(p))
                    r=rt.prepare_result(p.snapshot,status,scalar=torch.tensor(.25,device='cuda') if status==Status.ASSESSED and port!=Port.CHOICE else None,
                        selected_index=1 if status==Status.ASSESSED and port==Port.CHOICE else None,
                        reason=Reason(self.vectors(4),profile=self.wire) if status!=Status.ASSESSED else None)
                    self.assertEqual(len(r),3 if status==Status.ASSESSED else 7)
                    self.assertTrue(bool(self.c.encode(r).isfinite().all()))
                    self.assertEqual(r.handles[:len(p.handles)],p.handles)
                    if status==Status.ASSESSED and port==Port.CHOICE:
                        h=decode_atoms(ValueKind.REFERENCE,(r.positions[1].payload,))
                        self.assertEqual(r.handles[h],rt.registry.get(r.result).selected)
                    rt.semantic_positions(len(r),True);self.assertTrue(rt.consumed(p.snapshot));self.assertEqual(rt.summary()['held_positions'],0);count+=1
            self.assertEqual(count,13)
            record({'check':'v2-all-status','wire_profile':self.wire,'cells':count})
        self.wire=COMPACT_DIRECT
        for extra in (0,1):
            rt=Runtime('v2-boundary',16,1024,64,profile=self.wire);rt.startup()
            body=(self.prompt.encode()*100)[:96]
            source=rt.registry.publish(Kind.SOURCE,Source(body,tuple(range(96)),tuple((i,i+1) for i in range(96)),self.vectors(96)))
            rt.begin(Port.CHOICE);refs=rt.registry.refs()
            if extra:
                with self.assertRaises(Invalid):self.packet(rt,Port.CHOICE,(Span(source,0,32),Span(source,32,32),Span(source,64,32)))
                self.assertIsNone(rt.call.snapshot);self.assertEqual(rt.dispatched,0)
            else:
                p=self.packet(rt,Port.CHOICE,(Span(source,0,32),Span(source,32,32),Span(source,64,31)))
                self.assertEqual(len(p),256);self.assertEqual(len(self.c.encode(p)),256)
                rt.semantic_positions(256)
                r=rt.prepare_result(p.snapshot,Status.INSUFFICIENT,reason=Reason(self.vectors(4),(Span(source,0,31),),self.wire))
                self.assertEqual(len(r),39);self.assertEqual(len(self.c.encode(r)),39)
            rt.terminate('Cancelled',emit_end=False)
        record({'check':'v2-bounds','accepted_request':256,'rejected_request':257,'maximum_return':39})

    def test_c_FP64_projection_reference(self):
        from tests.test_redesign_qualification_gpu import RedesignQualificationGPU
        RedesignQualificationGPU.test_projection_forward_and_VJP_FP64_reference(self)

    def test_d_cancellation(self):
        for self.wire in (COMPACT_POOLED,COMPACT_DIRECT):self.cancellation()

    def cancellation(self):
        import torch
        from qwen_one_two.codec.schema import Port,Kind,Status,Invalid
        from qwen_one_two.codec.records import Semantic,Source,Span,Reason
        from qwen_one_two.codec.runtime import Runtime
        from qwen_one_two.codec.session import Session
        DIRECT=self.wire
        with torch.no_grad():
            for where in ('rpc_before_request','callback_before_request','callback_in_request','callback_before_return'):
                with self.subTest(where=where):
                    s=Session.open(self.b,self.c,self.ids(),64,override=512,profile=DIRECT);rt=s.runtime;rt.begin(Port.NOUL)
                    packet=self.packet(rt,Port.NOUL);before=rt.position
                    if where=='rpc_before_request':s.native_step(cancelled=True)
                    elif where=='callback_before_request':
                        s.cancelled=lambda:True
                        with self.assertRaises(InterruptedError):s.evaluate(packet)
                    elif where=='callback_in_request':
                        s.cancelled=lambda:s.state.position>before
                        with self.assertRaises(InterruptedError):s.evaluate(packet)
                    else:
                        _,heads,_=s.evaluate(packet)
                        result=rt.prepare_result(packet.snapshot,Status.ASSESSED,scalar=heads['value'])
                        before=rt.position;s.cancelled=lambda:True
                        with self.assertRaises(InterruptedError):s.deliver(result)
                    self.assertEqual(rt.terminal,'Cancelled');self.assertEqual(rt.position,s.state.position)
                    self.assertEqual(rt.summary()['held_positions'],0)
                    self.assertIsNone(rt.prepare_result(packet.snapshot,Status.ASSESSED,scalar=torch.tensor(.5,device='cuda')))
                    record({'check':'RD07-GPU-CANCELLATION-ENVELOPES','cancellation':where,'summary':s.summary()})

    def trace(self,port,profile):
        import torch
        from qwen_one_two.codec.session import Session
        from qwen_one_two.codec.schema import Status,Port
        DIRECT=self.wire
        self.b.checkpointing=profile=='checkpoint'
        self.b.set_precision(torch.float32 if profile=='fp32' else torch.bfloat16)
        self.b.recurrence_backend='eager' if profile=='fp32' else 'graph-v1'
        with torch.set_grad_enabled(profile!='serving'):
            s=Session.open(self.b,self.c,self.ids(),1024,override=512,profile=DIRECT,run='RD07-paired-trace')
            rt=s.runtime;rt.begin(port);packet=self.packet(rt,port);wire,evaluated,_=s.evaluate(packet)
            scalar=None if port==Port.CHOICE else evaluated['value']
            returned=rt.prepare_result(packet.snapshot,Status.ASSESSED,scalar=scalar,
                                       selected_index=0 if port==Port.CHOICE else None)
            self.assertTrue(s.deliver(returned));s.close()
            token=self.b.tokenizer.encode(self.prompt,add_special_tokens=False)[0]
            # Explicit prescribed existing-token tail, not sampled/new data.
            remaining=1023-rt.position
            self.assertGreaterEqual(remaining,0)
            while remaining:
                count=min(32,remaining)
                for _ in range(count):rt.native()
                s.step(self.b.embedding([token]*count)[0],'R','qualification_tail',False)
                remaining-=count
            logits,allowed=self.b.ordinary_logits(s.hidden[0,-1],1)
            target=(allowed==token).nonzero().flatten()
            native=torch.nn.functional.cross_entropy(logits[None],target)
            if port==Port.NOUL and profile!='serving':
                signal_return=torch.autograd.grad(native,evaluated['raw'],retain_graph=True)[0]
                self.assertTrue(bool(signal_return.isfinite().all()));self.assertGreater(float(signal_return.abs()),0)
            signal=evaluated['pointer_logits'][0] if port==Port.CHOICE else evaluated['raw']
            loss=evaluated['status_logits'][0]+.1*signal+.1*native
            vectors={}
            if profile!='serving':
                groups={'D':list(self.b.adapter_parameters('D')),**({'projection':list(self.c.direct_role_projection.parameters())} if self.wire==COMPACT_DIRECT else {})}
                params=[p for values in groups.values() for p in values]
                gradients=torch.autograd.grad(loss,params,allow_unused=True)
                offset=0
                for name,values in groups.items():vectors[name]=tuple(g.detach() if g is not None else None for g in gradients[offset:offset+len(values)]);offset+=len(values)
            rt.debit('H',1,'chat_end');s.step(self.b.embedding([248046])[0],'R','end',True);rt.terminate('Completed',emit_end=False)
            self.assertEqual(rt.position,1024);self.assertEqual(rt.position,s.state.position)
            self.assertEqual(s.prefills,1);self.assertEqual(s.state.lineage,1);self.assertEqual(s.summary()['held_positions'],0)
            call={'status':int(Status.ASSESSED),'predicted_status':int(evaluated['status_logits'].argmax())+1,
                  'q':len(packet),'r':len(returned),'handles':len(returned.handles),'selected':0 if port==Port.CHOICE else None,
                  'scalar':None if scalar is None else float(scalar.detach())}
            result={'summary':s.summary(),'snapshots':s.snapshots,'vectors':vectors,'logits':[logits.detach()],
                    'calls':[call],'budget_raw':float(s.raw_budget.detach())}
        record({'event':'long_trace','wire_profile':self.wire,'port':int(port),'profile':profile,'positions':rt.position,'q':len(packet),'r':len(returned),
                'prefills':s.prefills,'lineage':s.state.lineage,'optimizer_updates':0,'forcing':'existing prompt/token, direct engineering role states, Assessed, Choice index0, budget512, 1024 total'})
        return result

    def test_z_both_formats_1024_functional_checkpoint_serving_FP32(self):
        import torch
        from qwen_one_two.codec.qualify import compare
        from qwen_one_two.codec.schema import Port
        original_R=[p.requires_grad for p in self.b.adapter_parameters('R')]
        backend=self.b.recurrence_backend
        # Keep FP32 adapter masters out of the base-only precision conversion.
        for p in self.b.adapter_parameters('R'):p.requires_grad_(True)
        try:
            for self.wire in (COMPACT_POOLED,COMPACT_DIRECT):
              for port in (Port.NOUL,Port.CHOICE,Port.SCORE):
                  base=self.trace(port,'functional')
                  for profile in ('checkpoint','serving','fp32'):
                      test=self.trace(port,profile);metrics=compare(test,base,profile=='fp32')
                      record({'check':'RD07-GPU-LONG-PARITY','wire_profile':self.wire,'port':int(port),'profile':profile,'comparison':metrics})
                      self.assertTrue(metrics['pass'],metrics)
                      del test;gc.collect();torch.cuda.empty_cache()
                  del base;gc.collect();torch.cuda.empty_cache()
        finally:
            self.b.set_precision(torch.bfloat16);self.b.checkpointing=False;self.b.recurrence_backend=backend
            for p,value in zip(self.b.adapter_parameters('R'),original_R):p.requires_grad_(value)



from tests.test_joint_receiver_gpu import JointReceiverGPU
class CompactResumeGPU(unittest.TestCase):
    @classmethod
    def setUpClass(cls):JointReceiverGPU.setUpClass.__func__(cls)
    def setUp(self):JointReceiverGPU.setUp(self)
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
        window=extend_profile_window(self.w,self.owned,self.c,'direct-role-states-v2')
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


    def test_eight_example_v2_joint_accumulation(self):
        import torch
        from qwen_one_two.training.receiver_learning import ReceiverProducer,extend_profile_window
        from qwen_one_two.training.episodes import collect_session
        rows=[next(self.train.record(k) for k in sorted(self.train.selected_ids)
            if self.train.record(k)['training_category']=='receiver' and self.train.record(k)['operation']==op
            and self.train.record(k)['supervision']['result']['status']=='Assessed') for op in ('Noul','Choice','Score')]
        for profile in ('frozen-w-pooled-compact-v2','direct-role-states-v2'):
            w=extend_profile_window(self.w,self.owned,self.c,profile)
            producer=ReceiverProducer(self.b,self.c,self.ref,self.train,input_profile=profile,cache_sources=True)
            for i in range(8):
                s,losses,exposure=producer(rows[i%3]);self.assertNotIn('failure',s.episode_report)
                collect_session(w,s,losses,exposure);del s,losses
            self.assertEqual(w.counts['status'],8)
            expected={}
            for component,buffers in w.buffers.items():
                for i,g in buffers.items():
                    weight=w.coefficients[component]/w.counts[component]
                    if i in expected:expected[i].add_(g,alpha=weight)
                    else:expected[i]=g.mul(weight)
            w.combine()
            self.assertTrue(all(torch.equal(w.parameters[i].grad,value) for i,value in expected.items()))
            record({'check':'v2-eight-joint-VJPs','input_profile':profile,'eligible':dict(w.counts),
                    'owned_parameters':len(w.names),'optimizer_updates':0})
            self.assertEqual(w.optimizer_updates,188);w.abort()

    def test_cached_component_survives_exact_checkpoint_restore(self):
        import torch
        from qwen_one_two.training.receiver_learning import ReceiverProducer
        from qwen_one_two.training.checkpoint import save,bundle_tensors,restore_model
        from qwen_one_two.training.two_gpu import ROOT
        producer=ReceiverProducer(self.b,self.c,self.ref,self.train,input_profile='frozen-w-pooled-compact-v2',cache_sources=True)
        with torch.no_grad():before,_,_=producer(self.row)
        self.assertNotIn('failure',before.episode_report)
        heads=before.episode_report['head'];losses=before.episode_report['losses'];stamp=producer.encoder.reference_stamp
        with tempfile.TemporaryDirectory(dir=ROOT/'data') as temp:
            path=Path(temp)/'checkpoint'
            save(path,self.w,bundle_tensors(self.b,self.c,self.ref),self.saved['identities'],{'fixture':'exact restore, no update'},
                reference_frozen=True,reference_id=self.reference_id)
            restore_model(path,self.b,self.c,self.ref,self.saved['identities'],window=self.w)
            self.assertNotEqual(stamp,tuple((id(p),p._version) for p in self.ref.parameters()))
            with torch.no_grad():after,_,_=producer(self.row)
        self.assertNotIn('failure',after.episode_report);self.assertEqual(heads,after.episode_report['head'])
        self.assertEqual(losses,after.episode_report['losses']);self.assertGreater(producer.encoder.source_cache_hits,0)
        self.assertEqual(producer.encoder.reference_stamp,tuple((id(p),p._version) for p in self.ref.parameters()))
        self.assertEqual(self.reference_id,self.ref.fingerprint());self.assertEqual(self.w.optimizer_updates,188)
        record({'check':'v2-cached-input-restore','heads_and_losses_exact':True,'source_cache_hits':producer.encoder.source_cache_hits,
                'reference_unchanged':True,'optimizer_updates':0})
