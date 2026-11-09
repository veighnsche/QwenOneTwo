"""Deferred PERF-01. Explicit 3090 opt-in before importing any tensor library."""
import os
import gc
import json
import random
import statistics
import tempfile
import time
from pathlib import Path
import unittest


class PackedFingerprintGPU(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if os.environ.get('QOT_RECEIVER_PERFORMANCE_GPU')!='1':
            raise unittest.SkipTest('GPU hold: PERF-01 not executed')
        from qwen_one_two.g1_device import PRIMARY
        if os.environ.get('CUDA_VISIBLE_DEVICES')!=PRIMARY:raise ValueError('exact 3090 UUID required')
        from qwen_one_two.g1_proof import configure
        configure()

    def test_packed_hash_exact_for_mixed_shapes_and_data_mutation(self):
        import torch
        from qwen_one_two.training.reference import state_fingerprint
        from qwen_one_two.training.host_metrics import fingerprint_transfers,Recorder
        state={'fp32':torch.tensor([0.,-0.,float('nan'),float('inf')],device='cuda'),
               'bf16':torch.arange(17,device='cuda',dtype=torch.bfloat16),
               'noncontiguous':torch.arange(24,device='cuda',dtype=torch.int64).reshape(4,6).T,
               'empty':torch.empty((0,3),device='cuda'),'scalar':torch.tensor(3.,device='cuda')}
        old=Recorder()
        with old.activate():expected=state_fingerprint(state)
        for limit in (1,32,32*1024*1024):
            new=Recorder()
            with fingerprint_transfers(limit),new.activate():actual=state_fingerprint(state)
            self.assertEqual(actual,expected)
            self.assertEqual(old.counts['audit_requested_bytes'],new.counts['audit_requested_bytes'])
            if limit==32*1024*1024:self.assertEqual(new.counts['audit_d2h_calls'],1)
        version=state['fp32']._version
        state['fp32'].data[0]=.25  # Deliberately bypasses version counters.
        self.assertEqual(version,state['fp32']._version)
        with fingerprint_transfers(32*1024*1024):changed=state_fingerprint(state)
        self.assertNotEqual(changed,expected)
        self.assertEqual(changed,state_fingerprint(state))


class RetainedReceiverGPU(unittest.TestCase):
    """PERF-02/03: twelve discarded diagnostic updates, never RD-08 fitting.

    Both interfaces, three alternating cold-window pairs per interface. The
    same eight existing reviewed assessed Noul/Choice/Score presentations are
    used each time. Restore parent0188 before every window; compare component
    gradients, updated masters/moments and transactional resume exactly. Timings
    exclude setup and extra gradient-hash/checkpoint inspection. No retry or OOM
    fallback. Full-fit memory/steady-state throughput are separate limitations.
    """
    @classmethod
    def setUpClass(cls):
        if os.environ.get('QOT_RECEIVER_PERFORMANCE_GPU')!='1':
            raise unittest.SkipTest('GPU hold: PERF-02/03 not executed')
        from qwen_one_two.g1_device import PRIMARY
        if os.environ.get('CUDA_VISIBLE_DEVICES')!=PRIMARY:raise ValueError('exact 3090 UUID required')
        from qwen_one_two.training.two_gpu import ROOT
        cls.output=Path(os.environ['QOT_PERF_OUTPUT']).resolve()
        if not cls.output.is_relative_to(ROOT/'data') or cls.output.exists():
            raise ValueError('fresh local performance evidence output required')
        cls.output.parent.mkdir(parents=True,exist_ok=True)
        from tests.test_joint_receiver_gpu import JointReceiverGPU
        JointReceiverGPU.setUpClass.__func__(cls)

    def run_window(self,profile,execution):
        import torch
        from tests.test_joint_receiver_gpu import JointReceiverGPU
        from qwen_one_two.training.receiver_learning import ReceiverProducer,extend_profile_window
        from qwen_one_two.training.receiver_performance import execution_scope
        from qwen_one_two.training.episodes import collect_session
        from qwen_one_two.training.reference import state_fingerprint
        from qwen_one_two.training.checkpoint import bundle_tensors,save,resume
        from qwen_one_two.training.inputs import digest
        from qwen_one_two.training.two_gpu import ROOT
        JointReceiverGPU.setUp(self)
        w=extend_profile_window(self.w,self.owned,self.c,profile)
        rows=[next(self.train.record(k) for k in sorted(self.train.selected_ids)
            if self.train.record(k)['training_category']=='receiver' and self.train.record(k)['operation']==op
            and self.train.record(k)['supervision']['result']['status']=='Assessed') for op in ('Noul','Choice','Score')]
        with execution_scope(self.b,execution):
            def protected():
                return state_fingerprint({**{n:p for n,p in self.b.named_parameters() if n.endswith(('.R_A','.R_B'))},
                    **{'codec.'+n:p for n,p in self.c.named_parameters() if not p.requires_grad}})
            frozen=protected()
            producer=ReceiverProducer(self.b,self.c,self.ref,self.train,input_profile=profile,cache_sources=True)
            torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
            cpu=time.process_time();wall=time.perf_counter();episodes=[]
            for i in range(8):
                s,losses,exposure=producer(rows[i%3]);report=s.episode_report
                self.assertNotIn('failure',report)
                episodes.append({k:report[k] for k in ('record','record_sha256','input_sha256','head','losses')})
                collect_session(w,s,losses,exposure);del s,losses
            torch.cuda.synchronize();seconds=time.perf_counter()-wall;cpu_seconds=time.process_time()-cpu
            counts=dict(w.counts)
            gradients=state_fingerprint({component+'/'+w.names[i]:g
                for component,buffers in w.buffers.items() for i,g in buffers.items()})
            cpu=time.process_time();wall=time.perf_counter();step=w.step();torch.cuda.synchronize()
            seconds+=time.perf_counter()-wall;cpu_seconds+=time.process_time()-cpu
            peak=torch.cuda.max_memory_allocated()
            weights=bundle_tensors(self.b,self.c,self.ref);masters=state_fingerprint(weights)
            moments=state_fingerprint({w.names[i]+'/'+k:v for i,p in enumerate(w.parameters)
                for k,v in w.optimizer.state.get(p,{}).items()})
            identities=dict(self.saved['identities'],protocol=digest(b'PERF-02-discarded-qualification-only'))
            sampler={'scope':'performance qualification only','cursor':8}
            with tempfile.TemporaryDirectory(dir=ROOT/'data') as temp:
                path=Path(temp)/'checkpoint'
                save(path,w,weights,identities,sampler,reference_frozen=True,reference_id=self.reference_id)
                cuda_rng=torch.cuda.get_rng_state(0).clone();python_rng=random.getstate()
                with torch.no_grad():w.parameters[0].flatten()[0].add_(.25)
                restored=resume(path,w,weights,identities)
                self.assertEqual(restored['sampler'],sampler)
                self.assertEqual(state_fingerprint(weights),masters)
                self.assertEqual(state_fingerprint({w.names[i]+'/'+k:v for i,p in enumerate(w.parameters)
                    for k,v in w.optimizer.state.get(p,{}).items()}),moments)
                self.assertTrue(torch.equal(cuda_rng,torch.cuda.get_rng_state(0)))
                self.assertEqual(python_rng,random.getstate())
            self.assertEqual(w.optimizer_updates,189)
            self.assertEqual(self.ref.fingerprint(),self.reference_id)
            self.assertEqual(protected(),frozen)
            self.assertTrue(all(p.grad is None for p in w.parameters))
            result={'profile':profile,'execution':execution,'seconds':seconds,'cpu_seconds':cpu_seconds,
                    'peak_cuda_allocated_bytes':peak,'episodes':episodes,'counts':counts,'gradients':gradients,
                    'masters':masters,'moments':moments,'step':step,'actual_diagnostic_updates':1,'resume_exact':True}
        del producer,w,weights;self.w.abort();gc.collect();torch.cuda.empty_cache()
        return result

    def test_both_interfaces_exact_updates_resume_and_three_timing_pairs(self):
        from qwen_one_two.training.receiver_performance import LEGACY,RETAINED
        from qwen_one_two.g1_evidence import frozen_check
        results=[]
        for profile in ('frozen-w-pooled-compact-v2','direct-role-states-v2'):
            for order in ((LEGACY,RETAINED),(RETAINED,LEGACY),(LEGACY,RETAINED)):
                pair={name:self.run_window(profile,name) for name in order}
                with self.output.open('a') as stream:stream.write(json.dumps(pair,allow_nan=False)+'\n')
                for key in ('episodes','counts','gradients','masters','moments','step'):
                    self.assertEqual(pair[LEGACY][key],pair[RETAINED][key],(profile,key))
                results.append(pair)
        self.assertTrue(frozen_check(self.b,self.snapshot)['pass'])
        for profile in ('frozen-w-pooled-compact-v2','direct-role-states-v2'):
            pairs=[p for p in results if p[LEGACY]['profile']==profile]
            self.assertTrue(all(p[RETAINED]['seconds']<p[LEGACY]['seconds'] for p in pairs),(profile,results))
            self.assertLess(statistics.median(p[RETAINED]['cpu_seconds'] for p in pairs),
                            statistics.median(p[LEGACY]['cpu_seconds'] for p in pairs))
