"""Bounded read-only 3090 comparison: two replicas versus two shared lanes.

All modes use the same existing cases, serial per-lane warmups, private CUDA
streams and retained graph shapes. No timing cutoff, training or service control.
"""
import argparse
from contextlib import ExitStack
import json
import multiprocessing as mp
from pathlib import Path
import queue
import threading
import time
import traceback
import os
from qwen_one_two.training.inputs import check,canonical,digest
from qwen_one_two.training.two_gpu import ROOT,plan,validate_case,DEV_SHA,RECEIVER,REFERENCE,CHECKPOINT,source_hashes
from qwen_one_two.training.manifest import BASE_SHA,DIAGNOSTIC_SHA,file_hash
from qwen_one_two.training.release import load_release,decode
from qwen_one_two.g1_device import PRIMARY,diagnostic_device,require_cuda


def device_worker(index,prepared,lanes,jobs,events,start,finish):
    os.environ['CUDA_VISIBLE_DEVICES']=PRIMARY
    began=time.process_time_ns()
    try:
        with diagnostic_device(PRIMARY,execute_gpu=True),ExitStack() as stack:
            import torch
            from torch.nn.attention import sdpa_kernel,SDPBackend
            from qwen_one_two.g1_proof import configure
            from qwen_one_two.g1_evidence import frozen_check
            from qwen_one_two.codec.qualify import load
            from qwen_one_two.codec import recurrence
            from qwen_one_two.training.reference import Reference
            from qwen_one_two.training.receiver_learning import groups
            from qwen_one_two.training.checkpoint import restore_model
            from qwen_one_two.training.behavior_qualification import receiver_hash,evaluate_case
            from qwen_one_two.training.clean import CleanEncoder
            from qwen_one_two.training.shared_frozen import clone_frozen
            check(prepared['sources']==source_hashes(),'source drift')
            check(file_hash(Path(__file__))==prepared['shared_context_benchmark']['runner_sha256'],'runner drift')
            require_cuda(torch.cuda);check(torch.cuda.device_count()==1,'one physical CUDA context required')
            configure();torch.set_grad_enabled(False)
            stack.enter_context(sdpa_kernel(SDPBackend.MATH));stack.enter_context(torch.no_grad())
            snapshot=Path(prepared['snapshot']);initial=ROOT/'data/g1-proof/gate-repair-v2-20261105/diagnostic.safetensors'
            check(file_hash(snapshot/'model.safetensors-00001-of-00001.safetensors')==BASE_SHA,'base file hash')
            check(file_hash(initial)==DIAGNOSTIC_SHA,'initial diagnostic hash')
            checkpoint=ROOT/CHECKPOINT
            check(file_hash(checkpoint/'manifest.json')==prepared['checkpoint_manifest_sha256'],'checkpoint hash')
            manifest=decode((checkpoint/'manifest.json').read_bytes())
            release=load_release(ROOT/'data/training-collections-20261105/releases/development',expected_hash=DEV_SHA)
            b,c,_=load(snapshot,initial);ref=Reference(b,c,pooler_kind='target-v2');ref.freeze();groups(b,c,ref)
            restore_model(checkpoint,b,c,ref,manifest['identities']);b.requires_grad_(False);c.requires_grad_(False)
            check(b.compensated and b.precision_parts==2 and b.feedback_precision=='fp32'
                  and b.recurrence_backend=='graph-v1','v3 required')
            def integrity():
                base=frozen_check(b,snapshot)
                check(base['pass'] and base['base_tied_storage'],'W/tying mismatch')
                check(receiver_hash(b,c)==RECEIVER and ref.fingerprint()==REFERENCE==ref.artifact_id,'weights changed')
                check(all(not p.requires_grad and p.grad is None for m in (b,c,ref) for p in m.parameters()),'read-only ownership')
                return {'base':base,'receiver':RECEIVER,'reference':REFERENCE,'gradients_none':True,'optimizer_updates':0}
            initial_integrity=integrity();bundles=[(b,c,ref)];sharing=[]
            for _ in range(lanes-1):
                copied,evidence=clone_frozen(bundles[0]);bundles.append(copied);sharing.append(evidence)
            # The existing stream-keyed cache retains up to 32 shapes per lane.
            # Prewarm sequentially; global capture is forbidden once main starts.
            recurrence.CACHE_LIMIT=32*lanes;recurrence.clear_graph_cache()
            streams=[torch.cuda.Stream() for _ in bundles]
            for stream in streams:stream.wait_stream(torch.cuda.current_stream())
            warm_lock=threading.Lock();initialized=threading.Barrier(lanes+1)
            warmed=queue.Queue();stress_start=threading.Event();local_start=threading.Event();errors=queue.Queue()
            def lane(number,bundle):
                try:
                    with torch.no_grad(),torch.cuda.stream(streams[number]):
                        # Initialize thread-local CUDA state before any capture.
                        torch.empty(1,device='cuda');streams[number].synchronize()
                        initialized.wait()
                        lb,lc,lr=bundle
                        with warm_lock:
                            encoder=CleanEncoder(lb,lr,checkpoint_hash=RECEIVER,
                                profile_hash=digest(prepared['profile'].encode()),cache_references=True)
                            warm=[]
                            for key in dict.fromkeys(prepared['anchors']+[c['id'] for c in prepared['cases']]):
                                case=evaluate_case(lb,lc,lr,release,release.record(key),encoder)
                                streams[number].synchronize();warm.append(case)
                            warmed.put({'kind':'warm','process':index,'lane':number,'cases':warm})
                        stress_start.wait()
                        # Exercise actual reference functional_call concurrently,
                        # rather than hiding it behind the main-phase memo cache.
                        encoder.reference_cache.clear()
                        stress=[]
                        for key in dict.fromkeys(prepared['anchors']+[c['id'] for c in prepared['cases']]):
                            stress.append(evaluate_case(lb,lc,lr,release,release.record(key),encoder))
                        streams[number].synchronize()
                        warmed.put({'kind':'stress','process':index,'lane':number,'cases':stress})
                        local_start.wait()
                        count=0
                        while True:
                            job=jobs.get()
                            if job is None:break
                            wall=time.perf_counter_ns();cpu=time.thread_time_ns()
                            cases=[evaluate_case(lb,lc,lr,release,release.record(k),encoder) for k in job['ids']]
                            streams[number].synchronize()
                            events.put({'kind':'result','process':index,'lane':number,'job':job,'cases':cases,
                                'wall_ns':time.perf_counter_ns()-wall,'python_thread_cpu_ns':time.thread_time_ns()-cpu})
                            count+=len(cases)
                        events.put({'kind':'lane_done','process':index,'lane':number,'cases':count})
                except BaseException:
                    error=traceback.format_exc();errors.put(error);events.put({'kind':'error','process':index,'error':error})
                    try:initialized.abort()
                    except Exception:pass
            threads=[threading.Thread(target=lane,args=(n,v),name=f'case-lane-{n}') for n,v in enumerate(bundles)]
            for thread in threads:thread.start()
            initialized.wait()
            for _ in threads:
                while True:
                    if not errors.empty():raise RuntimeError(errors.get())
                    try:events.put(warmed.get(timeout=.1));break
                    except queue.Empty:pass
            with recurrence.replay_only():
                stress_start.set()
                for _ in threads:
                    while True:
                        if not errors.empty():raise RuntimeError(errors.get())
                        try:events.put(warmed.get(timeout=.1));break
                        except queue.Empty:pass
                torch.cuda.synchronize()
                events.put({'kind':'ready','process':index,'pid':os.getpid(),'uuid':PRIMARY,'lanes':lanes,
                    'initial':initial_integrity,'sharing':sharing,'setup_process_cpu_ns':time.process_time_ns()-began,
                    'memory':{'allocated':torch.cuda.memory_allocated(),'reserved':torch.cuda.memory_reserved()},
                    'graph_entries':len(recurrence._CACHE),'cache_limit':recurrence.CACHE_LIMIT})
                start.wait();cpu=time.process_time_ns();wall=time.perf_counter_ns()
                torch.cuda.reset_peak_memory_stats();local_start.set()
                for thread in threads:thread.join()
                main_cpu=time.process_time_ns()-cpu;main_wall=time.perf_counter_ns()-wall
            check(errors.empty(),'lane failed')
            memory={'allocated':torch.cuda.memory_allocated(),'reserved':torch.cuda.memory_reserved(),
                    'peak_allocated':torch.cuda.max_memory_allocated(),'peak_reserved':torch.cuda.max_memory_reserved()}
            # No worker may hash/check final weights while another runs cases.
            finish.wait()
            if lanes>1:
                from qwen_one_two.training.shared_frozen import check_views
                check(check_views(bundles[0],bundles[1])==sharing[0],'view ownership changed')
            final=integrity()
            events.put({'kind':'done','process':index,'main_process_cpu_ns':main_cpu,'main_wall_ns':main_wall,
                'total_process_cpu_ns':time.process_time_ns()-began,'final':final,'graph_entries':len(recurrence._CACHE),
                'memory':memory})
    except BaseException:events.put({'kind':'error','process':index,'error':traceback.format_exc()});raise


def run(snapshot,out,mode,repetitions):
    out=Path(out).resolve();check(out.is_relative_to(ROOT/'data'),'outputs belong in data/')
    out.mkdir(parents=True,exist_ok=False)
    prepared=plan(snapshot,selection='cpu-profile',devices=(PRIMARY,),cache_limit=32,repetitions=repetitions)
    prepared['shared_context_benchmark']={'mode':mode,'runner_sha256':file_hash(Path(__file__))}
    (out/'plan.json').write_bytes(canonical(prepared))
    nproc,lanes={'serial':(1,1),'replicas':(2,1),'shared':(1,2)}[mode]
    ctx=mp.get_context('spawn');jobs=ctx.Queue();events=ctx.Queue();start=ctx.Event()
    finish=ctx.Barrier(nproc)
    processes=[ctx.Process(target=device_worker,args=(i,prepared,lanes,jobs,events,start,finish)) for i in range(nproc)]
    began=time.perf_counter_ns();cpu_began=time.process_time_ns();ready={};done={};receipts=[];warm=[];stress=[];measurements=[];lane_counts={}
    scheduled={canonical(j) for j in prepared['jobs']};observed=set()
    release=load_release(ROOT/'data/training-collections-20261105/releases/development',expected_hash=DEV_SHA)
    expected={c['id']:c for c in prepared['cases']+prepared['anchor_cases']};baseline={}
    def admit(case):
        validate_case(case,expected[case['id']],release)
        key=case['id']
        if key in baseline:check(case==baseline[key],'same-GPU receipt changed: '+key)
        else:baseline[key]=case
    main_start=main_cpu_start=main_end=main_cpu_end=None
    try:
        with (out/'events.jsonl').open('x') as journal:
            for process in processes:process.start()
            while len(done)<nproc:
                try:m=events.get(timeout=.2)
                except queue.Empty:
                    check(all(p.exitcode in (None,0) for p in processes),'child failed without receipt')
                    check(any(p.is_alive() for p in processes),'children exited without complete receipts')
                    continue
                journal.write(json.dumps(m,allow_nan=False)+'\n');journal.flush()
                check(m['kind']!='error',m.get('error','child error'))
                if m['kind'] in ('warm','stress'):
                    for case in m['cases']:admit(case)
                    (warm if m['kind']=='warm' else stress).append(m)
                elif m['kind']=='ready':
                    ready[m['process']]=m
                    if len(ready)==nproc:
                        for job in prepared['jobs']:jobs.put(job)
                        for _ in range(nproc*lanes):jobs.put(None)
                        main_start=time.perf_counter_ns();main_cpu_start=time.process_time_ns();start.set()
                elif m['kind']=='result':
                    job_key=canonical(m['job'])
                    check(job_key in scheduled and job_key not in observed,'unplanned/duplicate whole job')
                    observed.add(job_key)
                    check([c['id'] for c in m['cases']]==m['job']['ids'],'whole job coverage')
                    for case in m['cases']:
                        admit(case);receipts.append({'repetition':m['job'].get('repetition',0),
                            'process':m['process'],'lane':m['lane'],'case':case})
                    measurements.append({k:v for k,v in m.items() if k!='cases'})
                    if len(receipts)==prepared['presentations']['main']:
                        main_end=time.perf_counter_ns();main_cpu_end=time.process_time_ns()
                elif m['kind']=='lane_done':
                    key=f"{m['process']}/{m['lane']}"
                    check(key not in lane_counts,'duplicate lane final');lane_counts[key]=m['cases']
                elif m['kind']=='done':done[m['process']]=m
                else:raise ValueError('unknown worker event')
            for p in processes:p.join();check(p.exitcode==0,'worker exit')
        count=prepared['presentations']['main']
        check(len(receipts)==count and len({(r['repetition'],r['case']['id']) for r in receipts})==count,'complete unique presentations')
        check(observed==scheduled and len(lane_counts)==nproc*lanes and sum(lane_counts.values())==count,'complete lane/job counts')
        check(len(warm)==nproc*lanes and all(len(w['cases'])==len(expected) for w in warm),'complete lane warmup')
        check(len(stress)==nproc*lanes and all(len(w['cases'])==len(expected) for w in stress),'complete cold-reference stress')
        cpu=sum(d['main_process_cpu_ns'] for d in done.values())+main_cpu_end-main_cpu_start
        report={'status':'COMPLETED_UNQUALIFIED','mode':mode,'processes':nproc,'lanes_per_process':lanes,
            'main_case_count':count,'main_wall_seconds':(main_end-main_start)/1e9,
            'lane_case_counts':lane_counts,
            'main_cpu_seconds':cpu/1e9,'main_cpu_seconds_per_case':cpu/1e9/count,
            'cases_per_second':count/((main_end-main_start)/1e9),'end_to_end_seconds':(time.perf_counter_ns()-began)/1e9,
            'total_cpu_seconds':(time.process_time_ns()-cpu_began+sum(d['total_process_cpu_ns'] for d in done.values()))/1e9,
            'ready':ready,'done':done,'receipts':receipts,'warmups':warm,'cold_reference_stress':stress,'job_measurements':measurements,
            'training_authorized':False,'optimizer_updates':0,'accelerator_qualification_granted':False}
        (out/'report.json').write_bytes(canonical(report))
        print(json.dumps({k:v for k,v in report.items() if k not in {'receipts','warmups','cold_reference_stress','ready','done','job_measurements'}}),flush=True)
    except BaseException:
        (out/'failure.json').write_bytes(canonical({'error':traceback.format_exc(),'receipts':receipts,'ready':ready,'done':done}))
        raise
    finally:
        for p in processes:
            if p.is_alive():p.terminate()
        for p in processes:
            if p.pid:p.join()
        jobs.close();events.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--snapshot',required=True);p.add_argument('--output',required=True)
    p.add_argument('--mode',choices=('serial','replicas','shared'),required=True)
    p.add_argument('--repetitions',type=int,default=3);p.add_argument('--execute-gpu',action='store_true')
    a=p.parse_args();check(a.execute_gpu,'explicit --execute-gpu required')
    run(a.snapshot,a.output,a.mode,a.repetitions)
