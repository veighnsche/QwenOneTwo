"""Explicitly executed read-only child. Importing this file does no GPU work."""
from contextlib import ExitStack,nullcontext
import os
from pathlib import Path
import sys
import traceback
from ..g1_device import diagnostic_device,require_cuda
from .inputs import check,digest
from .manifest import BASE_SHA,DIAGNOSTIC_SHA,file_hash
from .host_metrics import Recorder,stamp,elapsed
from .two_gpu import ROOT,CHECKPOINT,PROFILE,RECEIVER,REFERENCE


def worker(connection, worker_id, prepared, execute_gpu=False):
    try:
        check(execute_gpu is True,'GPU execution must be explicitly requested')
        specs=[w for w in prepared['workers'] if w['id']==worker_id]
        check(len(specs)==1,'unplanned diagnostic worker');uuid=specs[0]['uuid']
        started=stamp();basic=Recorder();options=prepared['profiling']
        detail=Recorder() if options['host'] else None
        check('torch' not in sys.modules,'set worker CUDA visibility before model runtime import')
        os.environ['CUDA_VISIBLE_DEVICES']=uuid
        os.chdir(ROOT)
        with diagnostic_device(uuid,execute_gpu=True),ExitStack() as stack:
            if detail:stack.enter_context(detail.activate())
            with basic.span('setup'):
                import torch
                from torch.nn.attention import sdpa_kernel,SDPBackend
                from ..g1_proof import configure
                from ..g1_evidence import frozen_check
                from ..codec.qualify import load
                from .reference import Reference
                from .receiver_learning import groups
                from .checkpoint import restore_model
                from .behavior_qualification import receiver_hash,evaluate_case
                from .release import load_release,decode
                from .clean import CleanEncoder
                from ..codec import recurrence
                from .two_gpu import DEV_SHA,source_hashes
                check(prepared['sources']==source_hashes(),'worker source drift')
                require_cuda(torch.cuda);check(torch.cuda.device_count()==1,'one visible GPU per process')
                configure();torch.set_grad_enabled(False)
                recurrence.CACHE_LIMIT=prepared['recurrence_cache_limit'];recurrence.clear_graph_cache()
                stack.enter_context(sdpa_kernel(SDPBackend.MATH));stack.enter_context(torch.no_grad())
                snapshot=Path(prepared['snapshot']);initial=ROOT/'data/g1-proof/gate-repair-v2-20261105/diagnostic.safetensors'
                check(file_hash(snapshot/'model.safetensors-00001-of-00001.safetensors')==BASE_SHA,'base file hash')
                check(file_hash(initial)==DIAGNOSTIC_SHA,'initial diagnostic hash')
                checkpoint=ROOT/CHECKPOINT
                check(file_hash(checkpoint/'manifest.json')==prepared['checkpoint_manifest_sha256'],'checkpoint metadata drift')
                manifest=decode((checkpoint/'manifest.json').read_bytes())
                release=load_release(ROOT/'data/training-collections-20261105/releases/development',expected_hash=DEV_SHA)
                b,c,_=load(snapshot,initial)
                ref=Reference(b,c,pooler_kind='target-v2');ref.freeze();groups(b,c,ref)
                restore_model(checkpoint,b,c,ref,manifest['identities'])
                b.requires_grad_(False);c.requires_grad_(False)
                check(b.compensated and b.precision_parts==2 and b.feedback_precision=='fp32'
                      and b.recurrence_backend=='graph-v1','v3 arithmetic required')
                def integrity():
                    base=frozen_check(b,snapshot)
                    check(base['pass'] and base['base_tied_storage'],'frozen tied W mismatch')
                    check(receiver_hash(b,c)==RECEIVER and ref.fingerprint()==REFERENCE==ref.artifact_id,'receiver/reference changed')
                    check(all(not p.requires_grad and p.grad is None for m in (b,c,ref) for p in m.parameters()),'read-only model ownership')
                    return {'worker_id':worker_id,'pid':os.getpid(),'uuid':uuid,'device_name':torch.cuda.get_device_name(0),'base_frozen':True,'base_check':base,
                            'receiver':RECEIVER,'reference':REFERENCE,'gradients_none':True,'optimizer_updates':0,
                            'torch_version':torch.__version__,'cuda_runtime':torch.version.cuda,
                            'profile':PROFILE,'recurrence_cache_limit':recurrence.CACHE_LIMIT,
                            'accelerator_qualification_granted':False}
                encoder=CleanEncoder(b,ref,checkpoint_hash=RECEIVER,profile_hash=digest(PROFILE.encode()),cache_references=True)
                if detail:
                    from .host_profile import model_hooks
                    stack.enter_context(model_hooks(detail,b,c,ref,encoder))
                initial_integrity=integrity()
                torch.cuda.synchronize()
                setup_memory={'allocated_bytes':torch.cuda.memory_allocated(),'reserved_bytes':torch.cuda.memory_reserved(),
                    'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved()}
            def evaluated(key,phase,ordinal):
                torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
                before=detail.snapshot() if detail else None;start=stamp()
                with basic.span(phase):case=evaluate_case(b,c,ref,release,release.record(key),encoder)
                torch.cuda.synchronize()
                measurement={'id':key,'phase':phase,'worker_case_ordinal':ordinal,**elapsed(start)}
                measurement['memory']={'allocated_bytes':torch.cuda.memory_allocated(),
                    'reserved_bytes':torch.cuda.memory_reserved(),'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
                    'peak_reserved_bytes':torch.cuda.max_memory_reserved()}
                if detail:measurement['details']=Recorder.difference(before,detail.snapshot())
                return case,measurement
            anchors=[evaluated(k,'startup_anchors',i) for i,k in enumerate(prepared['anchors'])]
            with basic.span('ipc_send'):
                connection.send({'kind':'ready',**initial_integrity,'anchors':[v[0] for v in anchors],
                    'anchor_measurements':[v[1] for v in anchors],'setup_measurements':basic.snapshot(),
                    'setup_memory':setup_memory,'host_threads':{'torch_intraop':torch.get_num_threads(),
                    'torch_interop':torch.get_num_interop_threads()}})
            ordinal=0;traced=False;active_trial=None
            while True:
                with basic.span('ipc_wait'):message=connection.recv()
                if message['kind']=='stop':break
                check(message['kind']=='job','unknown worker command')
                check(message['job']==prepared['jobs'][message['index']],'unplanned diagnostic job')
                if 'cache_trial' in message['job']:
                    job=message['job'];check(prepared.get('cache_benchmark') is True,'unplanned cache benchmark')
                    check(job['cache_limit'] in (8,32),'cache trial capacity')
                    if job['cache_trial']!=active_trial:
                        recurrence.clear_graph_cache();recurrence.CACHE_LIMIT=job['cache_limit'];active_trial=job['cache_trial']
                start=stamp();trace_this=options['cuda_trace_first_job_per_worker'] and not traced
                counter=Recorder() if message['job'].get('count_cache') else None
                trace_context=(torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                torch.profiler.ProfilerActivity.CUDA]) if trace_this else nullcontext())
                with (counter.activate() if counter else nullcontext()),trace_context as trace:
                    if trace_this:
                        with torch.profiler.record_function('qot_diagnostic_job'):
                            results=[evaluated(k,'main_cases',ordinal+i) for i,k in enumerate(message['job']['ids'])]
                    else:results=[evaluated(k,'main_cases',ordinal+i) for i,k in enumerate(message['job']['ids'])]
                trace_path=None
                if trace_this:
                    trace_path=str(Path(prepared['output_directory'])/('cuda-trace-'+worker_id.replace('/','-worker-')+'.json'))
                    trace.export_chrome_trace(trace_path);traced=True
                ordinal+=len(results)
                job_measurement=elapsed(start)
                job_measurement['recurrence_cache_limit']=recurrence.CACHE_LIMIT
                if counter:job_measurement['cache_counts']=counter.snapshot()['counts']
                with basic.span('ipc_send'):
                    connection.send({'kind':'result','index':message['index'],'cases':[v[0] for v in results],
                        'case_measurements':[v[1] for v in results],'job_measurements':job_measurement,
                        'seconds':job_measurement['wall_ns']/1e9,'cuda_trace':trace_path})
            with basic.span('final_integrity'):final=integrity()
            connection.send({'kind':'final',**final,'host_measurements':{'process_total':elapsed(started),
                'basic':basic.snapshot(),'details':detail.snapshot() if detail else None,
                'main_case_count':ordinal,'cutoff':'before final send; main labels do not imply warm shapes'}})
    except BaseException:
        try:connection.send({'kind':'error','error':traceback.format_exc()})
        except (BrokenPipeError,EOFError):pass
        raise
    finally:connection.close()
