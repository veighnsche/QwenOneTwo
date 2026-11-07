"""Model-free planning, dispatch and readout comparison for two GPU processes.

Each child owns whole independent Sessions. This is never a distributed trainer
or an accelerator qualification certificate. No model import at module scope.
"""
from collections import deque
import multiprocessing as mp
from multiprocessing.connection import wait
from pathlib import Path
import math
import time
from ..g1_device import PRIMARY,SECONDARY,DEVICES
from .inputs import canonical,digest,check
from .release import load_release,decode
from .behavior_qualification import suite,grade,PROFILE
from .component_producer import component_view
from .manifest import BASE_SHA,DIAGNOSTIC_SHA,file_hash

PROTOCOL='two-gpu-receiver-diagnostic-v1'
DEV_SHA='5b8f4cffdc4f33e255d3a34306cac03a68c50dcd9dab905ae8d8adcf89c81e77'
E_RECEIPT_SHA='6fd5a5b676c4e031a25e5479d7b67de42b8089e123cb0b117ef78d263cc7a72f'
CHECKPOINT=Path('data/receiver-learning-20261106/continued/checkpoint-0188')
CHECKPOINT_SHA='0cd1d4bc55341e78339c1b0b05dec0b3650573d7501fb3f3d2fc6d2f649b6e58'
RECEIVER='94218aae6ec62a7745c349af8e9975fd4987520cf9f1345f38e44586e5e7742c'
REFERENCE='5c6015dfedcd0667ab893025b39e8c35cb08cb04701138ecd66e310b64c45d3f'
ROOT=Path(__file__).resolve().parents[2]


def source_hashes():
    paths=sorted((ROOT/'qwen_one_two').rglob('*.py'))+[ROOT/'scripts'/name for name in
        ('two_gpu_diagnostics.py','benchmark_recurrence_cache.py','benchmark_attention_block.py','qualify_attention_block.py')]
    return {str(p.relative_to(ROOT)):file_hash(p) for p in paths}


def plan(snapshot, *, selection='full', profile_host=False, trace_cuda=False, devices=(PRIMARY,SECONDARY), cache_limit=8,
         workers_per_device=1,repetitions=1):
    """Read existing data/manifest metadata, never weights or GPU libraries."""
    check(selection in {'full','cpu-profile'} and (not trace_cuda or profile_host),'profiling selection/options')
    check(tuple(devices) in ((PRIMARY,),(SECONDARY,),(PRIMARY,SECONDARY)),'explicit diagnostic devices')
    check(type(cache_limit) is int and cache_limit in (8,32),'diagnostic recurrence cache capacity')
    check(type(workers_per_device) is int and workers_per_device in (1,2),'diagnostic workers per device')
    check(type(repetitions) is int and repetitions>=1,'positive diagnostic repetitions')
    release=load_release(ROOT/'data/training-collections-20261105/releases/development',expected_hash=DEV_SHA)
    manifest=decode((ROOT/CHECKPOINT/'manifest.json').read_bytes())
    check(file_hash(ROOT/CHECKPOINT/'manifest.json')==CHECKPOINT_SHA and manifest['reference_frozen']
          and manifest['reference_id']==REFERENCE and manifest['identities']['base']==BASE_SHA,'checkpoint metadata drift')
    groups,omitted=suite(release)
    eids=[e['id'] for e in release.manifest['records'] if e['selected'] and
          release.record(e['id']).get('evaluation_slice')=='E']
    check(len(eids)==130 and len(groups)==90,'complete existing E/X panels')
    jobs=[{'ids':[key],'panel':'E'} for key in eids]+[{'ids':g['ids'],'panel':'X','pair':g['id']} for g in groups]
    cases=[]
    for job in jobs:
        for key in job['ids']:
            row=release.record(key);view,p,_=component_view(release,row,intervention=True)
            cases.append({'id':key,'panel':job['panel'],'operation':row['operation'],
                          'record_sha256':digest(canonical(row)),'view_sha256':digest(view.materialize(p)[1])})
    anchors=[next(c['id'] for c in cases if c['panel']=='E' and c['operation']==op) for op in ('Noul','Choice','Score')]
    anchor_cases=[next(c for c in cases if c['id']==key) for key in anchors]
    selection_evidence=None
    if selection=='cpu-profile':
        receipts=ROOT/'data/receiver-learning-20261106/continued/final-E.jsonl'
        check(file_hash(receipts)==E_RECEIPT_SHA,'profiling selection receipt drift')
        saved=[decode(line) for line in receipts.read_bytes().splitlines()]
        check({c['id'] for c in saved}==set(eids) and len(saved)==130,'existing E receipt coverage')
        lookup={c['id']:c for c in cases};chosen=set();extremes={}
        check(all(c['record_sha256']==lookup[c['id']]['record_sha256'] and
                  c['view_sha256']==lookup[c['id']]['view_sha256'] for c in saved),'profiling receipt input drift')
        for op in ('Noul','Choice','Score'):
            panel=sorted((c for c in saved if lookup[c['id']]['operation']==op),key=lambda c:(c['runtime']['total'],c['id']))
            extremes[op]=[panel[0]['id'],panel[-1]['id']];chosen.update(extremes[op])
        chosen.update(groups[0]['ids'])
        jobs=[j for j in jobs if all(key in chosen for key in j['ids'])]
        cases=[c for c in cases if c['id'] in chosen]
        check(len(cases)==8 and len(jobs)==7,'six existing extrema and one intact X pair')
        selection_evidence={'method':'min/max historical Session total positions per operation; first canonical X pair',
            'receipt_path':str(receipts.relative_to(ROOT)),'receipt_sha256':file_hash(receipts),
            'extrema':extremes,'pair':groups[0]}
    if repetitions>1:jobs=[dict(job,repetition=r) for r in range(repetitions) for job in jobs]
    workers=[{'id':u if workers_per_device==1 else u+'/'+str(i),'uuid':u,'expected_name':DEVICES[u]}
             for u in devices for i in range(workers_per_device)]
    inventory=ROOT/'data/g1-proof/run-20261105/manifest.json'
    check(SECONDARY in decode(inventory.read_bytes())['driver'],'secondary UUID absent from saved inventory')
    return {'protocol':PROTOCOL,'profile':PROFILE,'sources':source_hashes(),'dataset':DEV_SHA,
        'snapshot':str(Path(snapshot).absolute()),'base_sha256':BASE_SHA,'diagnostic_sha256':DIAGNOSTIC_SHA,
        'checkpoint':str(CHECKPOINT),'checkpoint_manifest_sha256':CHECKPOINT_SHA,
        'receiver':RECEIVER,'reference':REFERENCE,'devices':[{'uuid':u,'expected_name':DEVICES[u]} for u in devices],
        'workers':workers,'workers_per_device':workers_per_device,'repetitions':repetitions,
        'device_inventory':{'path':str(inventory.relative_to(ROOT)),'sha256':file_hash(inventory),
                            'names_verified_in_this_slice':False},
        'jobs':jobs,'cases':cases,'anchors':anchors,'omitted_reader_ids':omitted,
        'anchor_cases':anchor_cases,'selection':selection,'selection_evidence':selection_evidence,
        'profiling':{'host':bool(profile_host),'cuda_trace_first_job_per_worker':bool(trace_cuda)},
        'recurrence_cache_limit':cache_limit,
        'presentations':{'main':len(cases)*repetitions,'startup_anchors':3*len(workers),'distinct_existing_cases':len({c['id'] for c in cases}|set(anchors))},
        'status':'PREPARED_GPU_NOT_RUN','training_authorized':False,
        'qualification':'experimental read-only diagnostics; neither cross-device parity nor 3060 numerics qualified',
        'weight_file_hash_verification':'deferred to explicitly executed workers'}


def validate_case(case, expected, release):
    check(case['id']==expected['id'] and case['record_sha256']==expected['record_sha256'],'case identity drift')
    check(case.get('view_sha256')==expected['view_sha256'],'case permitted input drift')
    check(case['score']==grade(release.record(case['id']),case),'case grading drift')
    summary=case.get('runtime')
    check(summary is not None and summary['prefills']==summary['lineage']==1,'missing one-cache Session receipt')
    check(summary['total']==sum(summary[k] for k in ('P','B','A','H')) and summary['held_positions']==0,
          'case accounting/holds')
    check(summary['boundaries'] and summary['boundaries'][-1][1]==summary['total'],'case final cache position')


def compare_ready(messages, prepared):
    """Small shared-case observation only; not full state/gradient qualification."""
    specs=prepared.get('workers',[{'id':d['uuid'],'uuid':d['uuid']} for d in
        prepared.get('devices',[{'uuid':PRIMARY},{'uuid':SECONDARY}])])
    devices={w['id']:w['uuid'] for w in specs}
    check(len(devices)==len(specs),'duplicate planned worker identity')
    check(set(messages)==set(devices),'planned unique worker readiness receipts required')
    same_device={}
    for worker_id,m in messages.items():
        uuid=devices[worker_id]
        check(m['uuid']==uuid and m['receiver']==prepared['receiver'] and m['reference']==prepared['reference']
              and m['base_frozen'] and m['gradients_none'] and m['optimizer_updates']==0
              and m.get('worker_id',worker_id)==worker_id
              and m['device_name'].endswith(DEVICES[uuid]),'worker weight identity/device/read-only check')
        check([c['id'] for c in m['anchors']]==prepared['anchors'],'shared anchor order')
        if uuid in same_device:check(m['anchors']==same_device[uuid],'same-device anchor exact mismatch')
        else:same_device[uuid]=m['anchors']
        for case in m['anchors']:
            check(case['failure'] is None,'invalid shared anchor')
            flags=('latent_request','latent_return','bound_snapshot','candidate_binding','consumed',
                   'single_prefill','cache_continuity','accounting')
            check(all(case['transport'].get(k) is True for k in flags),'invalid shared anchor transport')
    reference=next(iter(messages));left=messages[reference]['anchors']
    for worker_id in list(messages)[1:]:
        right=messages[worker_id]['anchors']
        for a,b in zip(left,right):
            check(a['failure'] is None and b['failure'] is None,'invalid shared anchor')
            flags=('latent_request','latent_return','bound_snapshot','candidate_binding','consumed',
                   'single_prefill','cache_continuity','accounting')
            check(all(a['transport'].get(k) is True and b['transport'].get(k) is True for k in flags),'invalid shared anchor transport')
            for key in ('predicted_status','score','transport'):
                check(a[key]==b[key],'cross-device anchor discrete mismatch: '+key)
            for key in ('status','selected','reason_present'):
                check(a['actual'][key]==b['actual'][key],'cross-device anchor result mismatch')
            for x,y in ((a.get('predicted_value'),b.get('predicted_value')),(a['actual']['value'],b['actual']['value'])):
                check((x is None)==(y is None),'cross-device value presence')
                if x is not None:check(math.isfinite(x) and math.isfinite(y) and abs(x-y)<=.002,'cross-device value tolerance')
            check(a['runtime']==b['runtime'],'cross-device anchor control/accounting mismatch')

def dispatch(jobs, devices, target, args, *, on_ready, on_result, on_final):
    """One to four processes pull whole jobs on completion; no throughput ratio.

    Any infrastructure failure stops our own children; no fallback, retry, or
    existing service control. Callbacks persist partial receipts before cleanup.
    """
    ctx=mp.get_context('spawn');pending=deque(enumerate(jobs));active={};processes=[];ready={};busy={}
    completed=set();stopping=set();started=time.monotonic();cpu_started=time.process_time_ns();sends=receives=0
    main_started=main_finished=None
    check(1<=len(devices)<=4 and len(set(devices))==len(devices),'one to four unique workers required')
    try:
        for uuid in devices:
            parent,child=ctx.Pipe();proc=ctx.Process(target=target,args=(child,uuid,*args))
            proc.start();child.close();active[parent]=(uuid,proc);processes.append(proc)
        def next_job(conn,uuid):
            nonlocal sends
            if pending:
                index,job=pending.popleft();busy[uuid]=index;message={'kind':'job','index':index,'job':job}
            else:stopping.add(uuid);message={'kind':'stop'}
            try:conn.send(message);sends+=1
            except (OSError,EOFError) as exc:raise RuntimeError('worker closed dispatch channel: '+uuid) from exc
        while active:
            signals=wait(list(active)+[p.sentinel for _,p in active.values()])
            for conn in list(active):
                uuid,proc=active[conn]
                if conn not in signals:continue
                try:message=conn.recv()
                except (EOFError,OSError) as exc:raise RuntimeError('worker disconnected before final receipt: '+uuid) from exc
                receives+=1;kind=message.get('kind')
                if kind=='ready':
                    check(uuid not in ready,'duplicate worker readiness');ready[uuid]=message
                    if len(ready)==len(devices):
                        on_ready(ready)
                        main_started=time.monotonic()
                        for pipe,(device,_) in active.items():next_job(pipe,device)
                elif kind=='result':
                    index=message['index'];check(busy.get(uuid)==index and index not in completed,'duplicate/unscheduled result')
                    on_result(uuid,index,message);completed.add(index)
                    if len(completed)==len(jobs):main_finished=time.monotonic()
                    del busy[uuid];next_job(conn,uuid)
                elif kind=='final':
                    check(uuid in stopping and uuid not in busy,'premature worker final receipt')
                    on_final(uuid,message);conn.close();del active[conn]
                elif kind=='error':raise RuntimeError('worker '+uuid+': '+message['error'])
                else:raise ValueError('unknown worker message')
            for conn,(uuid,proc) in active.items():
                if proc.sentinel in signals and not conn.poll():raise RuntimeError('worker exited without final receipt: '+uuid)
        check(not pending and len(completed)==len(jobs),'missing diagnostic jobs')
        for proc in processes:proc.join();check(proc.exitcode==0,'worker nonzero exit')
        return {'jobs_completed':len(completed),'wall_seconds':time.monotonic()-started,
                'main_wall_seconds':None if main_finished is None else main_finished-main_started,
                'parent_dispatch_cpu_ns':time.process_time_ns()-cpu_started,'ipc_sends':sends,'ipc_receives':receives}
    finally:
        for proc in processes:
            if proc.is_alive():proc.terminate()
            proc.join()
        for conn in active:conn.close()
