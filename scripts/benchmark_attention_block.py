"""Resident original/larger-block A/B on existing cases; no model fitting."""
import argparse
from collections import Counter
import json
from pathlib import Path
import statistics
from qwen_one_two.g1_device import PRIMARY
from qwen_one_two.training.inputs import canonical,check,digest
from qwen_one_two.training.host_metrics import stamp,elapsed
from qwen_one_two.training.two_gpu import ROOT,DEV_SHA,plan,dispatch,compare_ready,validate_case
from qwen_one_two.training.release import load_release
from scripts.two_gpu_diagnostics import output_directory


def benchmark_plan(snapshot):
    p=plan(snapshot,selection='cpu-profile',devices=(PRIMARY,),cache_limit=32)
    jobs=[]
    def add(trial,backend,phase,count=False,trace=False):
        for i,job in enumerate(p['jobs']):
            jobs.append(dict(job,cache_trial=trial,cache_limit=32,attention_block_backend=backend,
                phase=phase,count_cache=count,last_in_pack=i==len(p['jobs'])-1,
                refresh_reference_cache=bool(phase=='reference_parity' and i==0),
                trace_attention_block=bool(trace and i==0)))
    add(-1,'eager','reference_prewarm')
    add(-2,'graph-v1','reference_parity',True)
    for trial,backend in enumerate(('eager','graph-v1','graph-v1','eager','eager','graph-v1')):
        add(trial,backend,'warmup',True);add(trial,backend,'timed')
    # CUPTI initialization can leave overhead after profiler exit. Complete
    # every timing before tracing, with separate cold/warm observation packs.
    for trial,backend in ((6,'eager'),(7,'graph-v1')):
        add(trial,backend,'observation_warmup',True);add(trial,backend,'observe_warm',True,True)
    return dict(p,jobs=jobs,cache_benchmark=True,attention_block_benchmark=True,
        benchmark_protocol='linear-attention-block-v1',
        presentations=dict(main=144,startup_anchors=3,distinct_existing_cases=11),
        execution_profiles=['v3/recurrence-graph-v1','v3/linear-attention-block-v1'],
        timing_scope='three alternating resident warm pairs; no profiling/counters in timed packs')


def run(prepared,out,*,execute_gpu=False):
    check(execute_gpu is True,'benchmark requires explicit --execute-gpu')
    from qwen_one_two.training.two_gpu_worker import worker
    release=load_release(ROOT/'data/training-collections-20261105/releases/development',expected_hash=DEV_SHA)
    expected={c['id']:c for c in prepared['cases']+prepared['anchor_cases']}
    receipts=[];cases={};finals={};starts={};trials={};startup={};process_start=stamp()
    with (out/'events.jsonl').open('x') as journal:
        def emit(m):journal.write(json.dumps(m,allow_nan=False)+'\n');journal.flush()
        def ready(messages):
            emit({'event':'ready','workers':messages});compare_ready(messages,prepared);startup.update(messages)
            for m in messages.values():
                for c in m['anchors']:validate_case(c,expected[c['id']],release)
        def result(uuid,index,message):
            job=prepared['jobs'][index]
            check([c['id'] for c in message['cases']]==job['ids'],'whole job coverage')
            check([m['id'] for m in message['case_measurements']]==job['ids'],'measurement coverage')
            check(message['job_measurements']['attention_block_backend']==job['attention_block_backend'],'execution profile drift')
            for case in message['cases']:
                validate_case(case,expected[case['id']],release)
                if case['id'] in cases:check(case==cases[case['id']],'exact case receipt changed')
                else:cases[case['id']]=case
            receipt=dict(message,job=job,worker=uuid);receipts.append(receipt);emit({'event':'result',**receipt})
            if job['last_in_pack'] and job['phase']=='timed':
                trials[job['cache_trial']]={'backend':job['attention_block_backend'],
                    'parent_interval':elapsed(starts[job['cache_trial']])}
            if job['last_in_pack'] and job['phase']=='warmup':starts[job['cache_trial']]=stamp()
        def final(uuid,message):
            emit({'event':'final','worker':message})
            check(message['base_frozen'] and message['gradients_none'] and message['optimizer_updates']==0
                  and message['receiver']==prepared['receiver'] and message['reference']==prepared['reference'],'frozen final')
            finals[uuid]=message
        try:
            timing=dispatch(prepared['jobs'],[PRIMARY],worker,(dict(prepared,output_directory=str(out)),True),
                            on_ready=ready,on_result=result,on_final=final)
            check(set(cases)=={c['id'] for c in prepared['cases']} and set(finals)=={PRIMARY},'complete benchmark')
            for trial,row in trials.items():
                groups={phase:[r for r in receipts if r['job']['cache_trial']==trial and r['job']['phase']==phase]
                        for phase in ('warmup','timed')}
                timed=[m for r in groups['timed'] for m in r['case_measurements']]
                check(len(timed)==8 and len(groups['timed'])==7,'timed pack coverage')
                check(all('cache_counts' not in r['job_measurements'] and r['cuda_trace'] is None for r in groups['timed']),
                      'timed profiler/counters forbidden')
                worker_cpu=sum(m['cpu_ns'] for m in timed)
                row.update(worker_case_wall_seconds=sum(m['wall_ns'] for m in timed)/1e9,
                    worker_case_cpu_seconds=worker_cpu/1e9,
                    worker_plus_parent_cpu_seconds_per_case=(worker_cpu+row['parent_interval']['cpu_ns'])/8e9,
                    parent_cases_per_second=8e9/row['parent_interval']['wall_ns'],
                    peak_allocated_bytes=max(m['memory']['peak_allocated_bytes'] for m in timed),
                    peak_reserved_bytes=max(m['memory']['peak_reserved_bytes'] for m in timed))
                for phase in ('warmup',):
                    counts=Counter()
                    for r in groups[phase]:counts.update(r['job_measurements']['cache_counts'])
                    row[phase+'_counts']=dict(counts)
            check(set(trials)==set(range(6)),'six trials')
            keys=('worker_case_wall_seconds','worker_case_cpu_seconds','worker_plus_parent_cpu_seconds_per_case',
                  'parent_cases_per_second','peak_allocated_bytes','peak_reserved_bytes')
            medians={backend:{k:statistics.median(r[k] for r in trials.values() if r['backend']==backend) for k in keys}
                     for backend in ('eager','graph-v1')}
            report={'status':'COMPLETED_UNQUALIFIED','protocol':prepared['benchmark_protocol'],
                'plan_sha256':digest(canonical(prepared)),'trials':trials,'medians':medians,'timing':timing,
                'cases':list(cases.values()),'worker_initial':startup,'worker_final':finals,
                'presentations':prepared['presentations'],'process_measurement':elapsed(process_start),
                'exact_receipts_all_presentations':True,'optimizer_updates':0,'training_authorized':False,
                'accelerator_qualification_granted':False}
            report['observations']={}
            for trial,backend in ((6,'eager'),(7,'graph-v1')):
                observed=[r for r in receipts if r['job']['cache_trial']==trial and r['job']['phase']=='observe_warm']
                check(len(observed)==7,'observation pack coverage')
                counts=Counter()
                for r in observed:counts.update(r['job_measurements']['cache_counts'])
                report['observations'][backend]={'counts':dict(counts),
                    'cuda_trace':next(r['cuda_trace'] for r in observed if r['cuda_trace'] is not None)}
        except BaseException as error:
            (out/'report.json').write_bytes(canonical({'status':'FAILED_UNQUALIFIED','error':repr(error),
                'completed_jobs':len(receipts),'finals':finals}));raise
    (out/'report.json').write_bytes(canonical(report));return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--snapshot',required=True)
    p.add_argument('--output',required=True);p.add_argument('--execute-gpu',action='store_true');a=p.parse_args()
    out=output_directory(a.output);out.mkdir(parents=True,exist_ok=False)
    prepared=benchmark_plan(a.snapshot);(out/'plan.json').write_bytes(canonical(prepared))
    if not a.execute_gpu:print(json.dumps({'status':'PREPARED_GPU_NOT_RUN'}));return
    report=run(prepared,out,execute_gpu=True);print(json.dumps({'status':report['status'],'medians':report['medians']}))


if __name__=='__main__':main()
