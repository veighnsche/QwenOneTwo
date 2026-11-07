"""Resident, finite cache-policy A/B using the existing diagnostic worker/dispatch.

One prewarm pack, three alternating pairs, per-trial cold warmup and independent
warm counter observations. Timed packs have no detailed profiling or counters.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import statistics
from qwen_one_two.g1_device import PRIMARY,SECONDARY
from qwen_one_two.training.host_metrics import stamp,elapsed
from qwen_one_two.training.inputs import canonical,check,digest
from qwen_one_two.training.two_gpu import plan,dispatch,compare_ready,validate_case,ROOT,DEV_SHA
from qwen_one_two.training.release import load_release
from scripts.two_gpu_diagnostics import output_directory


def benchmark_plan(snapshot,device):
    p=plan(snapshot,selection='cpu-profile',devices=(device,))
    pack=p['jobs'];jobs=[]
    def add(trial,limit,phase,count):
        for i,j in enumerate(pack):
            jobs.append(dict(j,cache_trial=trial,cache_limit=limit,phase=phase,
                             count_cache=count,last_in_pack=i==len(pack)-1))
    add(-1,8,'reference_prewarm',False)
    for trial,limit in enumerate((8,32,32,8,8,32)):
        add(trial,limit,'warmup',True)
        add(trial,limit,'timed',False)
        if trial<2:add(trial,limit,'observe_warm',True)
    return dict(p,jobs=jobs,cache_benchmark=True,
                benchmark_protocol='recurrence-cache-retention-v1',
                presentations=dict(main=120,startup_anchors=3,distinct_existing_cases=11),
                timing_scope='unprofiled timed packs after cold per-trial warmup; frozen references prewarmed')


def run(prepared,out,*,execute_gpu=False):
    check(execute_gpu is True,'cache benchmark requires explicit --execute-gpu authorization')
    from qwen_one_two.training.two_gpu_worker import worker
    release=load_release(ROOT/'data/training-collections-20261105/releases/development',expected_hash=DEV_SHA)
    expected={c['id']:c for c in prepared['cases']+prepared['anchor_cases']}
    device=prepared['devices'][0]['uuid'];cases={};receipts=[];finals={};starts={};trials={}
    process_start=stamp()
    with (out/'events.jsonl').open('x') as journal:
        def emit(m):journal.write(json.dumps(m,allow_nan=False)+'\n');journal.flush()
        def ready(messages):
            emit({'event':'ready','workers':messages});compare_ready(messages,prepared)
            for m in messages.values():
                for c in m['anchors']:validate_case(c,expected[c['id']],release)
        def result(uuid,index,message):
            arrived=stamp();job=prepared['jobs'][index]
            check([c['id'] for c in message['cases']]==job['ids'],'benchmark case order/coverage')
            check([m['id'] for m in message['case_measurements']]==job['ids'],'benchmark measurement order')
            check(message['job_measurements']['recurrence_cache_limit']==job['cache_limit'],'cache policy drift')
            for c in message['cases']:
                validate_case(c,expected[c['id']],release)
                if c['id'] in cases:check(c==cases[c['id']],'cache trial changed exact case receipt')
                else:cases[c['id']]=c
            receipt=dict(message,job=job,worker=uuid)
            receipts.append(receipt);emit({'event':'result',**receipt})
            if job['last_in_pack'] and job['phase']=='timed':
                trials[job['cache_trial']]={'cache_limit':job['cache_limit'],
                    'parent_timed_interval':elapsed(starts[job['cache_trial']],arrived)}
            if job['last_in_pack'] and job['phase']=='warmup':starts[job['cache_trial']]=stamp()
        def final(uuid,message):
            emit({'event':'final','worker':message})
            check(message['base_frozen'] and message['gradients_none'] and message['optimizer_updates']==0
                  and message['receiver']==prepared['receiver'] and message['reference']==prepared['reference'],
                  'benchmark read-only final integrity')
            finals[uuid]=message
        try:
            timing=dispatch(prepared['jobs'],[device],worker,(dict(prepared,output_directory=str(out)),True),
                            on_ready=ready,on_result=result,on_final=final)
            check(set(cases)=={c['id'] for c in prepared['cases']} and set(finals)=={device},'incomplete cache benchmark')
            for trial,row in trials.items():
                groups={phase:[r for r in receipts if r['job']['cache_trial']==trial and r['job']['phase']==phase]
                        for phase in ('warmup','timed','observe_warm')}
                timed=[m for r in groups['timed'] for m in r['case_measurements']]
                check(len(timed)==8 and len(groups['timed'])==7,'timed pack coverage')
                check(all('cache_counts' not in r['job_measurements'] for r in groups['timed']), 'timed profiling prohibited')
                row.update(main_wall_seconds=sum(m['wall_ns'] for m in timed)/1e9,
                    main_cpu_seconds=sum(m['cpu_ns'] for m in timed)/1e9,
                    peak_allocated_bytes=max(m['memory']['peak_allocated_bytes'] for m in timed),
                    peak_reserved_bytes=max(m['memory']['peak_reserved_bytes'] for m in timed))
                for phase in ('warmup','observe_warm'):
                    c=Counter()
                    for r in groups[phase]:c.update(r['job_measurements']['cache_counts'])
                    row[phase+'_counts']=dict(c)
            check(set(trials)==set(range(6)),'missing cache trials')
            medians={str(limit):{k:statistics.median(r[k] for r in trials.values() if r['cache_limit']==limit)
                     for k in ('main_wall_seconds','main_cpu_seconds','peak_allocated_bytes','peak_reserved_bytes')}
                     for limit in (8,32)}
            parent_total=elapsed(process_start)
            report={'status':'COMPLETED_UNQUALIFIED','protocol':prepared['benchmark_protocol'],
                'plan_sha256':digest(canonical(prepared)),'trials':trials,'medians':medians,'timing':timing,
                'cases':list(cases.values()),'worker_final':finals,'process_measurement':parent_total,
                'parent_plus_worker_cpu_ns':parent_total['cpu_ns']+sum(m['host_measurements']['process_total']['cpu_ns'] for m in finals.values()),
                'model_presentations':prepared['presentations'],'exact_receipts_all_presentations':True,
                'optimizer_updates':0,'training_authorized':False,'accelerator_qualification_granted':False}
        except BaseException as exc:
            (out/'report.json').write_bytes(canonical({'status':'FAILED_UNQUALIFIED','error':repr(exc),
                'completed_jobs':len(receipts),'finals':finals}));raise
    (out/'report.json').write_bytes(canonical(report));return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--snapshot',required=True)
    p.add_argument('--device',choices=('3090','3060'),required=True);p.add_argument('--output',required=True)
    p.add_argument('--execute-gpu',action='store_true');a=p.parse_args()
    out=output_directory(a.output);out.mkdir(parents=True,exist_ok=False)
    prepared=benchmark_plan(a.snapshot,PRIMARY if a.device=='3090' else SECONDARY)
    (out/'plan.json').write_bytes(canonical(prepared))
    if not a.execute_gpu:print(json.dumps({'status':'PREPARED_GPU_NOT_RUN','jobs':len(prepared['jobs'])}));return
    report=run(prepared,out,execute_gpu=True);print(json.dumps({'status':report['status'],'medians':report['medians']}))


if __name__=='__main__':main()
