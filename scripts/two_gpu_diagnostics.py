"""Prepare without GPU imports; explicit run launches two read-only GPU children."""
import argparse
from collections import Counter
import json
from pathlib import Path
from qwen_one_two.training.inputs import canonical,digest,check
from qwen_one_two.training.release import decode,load_release
from qwen_one_two.training.two_gpu import (ROOT,PROTOCOL,DEV_SHA,plan,dispatch,compare_ready,validate_case)
from qwen_one_two.g1_device import PRIMARY,SECONDARY,DEVICES
from qwen_one_two.training.host_metrics import Recorder,stamp,elapsed,aggregate


def output_directory(path):
    target=Path(path).resolve()
    check(target.is_relative_to(ROOT/'data'),'diagnostic outputs must stay inside repository data/')
    return target


def run(path, *, execute_gpu=False):
    check(execute_gpu is True,'run requires explicit --execute-gpu authorization')
    parent_start=stamp();parent_metrics=Recorder()
    path=Path(path).resolve();out=output_directory(path.parent)
    prepared=decode(path.read_bytes())
    check({'selection','profiling','anchor_cases'}<=set(prepared),'old measurement plan; prepare a new source-bound plan')
    check(prepared==plan(prepared['snapshot'],selection=prepared['selection'],
        profile_host=prepared['profiling']['host'],trace_cuda=prepared['profiling']['cuda_trace_first_job_per_worker'],
        devices=tuple(d['uuid'] for d in prepared['devices'])),
        'plan/source/data/checkpoint metadata drift')
    # An existing journal/lock is never overwritten or silently replayed. Partial
    # receipts are forensic evidence, not admitted accelerator qualification.
    check(not (out/'events.jsonl').exists() and not (out/'report.json').exists(),'existing run receipts; do not replay')
    with (out/'run.lock').open('x') as lock:lock.write(digest(canonical(prepared))+'\n')
    release=load_release(ROOT/'data/training-collections-20261105/releases/development',expected_hash=DEV_SHA)
    parent_preparation=elapsed(parent_start)
    expected={c['id']:c for c in prepared['cases']};anchor_expected={c['id']:c for c in prepared['anchor_cases']}
    received={};finals={};readiness={};measurements=[]
    with (out/'events.jsonl').open('x') as stream:
        def emit(entry):
            with parent_metrics.span('journal_encode'):body=json.dumps(entry,allow_nan=False)+'\n'
            with parent_metrics.span('journal_write_flush'):stream.write(body);stream.flush()
        def write_report(report):
            start=stamp();body=canonical(report);encoded=stamp()
            (out/'report.json').write_bytes(body);written=stamp()
            emit({'event':'report_write_measurement','encode':elapsed(start,encoded),
                  'write_close':elapsed(encoded,written),'bytes':len(body)})
        def ready(messages):
            emit({'event':'ready','workers':messages})
            for m in messages.values():
                for case in m['anchors']:validate_case(case,anchor_expected[case['id']],release)
            compare_ready(messages,prepared);readiness.update(messages)
        def result(uuid,index,message):
            emit({'event':'result','uuid':uuid,**message})
            check([c['id'] for c in message['cases']]==prepared['jobs'][index]['ids'],'job case coverage/order')
            check([m['id'] for m in message['case_measurements']]==prepared['jobs'][index]['ids'],'measurement case coverage/order')
            measurements.extend(dict(m,uuid=uuid) for m in message['case_measurements'])
            for case in message['cases']:
                validate_case(case,expected[case['id']],release)
                check(case['id'] not in received,'duplicate case receipt');received[case['id']]=(uuid,case)
        def final(uuid,message):
            emit({'event':'final','worker':message})
            check(message['uuid']==uuid and message['base_frozen'] and message['gradients_none']
                  and message['optimizer_updates']==0 and message['receiver']==prepared['receiver']
                  and message['reference']==prepared['reference'] and message['device_name'].endswith(DEVICES[uuid]),
                  'invalid final read-only receipt')
            finals[uuid]=message
        try:
            from qwen_one_two.training.two_gpu_worker import worker
            worker_plan=dict(prepared,output_directory=str(out))
            devices=[d['uuid'] for d in prepared['devices']]
            timing=dispatch(prepared['jobs'],devices,worker,(worker_plan,True),
                on_ready=lambda m:parent_metrics.call('ready_validate',ready,m),
                on_result=lambda u,i,m:parent_metrics.call('result_validate',result,u,i,m),
                on_final=lambda u,m:parent_metrics.call('final_validate',final,u,m))
            check(set(received)==set(expected) and set(finals)==set(devices),'incomplete diagnostic run')
            cases=[received[c['id']][1] for c in prepared['cases']]
            from qwen_one_two.training.behavior_qualification import summarize,grade
            x=[c for c in cases if expected[c['id']]['panel']=='X']
            summary=(summarize(release,x,unchanged=True) if prepared['selection']=='full' else
                     {'status':'PROFILING_SUBSET_NOT_QUALIFICATION','scheduled':len(x),
                      'passed':sum(grade(release.record(c['id']),c)['passed'] for c in x)})
            e=[c for c in cases if expected[c['id']]['panel']=='E']
            e_summary={op:{'cases':sum(expected[c['id']]['operation']==op for c in e),
                'passed':sum(grade(release.record(c['id']),c)['passed'] for c in e if expected[c['id']]['operation']==op)}
                for op in ('Noul','Choice','Score')}
            report={'protocol':PROTOCOL,'status':'COMPLETED_UNQUALIFIED','plan_sha256':digest(canonical(prepared)),
                'scope':'experimental forced latent receiver diagnostics; not learned invocation, native answer use or accelerator qualification',
                'timing':timing,'device_case_counts':dict(Counter(u for u,_ in received.values())),
                'worker_initial':readiness,'worker_final':finals,'cases':cases,'X_summary':summary,'E_summary':e_summary,
                'optimizer_updates':0,'training_authorized':False,'accelerator_qualification_granted':False}
            parent_total=elapsed(parent_start)
            report['host_measurements']=aggregate(dict(parent_total,preparation=parent_preparation,reporting=parent_metrics.snapshot()),
                {u:m['host_measurements'] for u,m in finals.items()},measurements,parent_total['wall_ns'])
            order={c['id']:i for i,c in enumerate(prepared['cases'])}
            report['case_measurements']=sorted(measurements,key=lambda m:order[m['id']])
            report['profiling']=prepared['profiling']
        except BaseException as exc:
            report={'protocol':PROTOCOL,'status':'FAILED_UNQUALIFIED','plan_sha256':digest(canonical(prepared)),
                'error':repr(exc),'received_case_ids':list(received),'final_worker_receipts':finals,
                'accelerator_qualification_granted':False,'training_authorized':False}
            report['parent_measurements']=dict(elapsed(parent_start),reporting=parent_metrics.snapshot())
            write_report(report);raise
        write_report(report)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    prepare=sub.add_parser('prepare');prepare.add_argument('--snapshot',required=True);prepare.add_argument('--output',required=True)
    prepare.add_argument('--selection',choices=('full','cpu-profile'),default='full')
    prepare.add_argument('--profile-host',action='store_true')
    prepare.add_argument('--trace-cuda',action='store_true',help='future execution only: trace first main job per worker; implies host profiling')
    prepare.add_argument('--device',choices=('3090','3060','both'),default='both')
    execute=sub.add_parser('run');execute.add_argument('--plan',required=True);execute.add_argument('--execute-gpu',action='store_true')
    a=p.parse_args()
    if a.command=='prepare':
        devices={'3090':(PRIMARY,),'3060':(SECONDARY,),'both':(PRIMARY,SECONDARY)}[a.device]
        prepared=plan(a.snapshot,selection=a.selection,profile_host=a.profile_host or a.trace_cuda,trace_cuda=a.trace_cuda,devices=devices)
        out=output_directory(a.output);out.mkdir(parents=True,exist_ok=False)
        (out/'plan.json').write_bytes(canonical(prepared))
        print(json.dumps({'status':prepared['status'],'jobs':len(prepared['jobs']),'cases':len(prepared['cases']),
                          'anchors':prepared['anchors'],'plan':str(out/'plan.json')}))
    else:
        report=run(a.plan,execute_gpu=a.execute_gpu);print(json.dumps({'status':report['status']}))


if __name__=='__main__':main()
