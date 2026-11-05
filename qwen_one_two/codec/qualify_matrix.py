"""Run the unchanged 324-case qualification in bounded RTX 3090 workers.

The parent only schedules subprocesses and verifies/merges JSON evidence. Each
child runs model/gradient work on the same explicitly selected CUDA device.
"""
import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys
import time

GPU_UUID='GPU-225509c3-d702-6c27-de03-76b5e99b33b1'


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--snapshot',required=True);ap.add_argument('--checkpoint',required=True)
    ap.add_argument('--output',required=True);ap.add_argument('--workers',type=int,choices=(1,2),default=2)
    args=ap.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=GPU_UUID:
        raise RuntimeError('Explicit RTX 3090 UUID required; no CPU fallback')
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    sources={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')}
    pending=list(itertools.product((512,256,128),(2,3,1)))
    active=[];finished=[];started=time.time()
    report={'status':'RUNNING','full_A3_matrix_complete':False,'cases':[],
            'scope':'prescribed 128/256/512 codec numeric qualification; no learned capability claim',
            'training_performed':False,'optimizer_updates':0,'frontend_changed':False,
            'gpu_uuid':GPU_UUID,'workers':args.workers,'source_hashes':sources}
    try:
        while pending or active:
            while pending and len(active)<args.workers:
                length,port=pending.pop(0);name=f'{length}-port{port}'
                handle=(out/(name+'.log')).open('w')
                cmd=[sys.executable,'-m','qwen_one_two.codec.qualify','--snapshot',args.snapshot,
                     '--checkpoint',args.checkpoint,'--output',str(out/name),'--lengths',str(length),
                     '--ports',str(port),'--reserve-state-memory']
                process=subprocess.Popen(cmd,stdout=handle,stderr=subprocess.STDOUT,env=os.environ.copy())
                active.append((name,process,handle))
            for name,process,handle in list(active):
                code=process.poll()
                if code is None:continue
                handle.close();active.remove((name,process,handle))
                if code:raise RuntimeError(f'{name} failed with exit {code}; see its log')
                child=json.loads((out/name/'report.json').read_text())
                if (child['status']!='PARTIAL_MATRIX_PASSED' or not child.get('requested_checks_passed')
                        or not child.get('all_live_parameters_unchanged') or not child['frozen_base']['pass']
                        or child['source_hashes']!=sources):
                    raise RuntimeError(f'{name} evidence mismatch')
                finished.append((name,child))
                print(json.dumps({'completed':name,'cases':len(child['cases']),'groups_complete':len(finished)}),flush=True)
            time.sleep(1)
        rows=[dict(row,evidence_group=name) for name,child in finished for row in child['cases']]
        keys=[tuple(row[k] for k in ('length','port','profile','seed','repeat')) for row in rows]
        expected=set(itertools.product((128,256,512),(1,2,3),('functional','checkpoint','serving','fp32'),(17,29,43),range(3)))
        if len(keys)!=324 or len(set(keys))!=324 or set(keys)!=expected:
            raise RuntimeError('Qualification matrix is incomplete or duplicated')
        if not all(row['comparison']['pass'] for row in rows):raise RuntimeError('Failed case in matrix')
        for field in ('checkpoint_sha256','precision','profile','environment','state_memory_reservation'):
            value=finished[0][1][field]
            if not all(child[field]==value for _,child in finished):raise RuntimeError('Mixed '+field)
            report[field]=value
        if {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')}!=sources:
            raise RuntimeError('Sources changed during qualification')
        report.update(status='PASSED',full_A3_matrix_complete=True,cases=rows,
                      all_live_parameters_unchanged=True,frozen_base_all_groups_pass=True,
                      child_reports={name:hashlib.sha256((out/name/'report.json').read_bytes()).hexdigest() for name,_ in finished})
    except BaseException as exc:
        report.update(status='FAILED',error=str(exc))
        for _,process,handle in active:
            process.terminate()
            try:process.wait(timeout=20)
            except subprocess.TimeoutExpired:process.kill();process.wait()
            handle.close()
    finally:
        report['elapsed_seconds']=time.time()-started
        (out/'report.json').write_text(json.dumps(report,indent=2))
        print(json.dumps({'status':report['status'],'cases':len(report['cases']),'output':str(out)}),flush=True)
    if report['status']!='PASSED':raise SystemExit(1)


if __name__=='__main__':main()
