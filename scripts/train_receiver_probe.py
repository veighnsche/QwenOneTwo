"""Eight-repeat, 32-record training-only learnability probe; frozen W/R/reference."""
import argparse
from collections import Counter
import json
import math
import os
import random
from pathlib import Path
import shutil
import time
from qwen_one_two.training.inputs import canonical,check,digest
from qwen_one_two.training.manifest import file_hash,BASE_SHA,DIAGNOSTIC_SHA
from qwen_one_two.training.two_gpu import ROOT,CHECKPOINT,CHECKPOINT_SHA,RECEIVER,REFERENCE,source_hashes
from qwen_one_two.training.release import load_release,decode
from qwen_one_two.training.learning_report import PROTOCOL,probe_panel,summarize_learning
from scripts.fit_reference import TRAIN_SHA,DEV_SHA
from qwen_one_two.g1_device import PRIMARY,require_visibility

BALANCE_PROTOCOL='receiver-loss-balance-v1'
BALANCE_OVERRIDES={'status':.1,'feasibility':.05}


def loss_balance_baseline(path,panel,sources):
    """Bind the completed control and forbid unrelated executable changes."""
    path=Path(path).resolve();check(path.is_relative_to(ROOT/'data'),'baseline belongs in data/')
    evidence=decode((ROOT/'docs/evidence/receiver-training-probe-v1.json').read_bytes())
    check(file_hash(path/'plan.json')==evidence['plan_sha256'] and
          file_hash(path/'report.json')==evidence['report_sha256'],'baseline plan/report drift')
    for name,sha in evidence['raw_file_sha256'].items():
        check(file_hash(ROOT/name)==sha,'baseline receipt drift: '+name)
    plan=decode((path/'plan.json').read_bytes());report=decode((path/'report.json').read_bytes())
    check(plan['protocol']==PROTOCOL and report['status']=='COMPLETED_PROBE_NOT_PROMOTED'
          and report['optimizer_updates']==32 and report['failed_exposures']=={},'incomplete baseline')
    check(plan['panel']==panel,'comparison panel/order changed')
    for name in ('parent_checkpoint','parent_manifest_sha256','receiver','reference','dataset','development'):
        expected={'parent_checkpoint':str(CHECKPOINT),'parent_manifest_sha256':CHECKPOINT_SHA,'receiver':RECEIVER,
                  'reference':REFERENCE,'dataset':TRAIN_SHA,'development':DEV_SHA}[name]
        check(plan[name]==expected,'comparison parent/data changed: '+name)
    check(set(plan['sources'])==set(sources),'comparison source set changed')
    changed=[n for n,h in plan['sources'].items() if sources[n]!=h]
    check(set(changed)<={'scripts/train_receiver_probe.py'},'unrelated comparison source changes')
    for name,sha in plan['sources'].items():
        check(file_hash(path/'source'/name)==sha,'baseline source archive drift: '+name)
    return {'protocol':BALANCE_PROTOCOL,'baseline':str(path.relative_to(ROOT)),
            'plan_sha256':evidence['plan_sha256'],'report_sha256':evidence['report_sha256'],
            'overrides':dict(BALANCE_OVERRIDES),'scale':.1,'changed_sources':changed},report


def gradient_report(window):
    """Read existing VJP buffers, without altering combination or optimizer math."""
    import torch
    d_names={p['name'] for g in window.ownership if g['name']=='D' for p in g['parameters']}
    d_indices={i for i,n in enumerate(window.names) if n in d_names}
    def squared(buffers,indices=None):
        xs=[g.detach() for i,g in buffers.items() if indices is None or i in indices]
        if not xs:return torch.zeros((),device='cuda',dtype=torch.float64)
        return torch.stack([g.double().square().sum() for g in xs]).sum()
    norms={}
    for name,buffers in window.buffers.items():
        scale=window.coefficients[name]/window.counts[name]
        norms[name]={'eligible':window.counts[name],'coefficient':window.coefficients[name],
            'weighted_norm':float(squared(buffers).sqrt()*scale),
            'weighted_D_norm':float(squared(buffers,d_indices).sqrt()*scale)}
    cosine={};status=window.buffers.get('status',{})
    for name in ('noul','score','choice','alignment'):
        other=window.buffers.get(name,{})
        common=set(status)&set(other)&d_indices
        if not common:cosine[name]=None;continue
        dot=torch.stack([(status[i].double()*other[i].double()).sum() for i in common]).sum()
        denominator=squared(status,d_indices).sqrt()*squared(other,d_indices).sqrt()
        cosine[name]=float(dot/denominator) if float(denominator)>0 else None
    return {'components':norms,'D_cosine_with_status':cosine,
            'scope':'window-aggregate eligible-mean gradients before shared clipping; observational, not causal attribution'}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--snapshot',required=True)
    parser.add_argument('--output',required=True);parser.add_argument('--execute-gpu',action='store_true')
    parser.add_argument('--loss-balance-baseline',help='completed v1 control; run the single fixed tenfold reduction')
    a=parser.parse_args()
    out=Path(a.output).resolve();check(out.is_relative_to(ROOT/'data'),'probe outputs belong in data/')
    check(not out.exists(),'never overwrite a probe');out.mkdir(parents=True)
    train=load_release(ROOT/'data/training-collections-20261105/releases/train',expected_hash=TRAIN_SHA)
    dev=load_release(ROOT/'data/training-collections-20261105/releases/development',expected_hash=DEV_SHA)
    panel=probe_panel(train);order=[p['id'] for p in panel]
    devfamilies={dev.record(e['id'])['family'] for e in dev.manifest['records'] if e['selected']
                 and dev.record(e['id']).get('evaluation_slice') in ('E','X')}
    check(not devfamilies&{p['family'] for p in panel},'probe/development family overlap')
    sources=source_hashes();sources['scripts/train_receiver_probe.py']=file_hash(Path(__file__))
    comparison,baseline=(loss_balance_baseline(a.loss_balance_baseline,panel,sources)
                         if a.loss_balance_baseline else (None,None))
    protocol=BALANCE_PROTOCOL if comparison else PROTOCOL
    plan={'protocol':protocol,'parent_checkpoint':str(CHECKPOINT),'parent_manifest_sha256':CHECKPOINT_SHA,
        'receiver':RECEIVER,'reference':REFERENCE,'dataset':TRAIN_SHA,'development':DEV_SHA,'sources':sources,
        'panel':panel,'repeats':8,'training_presentations':256,'scheduled_probe_windows':32,
        'measurement_repeats':[0,1,2,4,8],'optimizer':'restore checkpoint-0188 moments/RNG/schedule; no warmup reset',
        'training_scope':'D and existing receiver owners only; W/R/budget/reference frozen; no native training',
        'snapshot':str(Path(a.snapshot).resolve())}
    if comparison:plan['loss_balance']=comparison
    (out/'plan.json').write_bytes(canonical(plan))
    for name in sources:
        target=out/'source'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,target)
    if not a.execute_gpu:print(json.dumps({'status':'PREPARED_GPU_NOT_RUN','panel':len(panel)}));return
    require_visibility();check(os.environ['CUDA_VISIBLE_DEVICES']==PRIMARY,'3090 only')
    check(file_hash(ROOT/CHECKPOINT/'manifest.json')==CHECKPOINT_SHA,'parent manifest drift')
    check(file_hash(Path(a.snapshot)/'model.safetensors-00001-of-00001.safetensors')==BASE_SHA,'base file drift')
    check(file_hash(ROOT/'data/g1-proof/gate-repair-v2-20261105/diagnostic.safetensors')==DIAGNOSTIC_SHA,'initial weights drift')
    import torch
    from torch.nn.attention import sdpa_kernel,SDPBackend
    from qwen_one_two.g1_proof import configure
    from qwen_one_two.g1_evidence import frozen_check
    from qwen_one_two.codec.qualify import load
    from qwen_one_two.training.reference import Reference,state_fingerprint
    from qwen_one_two.training.receiver_learning import groups,ReceiverProducer
    from qwen_one_two.training.engine import Window
    from qwen_one_two.training.episodes import collect_session
    from qwen_one_two.training.checkpoint import restore_model,save,bundle_tensors
    from qwen_one_two.training.behavior_qualification import receiver_hash,evaluate_case,suite,summarize
    from qwen_one_two.training.clean import CleanEncoder
    from qwen_one_two.training.reference_qualification import PROFILE
    configure();started=time.monotonic()
    manifest=decode((ROOT/CHECKPOINT/'manifest.json').read_bytes())
    report={'protocol':protocol,'status':'RUNNING','plan_sha256':digest(canonical(plan)),
        'gpu_uuid':PRIMARY,'measurements':[],'checkpoints':[],'optimizer_updates':0,
        'scope':'training-panel learnability, not full-catalogue training or promotion','training_performed':True}
    def persist():(out/'report.json').write_bytes(canonical(report))
    def emit(**v):print(json.dumps(v,allow_nan=False),flush=True)
    with sdpa_kernel(SDPBackend.MATH):
        b,c,_=load(a.snapshot,ROOT/'data/g1-proof/gate-repair-v2-20261105/diagnostic.safetensors')
        ref=Reference(b,c,pooler_kind='target-v2');ref.freeze();owned=groups(b,c,ref)
        class BalanceWindow(Window):
            @property
            def coefficients(self):return dict(super().coefficients,**BALANCE_OVERRIDES)
        w=(BalanceWindow if comparison else Window)(owned,'components');b.checkpointing=True
        if comparison:
            check(Window.coefficients.fget(w)['status']==1. and Window.coefficients.fget(w)['feasibility']==.5,
                  'baseline coefficient defaults changed')
        report['loss_coefficients']=dict(w.coefficients)
        restore_model(ROOT/CHECKPOINT,b,c,ref,manifest['identities'],window=w)
        check(w.scheduled_windows==w.optimizer_updates==188 and receiver_hash(b,c)==RECEIVER
              and ref.fingerprint()==REFERENCE,'restored parent state')
        producer=ReceiverProducer(b,c,ref,train)
        native=lambda:state_fingerprint({**{n:p for n,p in b.named_parameters() if n.endswith(('.R_A','.R_B'))},
            **{'codec.'+n:p for n,p in c.named_parameters() if not p.requires_grad}})
        frozen_native=native();report['initial_base']=frozen_check(b,a.snapshot)
        check(report['initial_base']['pass'] and report['initial_base']['base_tied_storage'],'initial W/tie')
        report['parent_optimizer_restored']=True
        old_sources=decode((ROOT/CHECKPOINT.parent/'source-snapshot/manifest.json').read_bytes())
        report['source_changes_since_parent']=[k for k,v in old_sources.items() if sources.get(k)!=v]
        report['migration']='new explicitly declared probe fork; original identities validated on restore; moments/RNG/counters retained; original sampler is not relabeled as probe history'
        identities=dict(manifest['identities'],protocol=digest(protocol.encode()),sources=digest(canonical(sources)))
        def frozen():
            check(ref.fingerprint()==REFERENCE and native()==frozen_native,'frozen reference/R/codec changed')
            check(all(p.grad is None for m in (b,c,ref) for p in m.parameters()),'outstanding parameter gradients')
        def checkpoint(repeat,cursor):
            frozen();path=out/f'checkpoint-{w.scheduled_windows:04d}'
            sha=save(path,w,bundle_tensors(b,c,ref),identities,
                {'protocol':protocol,'order':order,'repeat':repeat,'probe_cursor':cursor,'parent_checkpoint':str(CHECKPOINT),
                 'parent_sampler':manifest['sampler'],'initial_scheduled_windows':188,
                 'loss_coefficients':dict(w.coefficients)},reference_frozen=True,reference_id=REFERENCE)
            report['checkpoints'].append({'path':str(path.relative_to(ROOT)),'sha256':sha});return path
        def measure(repeat):
            frozen();before=receiver_hash(b,c);cases=[]
            with torch.no_grad(),(out/f'panel-{repeat:02d}.jsonl').open('x') as stream:
                for key in order:
                    s,_,_=producer(train.record(key));case=s.episode_report;cases.append(case)
                    stream.write(json.dumps(case,allow_nan=False)+'\n');stream.flush();del s
            check(receiver_hash(b,c)==before,'measurement changed receiver');frozen()
            summary=summarize_learning(train,cases,order,kind='heads')
            if comparison and repeat==0:
                check(summary==baseline['measurements'][0]['report'],'initial control head/loss mismatch')
                saved=[decode(line) for line in (Path(a.loss_balance_baseline)/'panel-00.jsonl').read_bytes().splitlines()]
                check(all(all(left[k]==right[k] for k in ('record','record_sha256','input_sha256','head','losses'))
                          for left,right in zip(cases,saved)) and len(cases)==len(saved),'initial control receipt mismatch')
                report['baseline_initial_panel_exact']=True
            report['measurements'].append({'repeat':repeat,'receiver':before,'report':summary});persist()
            emit(event='panel',repeat=repeat,operations={op:x['counts'] for op,x in summary['operations'].items()})
        try:
            checkpoint(0,0);measure(0);exposures=Counter();eligible=Counter();failures=Counter()
            with (out/'training.jsonl').open('x') as stream:
                for repeat in range(1,9):
                    for offset in range(0,32,8):
                        cases=[];t=time.monotonic()
                        for key in order[offset:offset+8]:
                            s,losses,exposure=producer(train.record(key));case=s.episode_report
                            collect_session(w,s,losses,exposure);cases.append(case);exposures[key]+=1
                            if case.get('failure'):failures[case['failure']]+=1
                            else:eligible.update(case['losses'].keys())
                            del s,losses
                        diagnostic=gradient_report(w) if repeat in (1,8) else None
                        if comparison and w.scheduled_windows==188:
                            control=decode((Path(a.loss_balance_baseline)/'training.jsonl').read_bytes().splitlines()[0])
                            check(all(all(left[k]==right[k] for k in ('record','record_sha256','input_sha256','head','losses'))
                                      for left,right in zip(cases,control['episodes'])), 'first-window control mismatch')
                            old=control['gradient_diagnostics']
                            check(diagnostic['D_cosine_with_status']==old['D_cosine_with_status'], 'initial gradient cosine mismatch')
                            check(set(diagnostic['components'])==set(old['components']), 'initial gradient eligibility mismatch')
                            for component,values in diagnostic['components'].items():
                                scale=.1 if component in BALANCE_OVERRIDES else 1.
                                check(values['eligible']==old['components'][component]['eligible'], 'initial gradient denominator mismatch')
                                for name in ('weighted_norm','weighted_D_norm'):
                                    check(math.isclose(values[name],old['components'][component][name]*scale,
                                                       rel_tol=1e-12,abs_tol=1e-12), 'initial gradient scaling mismatch')
                            report['baseline_initial_gradient_norms_match']=True
                        result=w.step();entry={'repeat':repeat,'offset':offset,'window':result,'episodes':cases,
                            'gradient_diagnostics':diagnostic,'seconds':time.monotonic()-t}
                        stream.write(json.dumps(entry,allow_nan=False)+'\n');stream.flush()
                        report.update(optimizer_updates=w.optimizer_updates-188,scheduled_probe_windows=w.scheduled_windows-188,
                                      exposures=dict(exposures),eligible_examples=dict(eligible),failed_exposures=dict(failures));persist()
                        emit(event='update',repeat=repeat,window=w.scheduled_windows,updates=w.optimizer_updates-188,seconds=entry['seconds'])
                        if w.scheduled_windows==189:
                            path=checkpoint(repeat,8);before=receiver_hash(b,c)
                            moments=state_fingerprint({str(i)+'.'+k:v for i,state in enumerate(w.optimizer.state.values()) for k,v in state.items()})
                            cuda_rng=torch.cuda.get_rng_state(0);cpu_rng=torch.get_rng_state();python_rng=random.getstate()
                            restore_model(path,b,c,ref,identities,window=w)
                            check(receiver_hash(b,c)==before and moments==state_fingerprint({str(i)+'.'+k:v for i,state in enumerate(w.optimizer.state.values()) for k,v in state.items()}),'first-update roundtrip')
                            check(torch.equal(cuda_rng,torch.cuda.get_rng_state(0)) and torch.equal(cpu_rng,torch.get_rng_state())
                                  and python_rng==random.getstate(),'first-update RNG roundtrip')
                            report['first_update_roundtrip_exact']=True
                    if repeat in (1,2,4,8):checkpoint(repeat,repeat*32);measure(repeat)
            # Actual feedback decoding/delivery, on held-out data only after updates.
            b.checkpointing=False
            encoder=CleanEncoder(b,ref,checkpoint_hash=receiver_hash(b,c),profile_hash=digest(PROFILE.encode()),cache_references=True)
            encoder.reference_cache=producer.encoder.reference_cache
            eids=[e['id'] for e in dev.manifest['records'] if e['selected'] and dev.record(e['id']).get('evaluation_slice')=='E']
            pairs,omitted=suite(dev);xids=[k for pair in pairs for k in pair['ids']]
            for panel_name,keys in (('E',eids),('X',xids)):
                cases=[]
                with torch.no_grad(),(out/f'development-{panel_name}.jsonl').open('x') as stream:
                    for key in keys:
                        case=evaluate_case(b,c,ref,dev,dev.record(key),encoder);cases.append(case)
                        stream.write(json.dumps(case,allow_nan=False)+'\n');stream.flush()
                        emit(event='development',panel=panel_name,id=key,passed=case['score']['passed'])
                report['development_'+panel_name]=summarize_learning(dev,cases,keys,kind='behavior')
                if panel_name=='X':report['interventions']=summarize(dev,cases,unchanged=True)
                persist()
            report['omitted_reader_ids']=omitted;report['final_base']=frozen_check(b,a.snapshot);frozen()
            check(report['final_base']['pass'] and report['final_base']['base_tied_storage'],'final W/tie')
            check(sum(exposures.values())==256 and len(exposures)==32 and set(exposures.values())=={8},'complete probe exposure accounting')
            report.update(status='COMPLETED_PROBE_NOT_PROMOTED',receiver_after=receiver_hash(b,c),reference_unchanged=True,
                frozen_native_unchanged=True,gradients_none=True,seconds=time.monotonic()-started,peak_cuda_bytes=torch.cuda.max_memory_allocated())
        except BaseException as error:
            report.update(status='FAILED_PROBE',error=repr(error),seconds=time.monotonic()-started);raise
        finally:persist()
    emit(status=report['status'],optimizer_updates=report['optimizer_updates'])


if __name__=='__main__':main()
