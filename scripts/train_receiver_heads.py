"""Frozen-representation Noul/Score readout diagnostic; explicitly GPU-only."""
import argparse
from collections import Counter
import json
import random
from pathlib import Path
import shutil
import time
from qwen_one_two.training.inputs import canonical,check,digest
from qwen_one_two.training.manifest import file_hash,BASE_SHA,DIAGNOSTIC_SHA
from qwen_one_two.training.two_gpu import ROOT,CHECKPOINT,CHECKPOINT_SHA,RECEIVER,REFERENCE,source_hashes
from qwen_one_two.training.release import load_release,decode
from qwen_one_two.training.learning_report import probe_panel,summarize_learning
from qwen_one_two.training.behavior_qualification import expected
from scripts.train_receiver_probe import loss_balance_baseline
from scripts.fit_reference import TRAIN_SHA,DEV_SHA
from qwen_one_two.g1_device import PRIMARY,require_visibility

PROTOCOL='receiver-head-only-v1'
HEADS={'codec.prototypes','codec.score_row'}
MEASURE=(0,1,2,4,8,16,32,64,128,256,453)


def head_groups(groups):
    result=[dict(g,named_parameters=[(n,p) for n,p in g['named_parameters'] if n in HEADS]) for g in groups]
    result=[g for g in result if g['named_parameters']]
    names=[n for g in result for n,_ in g['named_parameters']]
    check(set(names)==HEADS and len(names)==2,'exact two-head ownership required')
    return result


def raw_readout(codec,h,operation):
    """The original readout arithmetic, without unrelated frozen Reason heads."""
    if operation=='Noul':
        logits=(codec.qev_k(codec.prototypes)*codec.qev_q(h.float())).sum(-1)/16
        return logits[1]-logits[0]
    check(operation=='Score','scalar readout operation')
    return codec.table.scalar(248078,h)


def readout_losses(codec,h,row):
    from qwen_one_two.training.losses import binary,regression
    # No labels are used until the actual fixed-feature readout has executed.
    if row['operation']=='Choice':return {}
    raw=raw_readout(codec,h,row['operation']);gold=expected(row)
    if gold['status']!='Assessed':return {}
    if row['operation']=='Noul':return {'noul':binary(raw,gold['payload']['p_true'])}
    return {'score':regression(raw,interval=tuple(gold['payload']['score_interval']))}


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--snapshot',required=True)
    ap.add_argument('--output',required=True);ap.add_argument('--execute-gpu',action='store_true');a=ap.parse_args()
    out=Path(a.output).resolve();check(out.is_relative_to(ROOT/'data') and not out.exists(),'fresh data/ output required')
    out.mkdir(parents=True)
    train=load_release(ROOT/'data/training-collections-20261105/releases/train',expected_hash=TRAIN_SHA)
    dev=load_release(ROOT/'data/training-collections-20261105/releases/development',expected_hash=DEV_SHA)
    panel=probe_panel(train);order=[e['id'] for e in panel]
    families={dev.record(e['id'])['family'] for e in dev.manifest['records'] if e['selected'] and dev.record(e['id']).get('evaluation_slice') in ('E','X')}
    check(not families&{e['family'] for e in panel},'development family overlap')
    sources=source_hashes();sources['scripts/train_receiver_probe.py']=file_hash(ROOT/'scripts/train_receiver_probe.py')
    binding,control=loss_balance_baseline(ROOT/'data/receiver-probe-20261107/run',panel,sources)
    sources['scripts/train_receiver_heads.py']=file_hash(Path(__file__))
    historical=decode((ROOT/'docs/evidence/receiver-training-probe-v1.json').read_bytes())['historical_development']
    for name,sha in historical['source_hashes'].items():check(file_hash(ROOT/name)==sha,'parent development receipt drift')
    plan={'protocol':PROTOCOL,'sources':sources,'parent_checkpoint':str(CHECKPOINT),'parent_manifest_sha256':CHECKPOINT_SHA,
          'receiver':RECEIVER,'reference':REFERENCE,'dataset':TRAIN_SHA,'development':DEV_SHA,'panel':panel,
          'heads':sorted(HEADS),'initial_window':188,'final_window':2000,'updates':1812,'repeats':453,
          'measure_repeats':MEASURE,'optimizer':'project original head moments/settings; retain A4 schedule at 188',
          'joint_control':{k:binding[k] for k in ('baseline','plan_sha256','report_sha256')},
          'development_baseline':historical,'snapshot':str(Path(a.snapshot).resolve()),
          'scope':'fixed-feature head learning only; cached visits are not causal episodes; no native capability training'}
    (out/'plan.json').write_bytes(canonical(plan))
    for name in sources:
        dst=out/'source'/name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,dst)
    if not a.execute_gpu:print(json.dumps({'status':'PREPARED_GPU_NOT_RUN','updates':1812}));return
    require_visibility()
    check(file_hash(ROOT/CHECKPOINT/'manifest.json')==CHECKPOINT_SHA,'parent manifest drift')
    check(file_hash(Path(a.snapshot)/'model.safetensors-00001-of-00001.safetensors')==BASE_SHA,'base file drift')
    diagnostic=ROOT/'data/g1-proof/gate-repair-v2-20261105/diagnostic.safetensors'
    check(file_hash(diagnostic)==DIAGNOSTIC_SHA,'diagnostic file drift')
    import torch
    from safetensors.torch import save_file
    from torch.nn.attention import sdpa_kernel,SDPBackend
    from qwen_one_two.g1_proof import configure
    from qwen_one_two.g1_evidence import frozen_check
    from qwen_one_two.codec.qualify import load
    from qwen_one_two.codec.schema import Port
    from qwen_one_two.training.reference import Reference,state_fingerprint
    from qwen_one_two.training.receiver_learning import groups,ReceiverProducer
    from qwen_one_two.training.engine import Window,clone_state
    from qwen_one_two.training.checkpoint import restore_model,save,bundle_tensors
    from qwen_one_two.training.behavior_qualification import receiver_hash,evaluate_case,suite,summarize
    from qwen_one_two.training.clean import CleanEncoder
    from qwen_one_two.training.component_producer import component_view
    from qwen_one_two.training.reference_qualification import PROFILE
    configure();started=time.monotonic()
    r={'protocol':PROTOCOL,'status':'RUNNING','gpu_uuid':PRIMARY,'measurements':[],'checkpoints':[],
       'optimizer_updates':0,'plan_sha256':file_hash(out/'plan.json'),'training_performed':True}
    def persist():(out/'report.json').write_bytes(canonical(r))
    def emit(**entry):print(json.dumps(entry,allow_nan=False),flush=True)
    with sdpa_kernel(SDPBackend.MATH):
        b,c,_=load(a.snapshot,diagnostic);ref=Reference(b,c,pooler_kind='target-v2');ref.freeze()
        original=groups(b,c,ref);full=Window(original,'components')
        parent=decode((ROOT/CHECKPOINT/'manifest.json').read_bytes())
        restore_model(ROOT/CHECKPOINT,b,c,ref,parent['identities'],window=full)
        check(receiver_hash(b,c)==RECEIVER and ref.fingerprint()==REFERENCE and full.scheduled_windows==188,'restored parent')
        owned=head_groups(original);moments={p:clone_state(full.optimizer.state[p]) for g in owned for _,p in g['named_parameters']}
        before_moments=state_fingerprint({str(i)+'.'+k:v for i,state in enumerate(moments.values()) for k,v in state.items()})
        b.requires_grad_(False);c.requires_grad_(False)
        for g in owned:
            for _,p in g['named_parameters']:p.requires_grad_(True)
        w=Window(owned,'components');w.optimizer.state.update(moments);w.scheduled_windows=188;w.optimizer_updates=188
        check(before_moments==state_fingerprint({str(i)+'.'+k:v for i,state in enumerate(w.optimizer.state.values()) for k,v in state.items()}),'projected head moments changed')
        del full,moments,original
        b.checkpointing=False
        check({n for n,p in c.named_parameters() if p.requires_grad}=={'prototypes','score_row'} and
              all(not p.requires_grad for p in b.parameters()) and all(not p.requires_grad for p in ref.parameters()),'head-only freeze')
        r['ownership']=w.ownership;r['projected_moments_exact']=True;r['initial_base']=frozen_check(b,a.snapshot)
        def fixed_state():
            return state_fingerprint({**{'backbone.'+n:p for n,p in b.named_parameters() if n.endswith(('.R_A','.R_B','.D_A','.D_B'))},
                **{'codec.'+n:p for n,p in c.state_dict(keep_vars=True).items() if 'codec.'+n not in HEADS}})
        fixed=fixed_state();r['frozen_representation']=fixed
        def frozen():
            check(fixed_state()==fixed and ref.fingerprint()==REFERENCE,'frozen representation/reference changed')
            check(all(p.grad is None for m in (b,c,ref) for p in m.parameters()),'outstanding gradients')
        identities=dict(parent['identities'],protocol=digest(PROTOCOL.encode()),sources=digest(canonical(sources)))
        def checkpoint(repeat):
            frozen();path=out/f'checkpoint-{w.scheduled_windows:04d}'
            sha=save(path,w,bundle_tensors(b,c,ref),identities,{'protocol':PROTOCOL,'order':order,'repeat':repeat,
                'parent_checkpoint':str(CHECKPOINT),'frozen_representation':fixed,'head_only':True},reference_frozen=True,reference_id=REFERENCE)
            r['checkpoints'].append({'path':str(path.relative_to(ROOT)),'sha256':sha});return path
        cache={};producer=ReceiverProducer(b,c,ref,train)
        ports={'Noul':Port.NOUL,'Choice':Port.CHOICE,'Score':Port.SCORE}
        def extract(key):
            captured=[]
            hook=c.qev_q.register_forward_pre_hook(lambda module,args:captured.append(args[0].detach().clone()))
            try:
                with torch.no_grad():s,_,_=producer(train.record(key))
            finally:hook.remove()
            case=s.episode_report
            check(not case.get('failure') and len(captured)==1,'complete feature extraction')
            h=captured[0];check(torch.equal(h,s.hidden[0,-1]),'feature boundary drift')
            packet=s.runtime.call.request
            candidates=torch.stack([s.hidden[0,i] for _,i in packet.candidate_ends]).detach().clone() if packet.candidate_ends else None
            identities_local=component_view(train,train.record(key))[2]['identities']
            del s
            return {'h':h,'candidates':candidates,'identities':identities_local,'case':case}
        def cached_case(key):
            item=cache[key];row=train.record(key)
            evaluated=c.evaluate(item['h'],ports[row['operation']],item['candidates'])
            head={'status':item['case']['head']['status']}
            from qwen_one_two.training.episodes import STATUS_NAMES
            head['status']=STATUS_NAMES[int(evaluated['status_logits'].argmax())+1]
            if row['operation']=='Choice':head['selected']=item['identities'][int(evaluated['pointer_logits'].argmax())]
            else:
                raw=raw_readout(c,item['h'],row['operation']);check(torch.equal(raw,evaluated['raw']),'minimal readout forward mismatch')
                head.update(raw=float(raw.detach()),value=float(evaluated['value'].detach()))
            losses=readout_losses(c,item['h'],row)
            return {'record':key,'record_sha256':item['case']['record_sha256'],'input_sha256':item['case']['input_sha256'],
                    'head':head,'losses':{n:float(v.detach()) for n,v in losses.items()},
                    'summary':{'terminal':'CachedReadout'},'feature_origin':'features.jsonl',
                    'new_causal_positions':0,'new_prefills':0}
        def measure(repeat):
            frozen()
            with torch.no_grad():cases=[cached_case(k) for k in order]
            (out/f'panel-{repeat:03d}.jsonl').write_text(''.join(json.dumps(v,allow_nan=False)+'\n' for v in cases))
            summary=summarize_learning(train,cases,order,kind='heads')
            r['measurements'].append({'repeat':repeat,'updates':w.optimizer_updates-188,'receiver':receiver_hash(b,c),'report':summary});persist()
            emit(event='panel',repeat=repeat,updates=w.optimizer_updates-188,operations={op:x['counts'] for op,x in summary['operations'].items()})
        try:
            check(r['initial_base']['pass'] and r['initial_base']['base_tied_storage'],'initial W/tie')
            checkpoint(0)
            controls={x['record']:x for x in map(decode,(ROOT/'data/receiver-probe-20261107/run/panel-00.jsonl').read_bytes().splitlines())}
            with (out/'features.jsonl').open('x') as stream:
                for key in order:
                    item=extract(key);case=item['case'];cache[key]=item
                    check(all(case[n]==controls[key][n] for n in ('head','losses','input_sha256','record_sha256')),'initial control mismatch')
                    stream.write(json.dumps(case,allow_nan=False)+'\n');stream.flush()
            tensors={key+'.h':v['h'] for key,v in cache.items()}
            tensors.update({key+'.candidates':v['candidates'] for key,v in cache.items() if v['candidates'] is not None})
            save_file(tensors,str(out/'features.safetensors'));feature_hash=state_fingerprint(tensors)
            r.update(initial_control_exact=True,feature_sha256=file_hash(out/'features.safetensors'),feature_tensor_fingerprint=feature_hash)
            # Compare actual codec and minimal-head VJPs without any update.
            parity={}
            for op in ('Noul','Score'):
                key=next(k for k in order if train.record(k)['operation']==op and expected(train.record(k))['status']=='Assessed');item=cache[key]
                raw=c.evaluate(item['h'],ports[op])['raw'];gold=expected(train.record(key))['payload']
                from qwen_one_two.training.losses import binary,regression
                loss=binary(raw,gold['p_true']) if op=='Noul' else regression(raw,interval=tuple(gold['score_interval']))
                actual=torch.autograd.grad(loss,w.parameters,allow_unused=True)
                fast=torch.autograd.grad(next(iter(readout_losses(c,item['h'],train.record(key)).values())),w.parameters,allow_unused=True)
                check(all((x is None and y is None) or (x is not None and y is not None and torch.equal(x,y)) for x,y in zip(actual,fast)),'head VJP mismatch')
                parity[op]=True
            r['minimal_readout_gradients_exact']=parity;measure(0)
            visits=Counter();eligible=Counter()
            with (out/'training.jsonl').open('x') as stream:
                for repeat in range(1,454):
                    for offset in range(0,32,8):
                        examples=[]
                        for key in order[offset:offset+8]:
                            losses=readout_losses(c,cache[key]['h'],train.record(key));eligible.update(losses.keys())
                            examples.append({'record':key,'losses':{n:float(v.detach()) for n,v in losses.items()}})
                            w.add_episode(losses,complete=True,positions=0);visits[key]+=1
                        result=w.step();stream.write(json.dumps({'repeat':repeat,'offset':offset,'window':result,'visits':examples},allow_nan=False)+'\n')
                        if w.scheduled_windows==189:
                            path=checkpoint(repeat);before=receiver_hash(b,c)
                            moment_hash=state_fingerprint({str(i)+'.'+k:v for i,state in enumerate(w.optimizer.state.values()) for k,v in state.items()})
                            rng=(torch.cuda.get_rng_state(0),torch.get_rng_state(),random.getstate())
                            restore_model(path,b,c,ref,identities,window=w)
                            check(receiver_hash(b,c)==before and moment_hash==state_fingerprint({str(i)+'.'+k:v for i,state in enumerate(w.optimizer.state.values()) for k,v in state.items()}) and
                                  torch.equal(rng[0],torch.cuda.get_rng_state(0)) and torch.equal(rng[1],torch.get_rng_state()) and rng[2]==random.getstate(),'first-update roundtrip')
                            r['first_update_roundtrip_exact']=True
                    if repeat in MEASURE:
                        stream.flush();checkpoint(repeat);measure(repeat)
                    r.update(optimizer_updates=w.optimizer_updates-188,replays=dict(visits),eligible_examples=dict(eligible));persist()
            check(w.scheduled_windows==2000 and w.optimizer_updates-188==1812 and set(visits.values())=={453},'complete head schedule')
            check(state_fingerprint(tensors)==feature_hash,'cached features changed')
            # Fresh causal extraction proves the replayed head outputs remain real.
            fresh=[]
            with (out/'fresh-final.jsonl').open('x') as stream:
                for key in order:
                    item=extract(key);old=cache[key];case=item['case']
                    check(torch.equal(item['h'],old['h']) and ((item['candidates'] is None and old['candidates'] is None) or torch.equal(item['candidates'],old['candidates'])),'frozen features changed on fresh extraction')
                    with torch.no_grad():cached=cached_case(key)
                    check(case['head']==cached['head'] and all(case['losses'][n]==v for n,v in cached['losses'].items()),'fresh/cached final readout mismatch')
                    fresh.append(case);stream.write(json.dumps(case,allow_nan=False)+'\n');stream.flush()
            r['fresh_final_matches_cached']=True
            encoder=CleanEncoder(b,ref,checkpoint_hash=receiver_hash(b,c),profile_hash=digest(PROFILE.encode()),cache_references=True)
            encoder.reference_cache=producer.encoder.reference_cache
            eids=[e['id'] for e in dev.manifest['records'] if e['selected'] and dev.record(e['id']).get('evaluation_slice')=='E']
            pairs,omitted=suite(dev);xids=[k for pair in pairs for k in pair['ids']];all_cases=[]
            for name,keys in (('E',eids),('X',xids)):
                cases=[]
                with torch.no_grad(),(out/f'development-{name}.jsonl').open('x') as stream:
                    for key in keys:
                        case=evaluate_case(b,c,ref,dev,dev.record(key),encoder);cases.append(case);all_cases.append(case)
                        stream.write(json.dumps(case,allow_nan=False)+'\n');stream.flush();emit(event='development',panel=name,id=key,passed=case['score']['passed'])
                r['development_'+name]=summarize_learning(dev,cases,keys,kind='behavior')
                if name=='X':r['interventions']=summarize(dev,cases,unchanged=True)
                persist()
            old_cases={x['id']:x for name in historical['source_hashes'] for x in map(decode,(ROOT/name).read_bytes().splitlines())}
            choice=[x for x in all_cases if dev.record(x['id'])['operation']=='Choice']
            check(len(choice)==90 and all(all(x[n]==old_cases[x['id']][n] for n in ('actual','predicted_status','view_sha256','score','failure')) for x in choice),'Choice retention mismatch')
            r['choice_retention_exact']=90;r['combined_development']=summarize_learning(dev,all_cases,eids+xids,kind='behavior')
            r['final_base']=frozen_check(b,a.snapshot);frozen()
            check(r['final_base']['pass'] and r['final_base']['base_tied_storage'],'final W/tie')
            r.update(status='COMPLETED_HEAD_ONLY_NOT_PROMOTED',receiver_after=receiver_hash(b,c),frozen_unchanged=True,reference_unchanged=True,
                     cached_features_unchanged=True,gradients_none=True,seconds=time.monotonic()-started,peak_cuda_bytes=torch.cuda.max_memory_allocated())
        except BaseException as exc:r.update(status='FAILED_HEAD_ONLY',error=repr(exc),seconds=time.monotonic()-started);raise
        finally:persist()
    emit(status=r['status'],updates=r['optimizer_updates'])


if __name__=='__main__':main()
