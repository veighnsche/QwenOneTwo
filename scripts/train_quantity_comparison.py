"""Two matched receiver fits on existing counts; development only after both fits."""
import argparse
from collections import Counter
import json
from pathlib import Path
import random
import shutil
import time
from qwen_one_two.g1_device import PRIMARY,require_visibility
from qwen_one_two.training.inputs import canonical,check,digest
from qwen_one_two.training.manifest import file_hash,BASE_SHA,DIAGNOSTIC_SHA
from qwen_one_two.training.two_gpu import ROOT,CHECKPOINT,CHECKPOINT_SHA,RECEIVER,REFERENCE,source_hashes
from qwen_one_two.training.release import load_release,decode
from qwen_one_two.training.quantity_learning import PROTOCOL,PASSES,COUNT_SCALE,training_panel,quantity_summary,quantity_receipt
from qwen_one_two.training.numeric_access import ARMS,NumericConstruction
from qwen_one_two.training.behavior_qualification import suite,PROFILE
from scripts.fit_reference import TRAIN_SHA,DEV_SHA


def check_joint_vjp(window,terms):
    """One post-update graph, separate versus combined VJPs; no replay/update."""
    import torch
    parts=[torch.autograd.grad(loss,window.parameters,allow_unused=True,retain_graph=True) for loss in terms]
    joint=torch.autograd.grad(sum(terms),window.parameters,allow_unused=True,retain_graph=True)
    worst=0.;min_cosine=1.;count=0
    for values,total in zip(zip(*parts),joint):
        active=[x for x in values if x is not None]
        check(bool(active)==(total is not None),'VJP connectivity agreement')
        if not active:continue
        reference=sum(active).double();candidate=total.double()
        rms=reference.square().mean().sqrt();crms=candidate.square().mean().sqrt()
        error=float((candidate-reference).square().mean().sqrt()/rms.clamp_min(1e-10))
        check(error<=1e-4,'joint VJP FP32 relative RMS tolerance');worst=max(worst,error);count+=1
        if min(float(rms),float(crms))>1e-10:
            cosine=float((candidate*reference).mean()/(rms*crms));min_cosine=min(min_cosine,cosine)
            check(cosine>=.99999,'joint VJP FP32 cosine tolerance')
    quantity_norms={prefix:float(torch.stack([g.double().square().sum() for n,g in zip(window.names,parts[1])
                   if g is not None and predicate(n)]).sum().sqrt()) for prefix,predicate in
                   [('D',lambda n:'.D_' in n),('atom',lambda n:n.startswith('codec.atom.')),
                    ('readout',lambda n:n.startswith('codec.quantity_probe.'))]}
    check(all(v>1e-10 for v in quantity_norms.values()),'connected quantity-only D/atom/readout gradients')
    return {'parameters':count,'max_relative_rms':worst,'min_cosine':min_cosine,
            'quantity_only_norms':quantity_norms,'same_graph_no_replay':True}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--snapshot',required=True);ap.add_argument('--output',required=True)
    ap.add_argument('--execute-gpu',action='store_true');a=ap.parse_args()
    out=Path(a.output).resolve();check(out.is_relative_to(ROOT/'data') and not out.exists(),'fresh data output')
    out.mkdir(parents=True)
    train=load_release(ROOT/'data/training-collections-20261105/releases/train',expected_hash=TRAIN_SHA)
    dev=load_release(ROOT/'data/training-collections-20261105/releases/development',expected_hash=DEV_SHA)
    panel=training_panel(train);order=[r['id'] for r in panel];exposure_order=order*PASSES
    pairs,omitted=suite(dev);xids=[k for p in pairs for k in p['ids']]
    eids=[e['id'] for e in dev.manifest['records'] if e['selected'] and dev.record(e['id']).get('evaluation_slice')=='E']
    check(len(eids)==130 and len(xids)==180,'complete held-out E/X')
    check(not {r['family'] for r in panel}&{dev.record(k)['family'] for k in eids+xids},'held-out family overlap')
    check(not set(order)&set(eids+xids),'held-out record overlap')
    check(len(exposure_order)==1016 and len(exposure_order)%8==0,'four full passes, no padding')
    sources=source_hashes()
    for n in ('scripts/train_quantity_comparison.py','scripts/train_receiver_probe.py','scripts/fit_reference.py',
              'docs/quantity-receiver-learning-v1.md','tests/test_quantity_learning.py'):
        sources[n]=file_hash(ROOT/n)
    plan={'protocol':PROTOCOL,'sources':sources,'parent_checkpoint':str(CHECKPOINT),'parent_manifest_sha256':CHECKPOINT_SHA,
          'receiver':RECEIVER,'reference':REFERENCE,'training':TRAIN_SHA,'development':DEV_SHA,
          'gpu_uuid':PRIMARY,'snapshot':str(Path(a.snapshot).resolve()),'panel':panel,'passes':PASSES,
          'exposures_per_arm':1016,'windows_per_arm':127,'schedule_positions':[189,315],
          'arms':list(ARMS),'count_scale':COUNT_SCALE,'coefficients':{'score':1.,'quantities':1.},
          'initialization':'parent receiver and moments, zero auxiliary readout/empty auxiliary moments; identical RNG and checkpoint for both arms',
          'development_after_both_fits':True,'development_E_source_only':eids,'development_X':pairs,
          'omitted_native_reader':omitted,'new_data':False,'scope':'count access and existing completion operation; unseen operators are transfer only'}
    (out/'plan.json').write_bytes(canonical(plan))
    for n in sources:
        dst=out/'source'/n;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/n,dst)
    if not a.execute_gpu:print('PREPARED_GPU_NOT_RUN');return
    require_visibility()
    check(file_hash(ROOT/CHECKPOINT/'manifest.json')==CHECKPOINT_SHA,'parent checkpoint identity')
    check(file_hash(Path(a.snapshot)/'model.safetensors-00001-of-00001.safetensors')==BASE_SHA,'base artifact')
    diagnostic=ROOT/'data/g1-proof/gate-repair-v2-20261105/diagnostic.safetensors'
    check(file_hash(diagnostic)==DIAGNOSTIC_SHA,'diagnostic artifact')
    import torch
    from torch.nn.attention import sdpa_kernel,SDPBackend
    from qwen_one_two.g1_proof import configure
    from qwen_one_two.g1_evidence import frozen_check
    from qwen_one_two.codec.qualify import load
    from qwen_one_two.codec.neural import norm
    from qwen_one_two.training.reference import Reference,state_fingerprint
    from qwen_one_two.training.receiver_learning import groups
    from qwen_one_two.training.quantity_learning import QuantityProducer
    from qwen_one_two.training.engine import Window,clone_state
    from qwen_one_two.training.episodes import collect_session
    from qwen_one_two.training.checkpoint import restore_model,save,bundle_tensors
    from qwen_one_two.training.behavior_qualification import receiver_hash,evaluate_case,summarize
    from qwen_one_two.training.clean import CleanEncoder
    from qwen_one_two.training.information_loss import CaptureRequest
    from qwen_one_two.training.learning_report import summarize_learning
    from qwen_one_two.training.direct_readout import paired_values
    from scripts.train_receiver_probe import gradient_report
    configure();torch.set_num_threads(1);started=time.monotonic()
    class QuantityWindow(Window):
        @property
        def coefficients(self):return {'quantity_score':1.}
    report={'protocol':PROTOCOL,'status':'RUNNING','plan_sha256':file_hash(out/'plan.json'),'arms':{},'gpu_uuid':PRIMARY}
    def persist():(out/'report.json').write_bytes(canonical(report))
    def emit(**v):print(json.dumps(v,allow_nan=False),flush=True)
    def write(stream,case):stream.write(json.dumps(case,allow_nan=False)+'\n');stream.flush()
    with sdpa_kernel(SDPBackend.MATH):
        try:
            b,c,_=load(a.snapshot,diagnostic);ref=Reference(b,c,pooler_kind='target-v2');ref.freeze()
            owned=groups(b,c,ref);parent_window=Window(owned,'components')
            parent=decode((ROOT/CHECKPOINT/'manifest.json').read_bytes())
            restore_model(ROOT/CHECKPOINT,b,c,ref,parent['identities'],window=parent_window)
            check(receiver_hash(b,c)==RECEIVER and ref.fingerprint()==REFERENCE,'parent receiver/reference')
            check(parent_window.scheduled_windows==parent_window.optimizer_updates==188,'parent counters')
            c.quantity_probe=torch.nn.Linear(1024,2,device='cuda',dtype=torch.float32)
            torch.nn.init.zeros_(c.quantity_probe.weight);torch.nn.init.zeros_(c.quantity_probe.bias)
            auxiliary={'name':'quantity_probe','named_parameters':[(f'codec.quantity_probe.{n}',p) for n,p in c.quantity_probe.named_parameters()],
                       'lr':2e-4,'weight_decay':0.}
            w=QuantityWindow(owned+[auxiliary],'components')
            # Existing owners keep the exact saved moments. Only the explicitly
            # new readout starts empty; neither arm resets the existing schedule.
            for p,state in parent_window.optimizer.state.items():w.optimizer.state[p]=clone_state(state)
            for old,new in zip(parent_window.optimizer.param_groups,w.optimizer.param_groups):new['lr']=old['lr']
            w.scheduled_windows=w.optimizer_updates=188
            del parent_window
            identities=dict(parent['identities'],protocol=digest(PROTOCOL.encode()),sources=digest(canonical(sources)))
            frozen_weights={**{n:p for n,p in b.named_parameters() if n.endswith(('.R_A','.R_B'))},
                            **{'codec.'+n:p for n,p in c.named_parameters() if not p.requires_grad}}
            frozen_hash=state_fingerprint(frozen_weights)
            def frozen():
                check(ref.fingerprint()==REFERENCE and state_fingerprint(frozen_weights)==frozen_hash,'frozen reference/R/nonreceiver')
                check(all(p.grad is None for m in (b,c,ref) for p in m.parameters()),'gradients cleared')
            def moments():
                return state_fingerprint({str(i)+'.'+k:v for i,p in enumerate(w.parameters)
                                          for k,v in w.optimizer.state.get(p,{}).items()})
            def checkpoint(path,cursor,arm):
                frozen()
                sha=save(path,w,bundle_tensors(b,c,ref),identities,
                    {'protocol':PROTOCOL,'order':order,'passes':PASSES,'cursor':cursor,'arm':arm,
                     'parent_checkpoint':str(CHECKPOINT),'parent_sampler':parent['sampler'],
                     'initial_scheduled_windows':188,'coefficients':plan['coefficients']},
                    reference_frozen=True,reference_id=REFERENCE)
                return {'path':str(path.relative_to(ROOT)),'sha256':sha,'cursor':cursor}
            report['initial_base']=frozen_check(b,a.snapshot)
            check(report['initial_base']['pass'] and report['initial_base']['base_tied_storage'],'initial W/tie')
            report['initial_checkpoint']=checkpoint(out/'initial',0,'shared')
            initial=(receiver_hash(b,c),moments(),torch.cuda.get_rng_state(0),torch.get_rng_state(),random.getstate())
            cache={};initial_capture=None;initial_receipt=None
            for arm in ARMS:
                armout=out/arm;armout.mkdir();restore_model(out/'initial',b,c,ref,identities,window=w)
                check((receiver_hash(b,c),moments())==initial[:2] and torch.equal(initial[2],torch.cuda.get_rng_state(0))
                      and torch.equal(initial[3],torch.get_rng_state()) and initial[4]==random.getstate(),'matched initial model/moments/RNG')
                item={'initial_identical':True,'measurements':{},'checkpoints':[],'updates':0,'failed_exposures':{}}
                report['arms'][arm]=item;b.checkpointing=True
                producer=QuantityProducer(b,c,ref,train,arm);producer.encoder.reference_cache=cache
                def measure(stage):
                    nonlocal initial_capture,initial_receipt
                    before=receiver_hash(b,c);cases=[]
                    with torch.no_grad(),(armout/f'train-{stage}.jsonl').open('x') as stream:
                        for i,key in enumerate(order):
                            if i==0 and stage=='initial':
                                with CaptureRequest(c) as cap:s,_,_=producer(train.record(key))
                                receipt=s.episode_report
                                if initial_capture is None:
                                    initial_capture=cap.data;initial_receipt=receipt
                                else:
                                    from qwen_one_two.codec.schema import Role
                                    skip={int(Role.ATTRIBUTE_DESC),int(Role.INT64),int(Role.UTF8)}
                                    mask=[i for i,r in enumerate(cap.data['roles']) if r not in skip]
                                    check(torch.equal(initial_capture['wire'][mask],cap.data['wire'][mask]),'matched nonattribute initial wire')
                                    check(receipt['construction_audit']==initial_receipt['construction_audit'],'matched initial source/reference tensors')
                                    check(receipt['summary']==initial_receipt['summary'],'matched initial complete ledger')
                                    report['initial_nonattribute_wire_sources_references_accounting_exact']=True
                            else:s,_,_=producer(train.record(key))
                            cases.append(s.episode_report);write(stream,s.episode_report)
                            del s;producer.loss_terms=None
                    check(receiver_hash(b,c)==before,'measurement modified model');frozen()
                    item['measurements'][stage]={'learning':summarize_learning(train,cases,order,kind='heads'),
                                                'quantities':quantity_summary(cases)}
                    persist();emit(event='training_measurement',arm=arm,stage=stage,quantities=quantity_summary(cases))
                measure('initial');exposures=Counter();failed=Counter()
                with (armout/'training.jsonl').open('x') as stream:
                    for offset in range(0,len(exposure_order),8):
                        cases=[];t=time.monotonic()
                        for key in exposure_order[offset:offset+8]:
                            s,losses,exposure=producer(train.record(key))
                            if offset==8 and not cases:item['joint_vjp_check']=check_joint_vjp(w,producer.loss_terms)
                            collect_session(w,s,losses,exposure)
                            cases.append(s.episode_report);exposures[key]+=1
                            if s.episode_report.get('failure'):failed[s.episode_report['failure']]+=1
                            del s,losses;producer.loss_terms=None
                        diagnostic=gradient_report(w) if offset in (0,8,1008) else None
                        if offset==8:
                            buffers=w.buffers['quantity_score']
                            norms={prefix:float(torch.stack([g.double().square().sum() for i,g in buffers.items()
                                   if predicate(w.names[i])]).sum().sqrt()) for prefix,predicate in
                                   [('D',lambda n:'.D_' in n),('atom',lambda n:n.startswith('codec.atom.')),
                                    ('quantity',lambda n:n.startswith('codec.quantity_probe.')),('score',lambda n:n=='codec.score_row')]}
                            check(all(v>0 for v in norms.values()),'connected nonzero D/atom/count/Score gradients')
                            item['connected_gradient_norms']=norms
                        step=w.step();write(stream,{'offset':offset,'window':step,'episodes':cases,'gradient_diagnostics':diagnostic,
                                                   'seconds':time.monotonic()-t})
                        item.update(updates=w.optimizer_updates-188,scheduled_windows=w.scheduled_windows-188,
                                    failed_exposures=dict(failed),exposures=dict(exposures));persist()
                        emit(event='update',arm=arm,window=w.scheduled_windows,of=315,seconds=time.monotonic()-t)
                        if offset==0 or (offset+8)%256==0 or offset+8==len(exposure_order):
                            path=armout/f'checkpoint-{w.scheduled_windows:04d}'
                            item['checkpoints'].append(checkpoint(path,offset+8,arm));persist()
                            if offset==0:
                                before=(receiver_hash(b,c),moments(),torch.cuda.get_rng_state(0),torch.get_rng_state(),random.getstate())
                                restore_model(path,b,c,ref,identities,window=w)
                                check((receiver_hash(b,c),moments())==before[:2] and torch.equal(before[2],torch.cuda.get_rng_state(0))
                                      and torch.equal(before[3],torch.get_rng_state()) and before[4]==random.getstate(),'first update checkpoint roundtrip')
                                item['first_update_roundtrip_exact']=True
                check(len(exposures)==254 and set(exposures.values())=={4} and sum(exposures.values())==1016,'complete matched exposure')
                measure('final');item['receiver_after']=receiver_hash(b,c);item['fit_complete']=True;persist()
            report['both_fits_complete_before_development']=all(v['fit_complete'] for v in report['arms'].values());persist()
            check(report['both_fits_complete_before_development'],'no development until both fits fixed')
            # No optimizer steps or checkpoint selection occur below this point.
            b.checkpointing=False
            for arm in ARMS:
                item=report['arms'][arm];armout=out/arm
                restore_model(armout/'checkpoint-0315',b,c,ref,identities,window=w)
                before=(receiver_hash(b,c),moments())
                encoder=CleanEncoder(b,ref,checkpoint_hash=before[0],profile_hash=digest(PROFILE.encode()),cache_references=True)
                encoder.reference_cache=cache
                for name,keys in (('E',eids),('X',xids)):
                    cases=[]
                    with torch.no_grad(),(armout/f'development-{name}.jsonl').open('x') as stream:
                        for key in keys:
                            row=dev.record(key);construction=NumericConstruction(arm) if name=='X' and row['operation']=='Score' else None
                            if construction is None:case=evaluate_case(b,c,ref,dev,row,encoder)
                            else:
                                with CaptureRequest(c) as cap:case=evaluate_case(b,c,ref,dev,row,encoder,construction=construction)
                                if 'h' in cap.data:
                                    case['quantities']=quantity_receipt(c.quantity_probe(norm(cap.data['h'].float())),construction.receipt['attributes'])
                            cases.append(case);write(stream,case)
                    item['development_'+name]=summarize_learning(dev,cases,keys,kind='behavior')
                    if name=='X':
                        item['interventions']=summarize(dev,cases,unchanged=True)
                        score_cases=[c for c in cases if dev.record(c['id'])['operation']=='Score']
                        item['development_quantities']=quantity_summary(score_cases)
                        item['score_pairs']=paired_values(dev,[p for p in pairs if p['operation']=='Score'],
                                                          {c['id']:c['predicted_value'] for c in score_cases if 'predicted_value' in c})
                    check((receiver_hash(b,c),moments())==before,'evaluation modified model or optimizer');frozen();persist()
                    emit(event='development_complete',arm=arm,panel=name,operations={op:v['counts'] for op,v in item['development_'+name]['operations'].items()})
            report['final_base']=frozen_check(b,a.snapshot);frozen()
            check(report['final_base']['pass'] and report['final_base']['base_tied_storage'],'final W/tie')
            report.update(status='COMPLETED_NOT_PROMOTED',reference_unchanged=True,frozen_native_unchanged=True,gradients_none=True,
                          optimizer_updates=sum(v['updates'] for v in report['arms'].values()),seconds=time.monotonic()-started,
                          peak_cuda_bytes=torch.cuda.max_memory_allocated(),cached_frozen_references=len(cache))
        except BaseException as exc:
            report.update(status='FAILED',error=repr(exc),seconds=time.monotonic()-started);raise
        finally:persist()
    emit(status=report['status'],updates=report['optimizer_updates'])


if __name__=='__main__':main()
