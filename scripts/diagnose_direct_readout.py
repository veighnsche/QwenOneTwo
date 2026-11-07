"""GPU affine probes of saved training/X h; no backbone or production changes."""
import argparse
import json
from pathlib import Path
import shutil
import time
from qwen_one_two.g1_device import PRIMARY, require_visibility
from qwen_one_two.training.inputs import canonical, check, digest
from qwen_one_two.training.manifest import file_hash
from qwen_one_two.training.release import load_release, decode
from qwen_one_two.training.two_gpu import ROOT, CHECKPOINT, CHECKPOINT_SHA, RECEIVER, REFERENCE, source_hashes
from qwen_one_two.training.receiver_repair import training_ids
from qwen_one_two.training.receiver_comparison import load_control, CONTROL
from qwen_one_two.training.behavior_qualification import suite, expected
from qwen_one_two.training.component_producer import component_view
from qwen_one_two.training.direct_readout import PROTOCOL, LAMBDAS, fold, selected, targets, fit_operation, fit_ridge, predict, metrics, paired_values
from scripts.fit_reference import TRAIN_SHA, DEV_SHA

CAPTURES = ROOT/'data/information-loss-20261107/run-v2'


def records(path):
    return [decode(line) for line in path.read_bytes().splitlines()]


def numerical_checks():
    import torch
    x = torch.tensor([[1.,0.],[-1.,0.],[0.,1.],[0.,-1.]],device='cuda',dtype=torch.float64)
    y = .4+.2*x[:,0]+.1*x[:,1]
    model, residual = fit_ridge(x,y,.01)
    exact = torch.tensor([.2,.1],device='cuda',dtype=torch.float64)*(2/(2+4*.01))
    check(torch.allclose(model['weight'],exact,rtol=1e-10,atol=1e-12) and abs(float(model['intercept'])-.4)<1e-12, 'analytic affine solution')
    z = torch.cat(((x-model['center'])/model['scale'],torch.ones((len(x),1),device='cuda',dtype=torch.float64)),dim=1)
    penalty = torch.diag(torch.tensor([.04,.04,0.],device='cuda',dtype=torch.float64))
    primal = torch.linalg.solve(z.T@z+penalty,z.T@y)
    actual = torch.cat((model['weight'],model['intercept'][None]))
    error = float((actual-primal).abs().max())
    check(error<1e-10,'independent primal/dual agreement')
    try:
        fit_ridge(torch.ones_like(x),y,.01)
    except ValueError:
        rejected=True
    else:
        rejected=False
    check(rejected,'constant representation must fail')
    return {'analytic_affine_pass':True,'primal_dual_max_absolute_error':error,
            'relative_equation_residual':residual,'constant_features_rejected':rejected}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',required=True);ap.add_argument('--execute-gpu',action='store_true');args=ap.parse_args()
    out=Path(args.output).resolve();check(out.is_relative_to(ROOT/'data') and not out.exists(),'fresh data/ output required')
    out.mkdir(parents=True)
    train=load_release(ROOT/'data/training-collections-20261105/releases/train',expected_hash=TRAIN_SHA)
    dev=load_release(ROOT/'data/training-collections-20261105/releases/development',expected_hash=DEV_SHA)
    tids=training_ids(train);binding,control_plan,control_report,_=load_control(tids)
    evidence=decode((ROOT/'docs/evidence/information-loss-v1.json').read_bytes())
    check(evidence['status']=='COMPLETED_READ_ONLY_NOT_QUALIFIED' and evidence['receiver']==RECEIVER
          and evidence['reference']==REFERENCE and evidence['frozen_unchanged'],'pinned frozen X captures')
    for name,sha in (('report.json',evidence['report_sha256']),('plan.json',evidence['plan_sha256']),
                     ('captures.safetensors',evidence['capture_sha256'])):
        check(file_hash(CAPTURES/name)==sha,'development capture artifact changed')
    for name in ('cases.jsonl','capture-index.json'):
        path=CAPTURES/name;check(file_hash(path)==evidence['raw_hashes'][str(path.relative_to(ROOT))],'capture receipt changed')
    for name,sha in evidence['plan']['sources'].items():
        check(file_hash(ROOT/name)==sha and file_hash(CAPTURES/'source'/name)==sha,'captured representation code drift: '+name)
    check(file_hash(ROOT/CHECKPOINT/'manifest.json')==CHECKPOINT_SHA,'frozen parent manifest')
    pairs,omitted=suite(dev);dids=[k for pair in pairs for k in pair['ids']]
    tcases={r['record']:r for r in records(CONTROL/'features.jsonl')}
    dcases={r['id']:r for r in records(CAPTURES/'cases.jsonl')}
    index=decode((CAPTURES/'capture-index.json').read_bytes())
    check(set(tcases)==set(tids) and set(dcases)==set(index)==set(dids),'complete feature/receipt scope')
    panels={};views={}
    for name,release,ids,receipts in (('training',train,tids,tcases),('development_X',dev,dids,dcases)):
        views[name]={}
        for key in ids:
            row=release.record(key);view,provenance,_=component_view(release,row,intervention=name!='training')
            sha=digest(view.materialize(provenance)[1]);case=receipts[key]
            check(case['record_sha256']==digest(canonical(row)) and sha==case['input_sha256' if name=='training' else 'view_sha256'],'row/view binding drift')
            views[name][key]=sha
            if name!='training':check(index[key]['view']==sha,'capture-index view drift')
        panels[name]={op:selected(release,ids,op) for op in ('Noul','Score')}
    check(not {train.record(k)['family'] for k in tids}&{dev.record(k)['family'] for k in dids},'train/development family overlap')
    check(not set(views['training'].values())&set(views['development_X'].values()),'train/development permitted-input overlap')
    for op,count in (('Noul',256),('Score',254)):
        keys=panels['training'][op]
        check(len(keys)==count and len({views['training'][k] for k in keys})==count,'complete unique training inputs')
        if op=='Noul':check(set(targets(train,keys))=={0.,1.},'binary reviewed Noul targets')
        for held in range(3):
            ys=targets(train,[k for k in keys if fold(train.record(k)['family'])==held])
            check(len(ys)>1 and min(ys)<max(ys),'nondegenerate training fold')
    check([len(panels['development_X'][op]) for op in ('Noul','Score')]==[70,40],'saved X scalar coverage')
    # Pin value-only receipts as a comparison, never as fit targets or selection.
    vo_root=ROOT/'data/receiver-value-only-20261107/run'
    vo_evidence=decode((ROOT/'docs/evidence/receiver-value-only-comparison-v1.json').read_bytes())
    comparisons={}
    for label,path,sha in (('parent',CAPTURES/'cases.jsonl',evidence['raw_hashes'][str((CAPTURES/'cases.jsonl').relative_to(ROOT))]),
                          ('joint',CONTROL/'development-X.jsonl',decode((ROOT/'docs/evidence/receiver-residual-repair-v1.json').read_bytes())['raw_hashes'][str((CONTROL/'development-X.jsonl').relative_to(ROOT))]),
                          ('value_only',vo_root/'development-X.jsonl',vo_evidence['raw_hashes'][str((vo_root/'development-X.jsonl').relative_to(ROOT))])):
        check(file_hash(path)==sha,'comparison receipt changed')
        comparisons[label]={'path':str(path.relative_to(ROOT)),'sha256':sha}
    sources=source_hashes();sources['scripts/diagnose_direct_readout.py']=file_hash(Path(__file__))
    sources['docs/direct-latent-readout-v1.md']=file_hash(ROOT/'docs/direct-latent-readout-v1.md')
    plan={'protocol':PROTOCOL,'gpu_uuid':PRIMARY,'sources':sources,'receiver':RECEIVER,'reference':REFERENCE,
          'parent_manifest_sha256':CHECKPOINT_SHA,'dataset':TRAIN_SHA,'development':DEV_SHA,'panels':panels,
          'training_binding':binding,'training_feature_sha256':control_report['feature_sha256'],
          'development_capture_sha256':evidence['capture_sha256'],'capture_evidence_sha256':file_hash(ROOT/'docs/evidence/information-loss-v1.json'),
          'lambdas':LAMBDAS,'folds':{k:fold(train.record(k)['family']) for op in panels['training'].values() for k in op},
          'training_families':{k:train.record(k)['family'] for op in panels['training'].values() for k in op},
          'input_hashes':views,'pairs':pairs,'omitted_reader_ids':omitted,'comparisons':comparisons,
          'scope':'standalone assessed scalar probe; no installation, status or delivery qualification',
          'planned_cv_solves':24,'planned_final_solves':2,'new_sessions':0,'new_annotations':0}
    (out/'plan.json').write_bytes(canonical(plan))
    for name in sources:
        dst=out/'source'/name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,dst)
    if not args.execute_gpu:print('PREPARED_GPU_NOT_RUN');return
    require_visibility()
    import torch
    from safetensors import safe_open
    from safetensors.torch import load_file,save_file
    from qwen_one_two.g1_proof import configure
    from qwen_one_two.training.reference import state_fingerprint
    from qwen_one_two.g1_device import require_cuda
    configure();torch.set_num_threads(1);started=time.monotonic()
    report={'protocol':PROTOCOL,'status':'RUNNING','plan_sha256':file_hash(out/'plan.json'),'gpu_uuid':PRIMARY,
            'receiver':RECEIVER,'reference':REFERENCE,'selection':{},'training':{},'training_oof':{},'development_X':{},
            'backbone_loaded':False,'reference_loaded':False,'production_installed':False,'optimizer_updates':0}
    def persist():(out/'report.json').write_bytes(canonical(report))
    def emit(**row):print(json.dumps(row,allow_nan=False),flush=True)
    with torch.no_grad():
        try:
            report['numerical_checks']=numerical_checks()
            cached=load_file(str(CONTROL/'features.safetensors'),device='cuda')
            cache_hash=state_fingerprint(cached);models={}
            for op in ('Noul','Score'):
                keys=panels['training'][op];x=torch.stack([cached[k] for k in keys])
                require_cuda(torch.cuda,x);check(x.dtype==torch.float32 and x.shape==(len(keys),1024),'original FP32 h shape')
                model,selection=fit_operation(train,keys,x);models[op]=model;report['selection'][op]=selection
                raw=predict(model,x).tolist()
                report['training'][op]=metrics(train,keys,raw)
                chosen=next(c for c in selection['candidates'] if c['lambda']==selection['selected_lambda'])
                report['training_oof'][op]=metrics(train,keys,chosen['oof_raw'])
                emit(event='fit',operation=op,selected_lambda=selection['selected_lambda'],training=report['training'][op],oof=report['training_oof'][op])
            tensors={op+'.'+key:value.contiguous() for op,m in models.items() for key,value in m.items()}
            save_file(tensors,str(out/'readouts.safetensors'))
            export={'protocol':PROTOCOL,'plan_sha256':report['plan_sha256'],'file_sha256':file_hash(out/'readouts.safetensors'),
                    'feature_receiver':RECEIVER,'reference':REFERENCE,'dataset':TRAIN_SHA,
                    'selection':report['selection'],'fit_complete_before_development_tensors_loaded':True,
                    'production_compatible':False,'capability_promoted':False}
            (out/'export.json').write_bytes(canonical(export));report['export_sha256']=file_hash(out/'export.json')
            loaded=load_file(str(out/'readouts.safetensors'),device='cuda')
            reloaded={op:{k:loaded[op+'.'+k] for k in models[op]} for op in models}
            check(all(torch.equal(t,loaded[k]) for k,t in tensors.items()),'probe export weights drift')
            # Development tensors are opened only after both final probes/export.
            with safe_open(str(CAPTURES/'captures.safetensors'),framework='pt',device='cuda') as handle:
                development={k:handle.get_tensor(k+'.h') for op in panels['development_X'].values() for k in op}
            dev_hash=state_fingerprint(development);predictions=[]
            for name,release,panel,features in (('training',train,panels['training'],cached),('development_X',dev,panels['development_X'],development)):
                for op in ('Noul','Score'):
                    keys=panel[op];x=torch.stack([features[k] for k in keys])
                    raw_tensor=predict(models[op],x)
                    check(torch.equal(raw_tensor,predict(reloaded[op],x)),'exact reload prediction parity')
                    raw=raw_tensor.tolist();raw32=raw_tensor.float().tolist();report[name][op]=metrics(release,keys,raw)
                    for key,v,v32 in zip(keys,raw,raw32):
                        predictions.append({'split':name,'operation':op,'id':key,'record_sha256':digest(canonical(release.record(key))),
                            'input_sha256':views[name][key],'raw_fp64':v,'raw_fp32':v32,
                            'new_prefills':0,'new_causal_positions':0,'diagnostic_only':True})
            (out/'predictions.jsonl').write_text(''.join(json.dumps(x,allow_nan=False)+'\n' for x in predictions))
            report['comparisons']={};report['paired_values']={'direct':paired_values(dev,pairs,{r['id']:r['raw_fp64'] for r in predictions if r['split']=='development_X'})}
            for label,info in comparisons.items():
                cases={r['id']:r for r in records(ROOT/info['path'])}
                report['comparisons'][label]={op:metrics(dev,keys,[cases[k]['predicted_value'] for k in keys]) for op,keys in panels['development_X'].items()}
                report['paired_values'][label]=paired_values(dev,pairs,{k:cases[k]['predicted_value'] for keys in panels['development_X'].values() for k in keys})
            report['comparison_input']='saved clipped scalar values; baseline raw_mse refers to those values, not LM-head/logit raw values'
            check(state_fingerprint(cached)==cache_hash and state_fingerprint(development)==dev_hash,'saved features mutated')
            check(file_hash(CONTROL/'features.safetensors')==plan['training_feature_sha256'] and
                  file_hash(CAPTURES/'captures.safetensors')==plan['development_capture_sha256'] and
                  file_hash(ROOT/CHECKPOINT/'manifest.json')==CHECKPOINT_SHA,'input/parent files changed')
            check(all(t.is_cuda and not t.requires_grad and t.grad is None for t in [*cached.values(),*development.values(),*tensors.values()]),'CUDA/no-backprop ownership')
            report.update(status='COMPLETED_DIRECT_READOUT_NOT_PROMOTED',export_reload_exact=True,feature_files_unchanged=True,
                frozen_model_not_loaded=True,training_tensor_fingerprint=cache_hash,development_tensor_fingerprint=dev_hash,
                completed_cv_solves=24,completed_final_solves=2,new_sessions=0,new_prefills=0,new_causal_positions=0,
                new_offline_source_tokens=0,new_reference_encodings=0,seconds=time.monotonic()-started,peak_cuda_bytes=torch.cuda.max_memory_allocated())
        except BaseException as exc:report.update(status='FAILED_DIRECT_READOUT',error=repr(exc),seconds=time.monotonic()-started);raise
        finally:persist()
    emit(status=report['status'],seconds=report['seconds'])


if __name__=='__main__':main()
