"""One complete existing-data receiver-learning pass, with frozen W/R/reference."""
import argparse,json,time
from collections import Counter
from pathlib import Path
from qwen_one_two.training.inputs import check,digest,canonical
from qwen_one_two.training.manifest import BASE_SHA,DIAGNOSTIC_SHA,file_hash
from scripts.fit_reference import TRAIN_SHA,DEV_SHA

REFERENCE=Path('data/reference-coverage-20261106/final-pass/reference-006064')
BUNDLE_SHA='92b0b199bd98bd56684eb4f26dbc64b146d1f57266f97faac49345749446bd9d'
BASELINE=Path('data/reference-behavior-20261106/final/report.json')




def learning_loop(body):
    """Compare executable update loops while excluding the known count bug."""
    import ast
    tree=ast.parse(body)
    loop=next(n for n in ast.walk(tree) if isinstance(n,ast.For)
              and isinstance(n.target,ast.Name) and n.target.id=='window_index')
    class Bookkeeping(ast.NodeTransformer):
        def visit_AugAssign(self,node):
            if (isinstance(node.target,ast.Subscript) and isinstance(node.target.value,ast.Name)
                    and node.target.value.id=='exposures'):return None
            return self.generic_visit(node)
        def visit_Expr(self,node):
            if isinstance(node.value,ast.Call):
                f=node.value.func
                if isinstance(f,ast.Name) and f.id=='count_report':return None
                if (isinstance(f,ast.Attribute) and isinstance(f.value,ast.Name)
                        and f.value.id in {'eligible','failures'} and f.attr=='update'):return None
            return self.generic_visit(node)
    return ast.dump(Bookkeeping().visit(loop),include_attributes=False)


def count_report(exposures,eligible,failures,report):
    """Count eligible examples, never sum loss values or retain loss graphs."""
    exposures[report['record']]+=1
    if 'failure' in report:failures[report['failure']]+=1
    else:eligible.update(report.get('losses',{}).keys())


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--snapshot',required=True)
    p.add_argument('--output',required=True);p.add_argument('--resume');a=p.parse_args()
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    from qwen_one_two.training.release import load_release
    train=load_release('data/training-collections-20261105/releases/train',expected_hash=TRAIN_SHA)
    dev=load_release('data/training-collections-20261105/releases/development',expected_hash=DEV_SHA)
    from qwen_one_two.training.receiver_learning import training_order,groups,ReceiverProducer,PROTOCOL
    order=training_order(train);check(len(order)==1504,'188 complete eight-example windows')
    eids=[e['id'] for e in dev.manifest['records'] if e['selected'] and dev.record(e['id']).get('evaluation_slice')=='E']
    check(len(eids)==130,'all existing E receiver records')
    from qwen_one_two.training.behavior_qualification import suite,source_hashes,PROFILE
    xgroups,_=suite(dev);xids=[k for g in xgroups for k in g['ids']]
    sources=source_hashes();sources['scripts/train_receiver.py']=file_hash(__file__)
    for name,sha in sources.items():
        target=out/'source-snapshot'/name;target.parent.mkdir(parents=True,exist_ok=True)
        body=Path(name).read_bytes();check(digest(body)==sha,'source snapshot drift');target.write_bytes(body)
    (out/'source-snapshot/manifest.json').write_bytes(canonical(sources))
    prior=json.loads(BASELINE.read_bytes());receipt=json.loads(Path('docs/evidence/reference-behavior-v1.json').read_bytes())
    check(file_hash(BASELINE)==receipt['audit']['report_sha256'],'prior behavioral evidence drift')
    # Only frozen-reference cache factoring and E-view admission changed in the
    # prior execution path. Check every original permitted view below and verify
    # cached/uncached reference tensors on CUDA before using any update.
    changed={'qwen_one_two/training/clean.py','qwen_one_two/training/component_producer.py'}
    check(all(sources[k]==v for k,v in prior['source_hashes'].items() if k not in changed),'baseline model source drift')
    from qwen_one_two.training.component_producer import component_view
    for case in prior['cases']:
        view,prov,_=component_view(dev,dev.record(case['id']),intervention=True)
        check(digest(view.materialize(prov)[1])==case['view_sha256'],'baseline permitted view drift')
    check(file_hash(Path(a.snapshot)/'model.safetensors-00001-of-00001.safetensors')==BASE_SHA,'base hash')
    check(file_hash('data/g1-proof/gate-repair-v2-20261105/diagnostic.safetensors')==DIAGNOSTIC_SHA,'initial live hash')
    check(file_hash(REFERENCE/'manifest.json')==BUNDLE_SHA,'reference bundle hash')
    manifest=json.loads((REFERENCE/'manifest.json').read_bytes())
    check(file_hash(REFERENCE/'weights.safetensors')==manifest['files']['weights.safetensors'],'reference weights hash')
    import torch
    from safetensors import safe_open
    from torch.nn.attention import sdpa_kernel,SDPBackend
    from qwen_one_two.g1_proof import configure
    from qwen_one_two.g1_evidence import frozen_check
    from qwen_one_two.codec.qualify import load
    from qwen_one_two.training.reference import Reference,state_fingerprint
    from qwen_one_two.training.engine import Window
    from qwen_one_two.training.episodes import collect_session
    from qwen_one_two.training.checkpoint import save,bundle_tensors,restore_model
    from qwen_one_two.training.behavior_qualification import receiver_hash,evaluate_case,grade,summarize
    from qwen_one_two.training.clean import CleanEncoder
    from qwen_one_two.training.binding import encode_annotation
    configure()
    def emit(**event):print(json.dumps(event),flush=True)
    with sdpa_kernel(SDPBackend.MATH):
        b,c,_=load(a.snapshot,'data/g1-proof/gate-repair-v2-20261105/diagnostic.safetensors')
        ref=Reference(b,c,pooler_kind=manifest['pooler_kind'])
        with safe_open(REFERENCE/'weights.safetensors',framework='pt',device='cuda') as f:
            ref.load_state_dict({n:f.get_tensor('reference.'+n) for n in ref.state_dict()},strict=True)
        ref.freeze(manifest['reference_id']);before=receiver_hash(b,c)
        check(before==prior['receiver_sha256'] and ref.artifact_id==prior['reference_sha256'],'baseline weight drift')
        owned=groups(b,c,ref);w=Window(owned,'components');b.checkpointing=True
        producer=ReceiverProducer(b,c,ref,train)
        native=lambda:state_fingerprint({**{n:p for n,p in b.named_parameters() if n.endswith(('.R_A','.R_B'))},
            **{'codec.'+n:p for n,p in c.named_parameters() if not p.requires_grad}})
        frozen_native=native();base_initial=frozen_check(b,a.snapshot);check(base_initial['pass'],'initial W mismatch')
        ids={'base':BASE_SHA,'qev':file_hash('data/qev/release_manifest.json'),'protocol':digest(PROTOCOL.encode()),
            'precision':digest(PROFILE.encode()),'reference':ref.artifact_id,'dataset':train.identity,
            'sources':digest(canonical(sources))}
        cache_checks=[]
        first_view,_,_=component_view(train,train.record(order[0]))
        for item in first_view.references:
            cached=producer.encoder.reference_payload(item);again=producer.encoder.reference_payload(item)
            exact=encode_annotation(ref,item)
            check(torch.equal(cached,exact) and torch.equal(cached,again),'reference cache changed payload')
            cache_checks.append(item.sha256)
        def checkpoint():
            check(ref.fingerprint()==ref.artifact_id and native()==frozen_native,'frozen reference/R/protocol changed')
            path=out/f'checkpoint-{w.scheduled_windows:04d}'
            save(path,w,bundle_tensors(b,c,ref),ids,{'protocol':PROTOCOL,'order':order,'cursor':w.scheduled_windows*8},
                reference_frozen=True,reference_id=ref.artifact_id)
            return path
        def evaluated(name,keys):
            encoder=CleanEncoder(b,ref,checkpoint_hash=receiver_hash(b,c),profile_hash=digest(PROFILE.encode()),cache_references=True)
            encoder.reference_cache=producer.encoder.reference_cache
            cases=[]
            with torch.no_grad(),(out/(name+'.jsonl')).open('x') as stream:
                for key in keys:
                    case=evaluate_case(b,c,ref,dev,dev.record(key),encoder);cases.append(case)
                    stream.write(json.dumps(case)+'\n');stream.flush()
                    emit(event='evaluation',panel=name,id=key,passed=case['score']['passed'],failure=case['failure'])
            result={}
            for op in ('Noul','Choice','Score'):
                selected=[v for v in cases if dev.record(v['id'])['operation']==op]
                result[op]={'scheduled':len(selected),'passed':sum(v['score']['passed'] for v in selected),
                    'status_correct':sum(v['score']['status_correct'] for v in selected),
                    'transport_valid':sum(v['score']['transport_valid'] for v in selected)}
            diagnostics={}
            for op,key in (('Score','score_interval'),('Noul','p_true')):
                errors=[]
                for case in cases:
                    row=dev.record(case['id']);gold=row['supervision'].get('result',row['labels'])
                    if row['operation']!=op or gold['status']!='Assessed' or 'predicted_value' not in case:continue
                    value=case['predicted_value'];target=gold['payload'][key]
                    errors.append(max(target[0]-value,0,value-target[1]) if op=='Score' else (value-target)**2)
                diagnostics[op]={'measured_heads':len(errors),'mean_interval_absolute_error' if op=='Score' else 'brier':
                    sum(errors)/len(errors) if errors else None,'scope':'head readout regardless of returned status; not complete success'}
            return {'operations':result,'cases':cases,'head_diagnostics':diagnostics}
        baseline_e=None;preflight=[];first_window=None;start_window=0;prefix=[];resume_evidence=None
        if a.resume:
            previous=Path(a.resume).parent
            saved=json.loads((Path(a.resume)/'manifest.json').read_text())
            old_sources=json.loads((previous/'source-snapshot/manifest.json').read_text())
            old_ids=saved['identities']
            check(digest(canonical(old_sources))==old_ids['sources'],'resume source manifest drift')
            check({k:v for k,v in old_ids.items() if k!='sources'}=={k:v for k,v in ids.items() if k!='sources'},
                  'resume data/model/protocol drift')
            check(set(old_sources)==set(sources) and all(old_sources[k]==v for k,v in sources.items()
                  if k!='scripts/train_receiver.py'),'resume learning implementation changed')
            old_script=(previous/'source-snapshot/scripts/train_receiver.py').read_bytes()
            check(digest(old_script)==old_sources['scripts/train_receiver.py'],'old launcher source drift')
            check(learning_loop(old_script)==learning_loop(Path(__file__).read_bytes()),
                  'resume update loop changed beyond bookkeeping')
            # Only launcher/accounting code differs. All optimizer/loss/model,
            # producer, sampling and numerical sources are exactly unchanged.
            restored=restore_model(a.resume,b,c,ref,old_ids,window=w)
            check(tuple(restored['sampler']['order'])==order,'resume order drift');start_window=w.scheduled_windows
            check(restored['sampler']['cursor']==start_window*8,'resume cursor drift')
            baseline_e=json.loads((previous/'baseline-E-summary.json').read_text())
            import shutil
            for name in ('baseline-E-summary.json','preflight.json','first-window.json'):
                shutil.copyfile(previous/name,out/name)
            preflight=json.loads((out/'preflight.json').read_text())['cases']
            first_window=json.loads((out/'first-window.json').read_text())
            prefix=[json.loads(line) for line in (previous/'training.jsonl').read_text().splitlines()
                    if json.loads(line)['window']['window']<=start_window]
            check([v['window']['window'] for v in prefix]==list(range(1,start_window+1)),
                  'resume prefix incomplete/reordered')
            resume_evidence={'checkpoint':str(a.resume),'checkpoint_manifest_sha256':file_hash(Path(a.resume)/'manifest.json'),
                'prefix_training_log':str(previous/'training.jsonl'),'prefix_log_sha256':file_hash(previous/'training.jsonl'),
                'prefix_windows':start_window,'old_sources':old_ids['sources'],'new_sources':ids['sources'],
                'unchanged_learning_sources':True,'update_loop_AST_unchanged_except_counts':True,'migration':'eligibility bookkeeping counts keys; no model/optimizer reset'}
            (out/'resume.json').write_text(json.dumps(resume_evidence,indent=2)+'\n')
        else:
            baseline_e=evaluated('baseline-E',eids)
            (out/'baseline-E-summary.json').write_text(json.dumps(baseline_e,indent=2)+'\n')
            # Probe actual reviewed operation/status strata without stepping.
            chosen={}
            for key in order[:1500]:
                row=train.record(key);chosen.setdefault((row['operation'],row['labels']['status']),key)
            for key in chosen.values():
                s,losses,exposure=producer(train.record(key))
                check(s.runtime.terminal=='Completed' and bool(losses),'receiver path not trainable: '+key)
                gradient_checks={}
                for component,loss in losses.items():
                    grads=torch.autograd.grad(loss,w.parameters,allow_unused=True,retain_graph=True)
                    check(float(loss.detach())==0 or any(g is not None and bool(g.abs().sum()>0) for g in grads),'disconnected '+component)
                    check(all(g is None or bool(g.isfinite().all()) for g in grads),'nonfinite '+component)
                    d_norm=sum(float(g.float().square().sum()) for name,g in zip(w.names,grads)
                               if g is not None and name.endswith(('.D_A','.D_B')))**.5
                    check(d_norm>0 or float(loss.detach())==0,'disconnected D adapter: '+component)
                    gradient_checks[component]={'D_gradient_norm':d_norm,'finite':True}
                check(all(p.grad is None for p in ref.parameters()) and all(p.grad is None for p in b.adapter_parameters('R')),'frozen owner gradient')
                s.episode_report['gradient_checks']=gradient_checks
                preflight.append(s.episode_report);emit(event='preflight',id=key,losses=list(losses))
                del s,losses,grads,loss
            check(ref.fingerprint()==ref.artifact_id and native()==frozen_native,'preflight changed frozen owners')
            (out/'preflight.json').write_text(json.dumps({'cases':preflight,'cache_exact':cache_checks,'optimizer_updates':0},indent=2)+'\n')
            checkpoint()
        exposures=Counter();eligible=Counter();failures=Counter();history=[]
        for entry in prefix:
            for episode in entry['episodes']:count_report(exposures,eligible,failures,episode)
            history.append(entry['window'])
        with (out/'training.jsonl').open('x') as log:
            for window_index in range(start_window,188):
                reports=[];t0=time.monotonic()
                for key in order[window_index*8:(window_index+1)*8]:
                    s,losses,exposure=producer(train.record(key));report=s.episode_report
                    collect_session(w,s,losses,exposure)
                    count_report(exposures,eligible,failures,report)
                    reports.append(report);del s,losses
                result=w.step();entry={'window':result,'episodes':reports,'elapsed_seconds':time.monotonic()-t0}
                log.write(json.dumps(entry)+'\n');log.flush();history.append(result)
                emit(event='training',window=w.scheduled_windows,updates=w.optimizer_updates,
                     eligible=result['eligible_examples'],failed=result['failed_exposures'],seconds=entry['elapsed_seconds'])
                if w.scheduled_windows==1:
                    path=checkpoint();fingerprint=receiver_hash(b,c)
                    optimizer_before=state_fingerprint({str(i)+'.'+k:t for i,state in enumerate(w.optimizer.state.values()) for k,t in state.items()})
                    restore_model(path,b,c,ref,ids,window=w)
                    check(receiver_hash(b,c)==fingerprint,'checkpoint restore changed model')
                    check(state_fingerprint({str(i)+'.'+k:t for i,state in enumerate(w.optimizer.state.values()) for k,t in state.items()})==optimizer_before,'checkpoint optimizer restore changed')
                    first_window={'checkpoint_roundtrip_exact':True,'window':result}
                    (out/'first-window.json').write_text(json.dumps(first_window,indent=2)+'\n')
                elif w.scheduled_windows%25==0 or w.scheduled_windows==188:checkpoint()
        final_e=evaluated('final-E',eids);final_x=evaluated('final-X',xids)
        final_check=frozen_check(b,a.snapshot)
        check(final_check['pass'] and final_check['base_tied_storage'] and ref.fingerprint()==ref.artifact_id
              and native()==frozen_native,'final frozen owner check')
        check(all(p.grad is None for p in b.parameters()) and all(p.grad is None for p in c.parameters())
              and all(p.grad is None for p in ref.parameters()),'outstanding gradients')
        report={'protocol':PROTOCOL,'status':'COMPLETED','gpu_uuid':'GPU-225509c3-d702-6c27-de03-76b5e99b33b1',
            'profile':PROFILE,'identities':ids,'reference':ref.artifact_id,'receiver_before':before,'receiver_after':receiver_hash(b,c),
            'scheduled_windows':w.scheduled_windows,'optimizer_updates':w.optimizer_updates,'unique_training_records':len(exposures),
            'presentations':sum(exposures.values()),'exposures':dict(exposures),'eligible_examples':dict(eligible),
            'failed_exposures':dict(failures),'first_window':first_window,'baseline_E':baseline_e['operations'],
            'baseline_X':prior['operations'],'baseline_X_report_sha256':file_hash(BASELINE),
            'baseline_X_permitted_views_unchanged':True,'reference_cache_exact_checks':cache_checks,
            'final_E':final_e['operations'],'final_X':final_x['operations'],
            'baseline_E_head_diagnostics':baseline_e['head_diagnostics'],'final_E_head_diagnostics':final_e['head_diagnostics'],
            'final_X_head_diagnostics':final_x['head_diagnostics'],
            'final_interventions':summarize(dev,final_x['cases'],unchanged=True),
            'base_initial':base_initial,'base_final':final_check,'reference_unchanged':ref.fingerprint()==ref.artifact_id,
            'R_and_frozen_codec_unchanged':native()==frozen_native,'all_gradients_none':True,
            'source_hashes':sources,'resume':resume_evidence,'seconds':time.monotonic()-start,'peak_cuda_bytes':torch.cuda.max_memory_allocated(),
            'scope':'one receiver-only learning pass; forced reference packets; no writer/gate/reader training or full-flow promotion'}
        (out/'report.json').write_text(json.dumps(report,indent=2)+'\n');emit(event='complete',results={k:report[k] for k in ('baseline_E','final_E','baseline_X','final_X','failed_exposures')})


if __name__=='__main__':main()
