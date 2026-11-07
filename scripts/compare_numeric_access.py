"""One frozen 3090 comparison of literal versus INT64 source-count attributes."""
import argparse
from pathlib import Path
import json
import shutil
import time
from qwen_one_two.g1_device import PRIMARY, diagnostic_device, require_visibility
from qwen_one_two.training.inputs import canonical, check, digest
from qwen_one_two.training.manifest import file_hash, BASE_SHA, DIAGNOSTIC_SHA
from qwen_one_two.training.release import load_release, decode
from qwen_one_two.training.two_gpu import ROOT, CHECKPOINT, CHECKPOINT_SHA, RECEIVER, REFERENCE, DEV_SHA, source_hashes
from qwen_one_two.training.behavior_qualification import suite, PROFILE, grade, expected
from qwen_one_two.training.component_producer import component_view
from qwen_one_two.training.numeric_access import PROTOCOL, ARMS, NumericConstruction, legacy_hash, frozen_encoder
from qwen_one_two.training.information_loss import CaptureRequest, validate_replay, difference
from qwen_one_two.training.learning_report import summarize_learning
from qwen_one_two.training.direct_readout import paired_values

CONTROL=ROOT/'data/information-loss-20261107/run-v2'
MIGRATED={'qwen_one_two/training/clean.py','qwen_one_two/training/component_producer.py',
          'qwen_one_two/training/clean_packets.py','qwen_one_two/training/behavior_qualification.py'}


def read_cases(path):
    return [decode(line) for line in path.read_bytes().splitlines()]


def request_prefix(summary):
    events=summary['events'];last=next(i for i,e in enumerate(events) if e['label']=='request')
    return events[:last+1]


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--snapshot',required=True);ap.add_argument('--output',required=True)
    ap.add_argument('--execute-gpu',action='store_true');args=ap.parse_args()
    out=Path(args.output).resolve();check(out.is_relative_to(ROOT/'data') and not out.exists(),'fresh data output required')
    out.mkdir(parents=True)
    evidence=decode((ROOT/'docs/evidence/information-loss-v1.json').read_bytes())
    check(evidence['receiver']==RECEIVER and evidence['reference']==REFERENCE and evidence['frozen_unchanged'],'frozen capture identity')
    artifact_hashes={}
    def pin(path,sha):
        check(file_hash(path)==sha,'saved artifact drift: '+str(path));artifact_hashes[str(path.relative_to(ROOT))]=sha
    pin(CONTROL/'captures.safetensors',evidence['capture_sha256'])
    for name in ('cases.jsonl','capture-index.json'):
        path=CONTROL/name;pin(path,evidence['raw_hashes'][str(path.relative_to(ROOT))])
    migration={}
    for name,sha in evidence['plan']['sources'].items():
        check(file_hash(CONTROL/'source'/name)==sha,'archived capture source changed')
        current=file_hash(ROOT/name)
        if current!=sha:
            check(name in MIGRATED,'unrelated representation change: '+name)
            migration[name]={'old':sha,'new':current}
    check(set(migration)==MIGRATED,'explicit hash/evaluator migration')
    pin(ROOT/CHECKPOINT/'manifest.json',CHECKPOINT_SHA)
    dev=load_release(ROOT/'data/training-collections-20261105/releases/development',expected_hash=DEV_SHA)
    all_pairs,omitted=suite(dev);pairs=[p for p in all_pairs if p['category'] in (7,8)]
    ids=[key for pair in pairs for key in pair['ids']]
    check(len(ids)==40 and len(pairs)==20,'complete Score X comparison')
    controls={x['id']:x for x in read_cases(CONTROL/'cases.jsonl')}
    index=decode((CONTROL/'capture-index.json').read_bytes())
    all_ids=[key for pair in all_pairs for key in pair['ids']]
    anchors=[next(k for k in all_ids if dev.record(k)['operation']==op) for op in ('Noul','Choice','Score')]
    anchors.append(next(k for k in ids if dev.record(k)['semantics']['rubric']['direction']=='lower'))
    roles={};keys={};mapping={};input_plans={}
    for key in dict.fromkeys(anchors+ids):
        row=dev.record(key);view,p,indices=component_view(dev,row,intervention=True)
        _,body=view.materialize(p)
        check(digest(canonical(row))==controls[key]['record_sha256'],'saved row binding')
        check(legacy_hash(body)==controls[key]['view_sha256']==index[key]['view'],'explicit old input binding')
        check([{'role':r.role,'sha256':r.sha256} for r in view.references]==index[key]['refs'],'saved role binding')
        mapping[key]={'legacy':legacy_hash(body),'v2':digest(body),'direction':view.rubric_direction}
        for ref in view.references:roles[ref.sha256]=ref.role
        if key in ids:
            input_plans[key]={}
            for arm in ARMS:
                builder=NumericConstruction(arm);augmented,_,_=builder.prepare(view,p,indices)
                input_plans[key][arm]=builder.receipt
                for ref in augmented.references[len(view.references):]:keys[ref.sha256]=ref.content.decode()
    check(len(keys)==2,'two verbatim existing property/unit key inputs')
    sources=source_hashes()
    for name in ('scripts/compare_numeric_access.py','docs/numeric-access-comparison-v1.md','tests/test_numeric_access.py'):
        sources[name]=file_hash(ROOT/name)
    plan={'protocol':PROTOCOL,'sources':sources,'artifact_hashes':artifact_hashes,'source_migration':migration,
        'receiver':RECEIVER,'reference':REFERENCE,'dataset':DEV_SHA,'checkpoint_manifest_sha256':CHECKPOINT_SHA,
        'gpu_uuid':PRIMARY,'snapshot':str(Path(args.snapshot).resolve()),'pairs':pairs,'ids':ids,'anchors':anchors,
        'view_hash_migration':mapping,'inputs':input_plans,'derived_verbatim_keys':keys,
        'new_sessions':84,'new_data':False,'training':False,'optimizer_updates':0,'candidate_count':1,
        'scope':'fixed receiver compatibility; original source evidence plus matched literal/INT64 attributes; no promotion'}
    (out/'plan.json').write_bytes(canonical(plan))
    for name in sources:
        dst=out/'source'/name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,dst)
    if not args.execute_gpu:print('PREPARED_GPU_NOT_RUN');return
    require_visibility()
    check(file_hash(Path(args.snapshot)/'model.safetensors-00001-of-00001.safetensors')==BASE_SHA,'pinned base file')
    diagnostic=ROOT/'data/g1-proof/gate-repair-v2-20261105/diagnostic.safetensors'
    check(file_hash(diagnostic)==DIAGNOSTIC_SHA,'pinned diagnostic file')
    import torch
    from safetensors.torch import load_file,save_file
    from torch.nn.attention import sdpa_kernel,SDPBackend
    from qwen_one_two.g1_proof import configure
    from qwen_one_two.g1_evidence import frozen_check
    from qwen_one_two.codec.qualify import load
    from qwen_one_two.codec.schema import Role
    from qwen_one_two.training.reference import Reference,state_fingerprint
    from qwen_one_two.training.receiver_learning import groups
    from qwen_one_two.training.checkpoint import restore_model
    from qwen_one_two.training.behavior_qualification import receiver_hash,evaluate_case
    configure();torch.set_num_threads(1);started=time.monotonic()
    report={'protocol':PROTOCOL,'status':'RUNNING','plan_sha256':file_hash(out/'plan.json'),
        'gpu_uuid':PRIMARY,'receiver':RECEIVER,'reference':REFERENCE,'optimizer_updates':0,'training':False,
        'completed':{arm:0 for arm in ARMS},'control_anchors_exact':0}
    captures={};cases={arm:[] for arm in ARMS};anchor_cases=[];encoder=None;comparisons=[]
    def persist():(out/'report.json').write_bytes(canonical(report))
    def emit(**item):print(json.dumps(item,allow_nan=False),flush=True)
    with diagnostic_device(PRIMARY,execute_gpu=True),sdpa_kernel(SDPBackend.MATH),torch.no_grad():
        try:
            b,c,_=load(args.snapshot,diagnostic)
            ref=Reference(b,c,pooler_kind='target-v2');ref.freeze();groups(b,c,ref)
            parent=decode((ROOT/CHECKPOINT/'manifest.json').read_bytes())
            restore_model(ROOT/CHECKPOINT,b,c,ref,parent['identities'])
            b.requires_grad_(False);c.requires_grad_(False);ref.requires_grad_(False);b.checkpointing=False
            check(receiver_hash(b,c)==RECEIVER and ref.fingerprint()==REFERENCE,'restored frozen weights')
            report['initial_base']=frozen_check(b,args.snapshot)
            check(report['initial_base']['pass'] and report['initial_base']['base_tied_storage'],'initial W and tying')
            cache=load_file(str(CONTROL/'captures.safetensors'),device='cuda');cache_hash=state_fingerprint(cache)
            encoder=frozen_encoder(b,ref,cache,roles,keys,receiver=RECEIVER,profile_hash=digest(PROFILE.encode()))
            with (out/'anchors.jsonl').open('x') as stream:
                for key in anchors:
                    with CaptureRequest(c) as observation:
                        case=evaluate_case(b,c,ref,dev,dev.record(key),encoder)
                    stream.write(json.dumps(case,allow_nan=False)+'\n');stream.flush();anchor_cases.append(case)
                    # Only the explicitly mapped audit identity changes.
                    check(case['view_sha256']==mapping[key]['v2'],'v2 control view')
                    saved=dict(controls[key],view_sha256=mapping[key]['v2']);validate_replay(case,saved)
                    check(observation.calls==1 and torch.equal(observation.data['wire'],cache[key+'.wire'])
                          and torch.equal(observation.data['h'],cache[key+'.h']),'control wire/h exact')
                    report['control_anchors_exact']+=1;persist();emit(event='control_exact',id=key)
            with (out/'cases.jsonl').open('x') as stream:
                for i,key in enumerate(ids):
                    data={};current={}
                    for arm in (ARMS if i%2==0 else tuple(reversed(ARMS))):
                        builder=NumericConstruction(arm)
                        with CaptureRequest(c) as observation:
                            case=evaluate_case(b,c,ref,dev,dev.record(key),encoder,construction=builder)
                        stream.write(json.dumps(dict(case,arm=arm),allow_nan=False)+'\n');stream.flush()
                        cases[arm].append(case);report['completed'][arm]+=1
                        check(observation.calls==1 and 'h' in observation.data,'complete actual numeric request')
                        data[arm]=observation.data;current[arm]=case
                        captures[arm+'.'+key]=data[arm]
                        check(bool(torch.isfinite(data[arm]['wire']).all() & torch.isfinite(data[arm]['h']).all()),'finite captured features')
                        receipt=case['input_construction'];planned=input_plans[key][arm]
                        check(all(receipt[k]==v for k,v in planned.items()),'input construction plan binding')
                        check(case['view_sha256']==planned['augmented_view_sha256'],'actual augmented view')
                        old_sources=[x for x in controls[key]['construction_audit'] if 'local_source' in x]
                        new_sources=[x for x in case['construction_audit'] if 'local_source' in x]
                        check(canonical(new_sources)==canonical(old_sources),'exact source tensor/byte lineage')
                        old_refs=[x for x in controls[key]['construction_audit'] if 'role' in x]
                        new_refs=[x for x in case['construction_audit'] if 'role' in x]
                        check(canonical(old_refs)==canonical(new_refs[:-2]),'original reference tensors unchanged')
                        runtime=case['runtime'];old=controls[key]['runtime']
                        check(runtime['prefills']==runtime['lineage']==1 and runtime['held_positions']==0
                              and runtime['total']==sum(runtime[k] for k in ('P','B','A','H')),'single lineage/accounting')
                        for field in ('P','predicted_budget','granted_budget','override'):
                            check(runtime[field]==old[field],'startup/budget preserved')
                        check(len(data[arm]['roles'])==len(index[key]['roles'])+8,'eight added request positions')
                        persist();emit(event='case',arm=arm,id=key,completed=report['completed'][arm],
                                       value=case.get('predicted_value'),passed=case['score']['passed'],failure=case['failure'])
                    x,y=data['literal'],data['int64']
                    normalize=lambda roles:[-1 if role in (int(Role.INT64),int(Role.UTF8)) else role for role in roles]
                    check(normalize(x['roles'])==normalize(y['roles']),'matched arm role layout')
                    mask=torch.tensor([r not in (int(Role.ATTRIBUTE_DESC),int(Role.UTF8)) for r in x['roles']],device='cuda')
                    check(torch.equal(x['wire'][mask],y['wire'][mask]),'all non-attribute wire rows exact')
                    check(canonical(request_prefix(current['literal']['runtime']))==canonical(request_prefix(current['int64']['runtime'])),
                          'matched prefill/request event lengths')
                    check(current['literal']['view_sha256']==current['int64']['view_sha256'],'same source/key view')
                    comparisons.append({'id':key,'non_attribute_wire_exact':True,'request_events_exact':True,
                        'literal_to_int64_h':difference(x['h'],y['h']),
                        'baseline_to_literal_h':difference(cache[key+'.h'],x['h']),
                        'baseline_to_int64_h':difference(cache[key+'.h'],y['h'])})
            panels={'baseline':[controls[k] for k in ids],**cases};report['metrics']={};report['subsets']={}
            for arm,rows in panels.items():
                report['metrics'][arm]=summarize_learning(dev,rows,ids,kind='behavior')['operations']['Score']
                predicted={row['id']:row['predicted_value'] for row in rows if 'predicted_value' in row}
                check(len(predicted)==40,'complete scalar head denominator')
                report.setdefault('paired_values',{})[arm]=paired_values(dev,pairs,predicted)
                report['subsets'][arm]={}
                for name,subset in (('completion',[k for k in ids if k.endswith('-0')]),
                                    ('remaining_fraction',[k for k in ids if k.startswith('development-X-07') and k.endswith('-1')]),
                                    ('half_completion',[k for k in ids if k.startswith('development-X-08') and k.endswith('-1')])):
                    selected=[row for row in rows if row['id'] in subset]
                    report['subsets'][arm][name]=summarize_learning(dev,selected,subset,kind='behavior')['operations']['Score']
            report['differences']=comparisons
            report['accounting']={'new_sessions':sum(len(rows) for rows in cases.values())+len(anchor_cases),
                'anchor_positions':sum(x['runtime']['total'] for x in anchor_cases),
                'arm_positions':{arm:sum(x['runtime']['total'] for x in rows) for arm,rows in cases.items()},
                'reused_baseline_sessions_executed':0,'offline_preparation':encoder.stats,'invocation_text_tokens':0}
            report['final_base']=frozen_check(b,args.snapshot)
            check(report['final_base']['pass'] and report['final_base']['base_tied_storage'] and
                  receiver_hash(b,c)==RECEIVER and ref.fingerprint()==REFERENCE and state_fingerprint(cache)==cache_hash
                  and all(not p.requires_grad and p.grad is None for m in (b,c,ref) for p in m.parameters()),'final frozen/cache/gradient checks')
            check(encoder.stats['reference_encodings']==2,'only two derived key encodings')
            report.update(status='COMPLETED_COMPARISON_NOT_PROMOTED',frozen_unchanged=True,cache_unchanged=True,
                gradients_none=True,seconds=time.monotonic()-started,peak_cuda_bytes=torch.cuda.max_memory_allocated())
        except BaseException as exc:
            report.update(status='FAILED_COMPARISON',error=repr(exc),seconds=time.monotonic()-started);raise
        finally:
            if captures:
                tensors={key+'.'+name:value.contiguous() for key,data in captures.items() for name,value in data.items() if isinstance(value,torch.Tensor)}
                if encoder is not None:tensors.update({'derived_key.'+sha:value.contiguous() for sha,value in encoder.key_cache.items()})
                save_file(tensors,str(out/'captures.safetensors'));report['capture_sha256']=file_hash(out/'captures.safetensors')
            persist()
    emit(status=report['status'],completed=report['completed'])


if __name__=='__main__':main()
