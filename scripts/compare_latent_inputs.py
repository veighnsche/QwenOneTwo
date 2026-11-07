"""One frozen 3090 comparison: ordered full-width means versus learned pooling."""
import argparse
from pathlib import Path
import json
import shutil
import time
from qwen_one_two.g1_device import PRIMARY, diagnostic_device, require_visibility
from qwen_one_two.training.inputs import canonical, check, digest, SLOTS
from qwen_one_two.training.manifest import file_hash, BASE_SHA, DIAGNOSTIC_SHA
from qwen_one_two.training.release import load_release, decode
from qwen_one_two.training.two_gpu import ROOT, CHECKPOINT, CHECKPOINT_SHA, RECEIVER, REFERENCE, DEV_SHA, source_hashes
from qwen_one_two.training.behavior_qualification import suite, PROFILE as NUMERIC_PROFILE
from qwen_one_two.training.component_producer import component_view
from qwen_one_two.training.information_loss import CaptureRequest, validate_replay, difference
from qwen_one_two.training.simple_latent import PROFILE, bins, ordered_means, cached_encoder, assert_input_match, paired_outcomes

PROTOCOL = 'simple-latent-comparison-v1'
CONTROL = ROOT/'data/information-loss-20261107/run-v2'


def read_cases(path):
    return [decode(line) for line in path.read_bytes().splitlines()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--execute-gpu', action='store_true')
    args = parser.parse_args()
    out = Path(args.output).resolve()
    check(out.is_relative_to(ROOT/'data') and not out.exists(), 'fresh local data/ output required')
    out.mkdir(parents=True)
    evidence_path = ROOT/'docs/evidence/information-loss-v1.json'
    evidence = decode(evidence_path.read_bytes())
    check(evidence['status']=='COMPLETED_READ_ONLY_NOT_QUALIFIED' and evidence['receiver']==RECEIVER
          and evidence['reference']==REFERENCE and evidence['frozen_unchanged'], 'frozen control evidence')
    check(file_hash(CONTROL/'report.json')==evidence['report_sha256'] and
          file_hash(CONTROL/'plan.json')==evidence['plan_sha256'] and
          file_hash(CONTROL/'captures.safetensors')==evidence['capture_sha256'], 'saved capture identity')
    for name in ('cases.jsonl','capture-index.json'):
        path = CONTROL/name
        check(file_hash(path)==evidence['raw_hashes'][str(path.relative_to(ROOT))], 'saved control receipt drift')
    for name, sha in evidence['plan']['sources'].items():
        check(file_hash(ROOT/name)==sha, 'control execution source changed: '+name)
    check(file_hash(ROOT/CHECKPOINT/'manifest.json')==CHECKPOINT_SHA, 'parent manifest')
    dev = load_release(ROOT/'data/training-collections-20261105/releases/development',expected_hash=DEV_SHA)
    pairs, omitted = suite(dev); ids = [key for p in pairs for key in p['ids']]
    controls = read_cases(CONTROL/'cases.jsonl'); index = decode((CONTROL/'capture-index.json').read_bytes())
    check([c['id'] for c in controls]==ids and set(index)==set(ids), 'complete paired control')
    old = {c['id']:c for c in controls}; roles = {}; annotations = {}
    for key in ids:
        row = dev.record(key); view, provenance, _ = component_view(dev,row,intervention=True)
        refs = [{'sha256':r.sha256,'role':r.role} for r in view.references]
        check(refs==index[key]['refs'] and digest(view.materialize(provenance)[1])==old[key]['view_sha256']
              and digest(canonical(row))==old[key]['record_sha256'], 'permitted input binding drift')
        for ref in view.references:
            check(ref.sha256 not in roles or roles[ref.sha256]==ref.role, 'reference role collision')
            roles[ref.sha256] = ref.role; annotations[ref.sha256] = ref
    anchors = [next(k for k in ids if dev.record(k)['operation']==op) for op in ('Noul','Choice','Score')]
    sources = source_hashes(); sources['scripts/compare_latent_inputs.py'] = file_hash(Path(__file__))
    plan = {'protocol':PROTOCOL, 'candidate_profile':PROFILE, 'numeric_profile':NUMERIC_PROFILE,
        'sources':sources, 'control_evidence_sha256':file_hash(evidence_path),
        'capture_sha256':evidence['capture_sha256'], 'receiver':RECEIVER, 'reference':REFERENCE,
        'checkpoint':str(CHECKPOINT), 'checkpoint_manifest_sha256':CHECKPOINT_SHA, 'dataset':DEV_SHA,
        'pairs':pairs, 'omitted_reader_ids':omitted, 'anchors':anchors, 'gpu_uuid':PRIMARY,
        'new_sessions':{'control_anchors':3,'candidate':180}, 'reused_control_cases':180,
        'candidate_count':1, 'optimizer_updates':0, 'new_annotations':0,
        'snapshot':str(Path(args.snapshot).resolve()),
        'scope':'frozen receiver compatibility only; fixed slot capacity; no fitted readout, new data or promotion'}
    (out/'plan.json').write_bytes(canonical(plan))
    for name in sources:
        path=out/'source'/name;path.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,path)
    if not args.execute_gpu:
        print('PREPARED_GPU_NOT_RUN');return
    require_visibility()
    check(file_hash(Path(args.snapshot)/'model.safetensors-00001-of-00001.safetensors')==BASE_SHA,'base hash')
    diagnostic=ROOT/'data/g1-proof/gate-repair-v2-20261105/diagnostic.safetensors'
    check(file_hash(diagnostic)==DIAGNOSTIC_SHA,'diagnostic codec hash')
    import torch
    from safetensors.torch import load_file, save_file
    from torch.nn.attention import sdpa_kernel, SDPBackend
    from qwen_one_two.g1_proof import configure
    from qwen_one_two.g1_evidence import frozen_check
    from qwen_one_two.codec.qualify import load
    from qwen_one_two.codec.schema import Role
    from qwen_one_two.training.reference import Reference, state_fingerprint
    from qwen_one_two.training.receiver_learning import groups
    from qwen_one_two.training.checkpoint import restore_model
    from qwen_one_two.training.behavior_qualification import receiver_hash, evaluate_case, summarize
    from qwen_one_two.training.learning_report import summarize_learning
    configure(); started=time.monotonic()
    report={'protocol':PROTOCOL,'status':'RUNNING','candidate_profile':PROFILE,'gpu_uuid':PRIMARY,
        'plan_sha256':file_hash(out/'plan.json'),'optimizer_updates':0,'training_performed':False,
        'completed_cases':0,'receiver':RECEIVER,'reference':REFERENCE}
    def persist():(out/'report.json').write_bytes(canonical(report))
    def emit(**row):print(json.dumps(row,allow_nan=False),flush=True)
    captures={}; candidate_encoder=None
    with diagnostic_device(PRIMARY,execute_gpu=True), sdpa_kernel(SDPBackend.MATH), torch.no_grad():
        try:
            b,c,_=load(args.snapshot,diagnostic)
            ref=Reference(b,c,pooler_kind='target-v2');ref.freeze();groups(b,c,ref)
            parent=decode((ROOT/CHECKPOINT/'manifest.json').read_bytes())
            restore_model(ROOT/CHECKPOINT,b,c,ref,parent['identities'])
            b.requires_grad_(False);c.requires_grad_(False);ref.requires_grad_(False);b.checkpointing=False
            check(receiver_hash(b,c)==RECEIVER and ref.fingerprint()==REFERENCE,'restored frozen identities')
            report['initial_base']=frozen_check(b,args.snapshot)
            check(report['initial_base']['pass'] and report['initial_base']['base_tied_storage'],'initial W/tie')
            cache=load_file(str(CONTROL/'captures.safetensors'),device='cuda');cache_hash=state_fingerprint(cache)
            options=dict(receiver=RECEIVER,profile_hash=digest(NUMERIC_PROFILE.encode()),cache_sha=evidence['capture_sha256'])
            control_encoder=cached_encoder(b,ref,cache,roles,candidate=False,**options)
            candidate_encoder=cached_encoder(b,ref,cache,roles,candidate=True,**options)
            errors=[];norm_errors=[]
            for sha,role in roles.items():
                states=cache['reference.'+sha+'.pre'];count=SLOTS[role][1]
                value=candidate_encoder.reference_payload(annotations[sha])
                # Independent high-precision arithmetic, on the same 3090 only.
                exact=torch.stack([states[a:z].double().sum(0)/(z-a) for a,z in bins(len(states),count)])
                magnitude=exact.square().mean(-1,keepdim=True)
                exact=exact/torch.sqrt(magnitude.clamp_min(1e-12))
                error=(value.double()-exact).square().mean().sqrt()
                rms=exact.square().mean().sqrt();relative=error/rms.clamp_min(1e-6)
                check(float(relative)<=1e-5 and (float(rms)>1e-6 or float(error)<=1e-11),'ordered pooling FP32/FP64 numeric mismatch')
                errors.append(float(relative));norm_errors.append(float((value.square().mean(-1)-1).abs().max()))
                check(value.shape==cache['reference.'+sha+'.pool'].shape and torch.isfinite(value).all(),'payload shape/finite')
            report['pooling_checks']={'real_roles_checked':len(errors),'max_fp32_reference_relative_rms':max(errors),
                'max_slot_squared_rms_error_from_one':max(norm_errors),'all_tokens_covered_once':True}
            pooler_calls=[]
            hook=ref.pooler.register_forward_hook(lambda *args:pooler_calls.append(1))
            anchor_cases=[]
            try:
                with (out/'anchors.jsonl').open('x') as stream:
                    for key in anchors:
                        with CaptureRequest(c) as observation:
                            case=evaluate_case(b,c,ref,dev,dev.record(key),control_encoder)
                        stream.write(json.dumps(case,allow_nan=False)+'\n');stream.flush()
                        validate_replay(case,old[key])
                        check(observation.calls==1 and torch.equal(observation.data['wire'],cache[key+'.wire'])
                              and torch.equal(observation.data['h'],cache[key+'.h']),'cached control feature parity')
                        anchor_cases.append(case);emit(event='control_exact',id=key)
                report['control_anchors_exact']=len(anchor_cases);persist()
                cases=[];comparisons=[]
                semantic={int(x) for x in (Role.QUESTION,Role.TARGET,Role.CANDIDATE,Role.DIMENSION,
                    Role.CONDITIONS,Role.ZERO,Role.ONE,Role.MIDPOINT,Role.KEY)}
                with (out/'candidate.jsonl').open('x') as stream:
                    for key in ids:
                        with CaptureRequest(c) as observation:
                            case=evaluate_case(b,c,ref,dev,dev.record(key),candidate_encoder)
                        stream.write(json.dumps(case,allow_nan=False)+'\n');stream.flush()
                        check(observation.calls==1 and 'wire' in observation.data,'complete candidate request forward')
                        data=observation.data;captures[key]=data
                        assert_input_match(case,old[key])
                        check(data['roles']==index[key]['roles'],'wire roles changed')
                        mask=torch.tensor([r in semantic for r in data['roles']],device='cuda')
                        check(torch.equal(data['wire'][~mask],cache[key+'.wire'][~mask]),'nonsemantic wire changed')
                        comparisons.append({'id':key,'wire':difference(data['wire'],cache[key+'.wire']),
                            'h':difference(data['h'],cache[key+'.h']), 'nonsemantic_wire_exact':True})
                        cases.append(case);report['completed_cases']+=1;persist()
                        emit(event='candidate',completed=len(cases),id=key,passed=case['score']['passed'],failure=case['failure'])
            finally:hook.remove()
            check(not pooler_calls,'unexpected reference pooler execution')
            invariance=[]
            for pair in pairs:
                a,z=pair['ids']
                if old[a]['view_sha256']==old[z]['view_sha256']:
                    check(torch.equal(captures[a]['wire'],captures[z]['wire']) and torch.equal(captures[a]['h'],captures[z]['h']),
                          'identical permitted inputs changed candidate features')
                    invariance.append(pair['id'])
            (out/'differences.json').write_bytes(canonical(comparisons))
            report['control_behavior']=summarize(dev,controls,unchanged=True)
            report['candidate_behavior']=summarize(dev,cases,unchanged=True)
            report['control_learning']=summarize_learning(dev,controls,ids,kind='behavior')
            report['candidate_learning']=summarize_learning(dev,cases,ids,kind='behavior')
            report['paired_outcomes']=paired_outcomes(dev,controls,cases)
            report['identical_input_invariance']=invariance
            report['accounting']={'candidate_sessions':len(cases),'candidate_positions':sum(x['runtime']['total'] for x in cases),
                'control_anchor_sessions':len(anchor_cases),'control_anchor_positions':sum(x['runtime']['total'] for x in anchor_cases),
                'reused_control_positions_executed':0,'new_reference_encodings':0,'reference_pooler_calls':0,
                'unique_reused_reference_states':len(roles),'candidate_offline_source_tokens':candidate_encoder.offline_source_tokens,
                'control_offline_source_tokens':control_encoder.offline_source_tokens,'text_invocation_tokens':0}
            report['final_base']=frozen_check(b,args.snapshot)
            check(report['final_base']['pass'] and report['final_base']['base_tied_storage'] and
                  receiver_hash(b,c)==RECEIVER and ref.fingerprint()==REFERENCE and state_fingerprint(cache)==cache_hash and
                  all(not p.requires_grad and p.grad is None for m in (b,c,ref) for p in m.parameters()),'frozen/cache/gradient boundary')
            report.update(status='COMPLETED_COMPARISON_NOT_PROMOTED',frozen_unchanged=True,cache_unchanged=True,
                gradients_none=True,seconds=time.monotonic()-started,peak_cuda_bytes=torch.cuda.max_memory_allocated())
        except BaseException as exc:
            report.update(status='FAILED_COMPARISON',error=repr(exc),seconds=time.monotonic()-started);raise
        finally:
            tensors={key+'.'+name:value.contiguous() for key,data in captures.items()
                     for name,value in data.items() if isinstance(value,torch.Tensor)}
            if candidate_encoder is not None:
                tensors.update({'candidate.'+sha:value.contiguous() for sha,value in candidate_encoder.payloads.items()})
            if tensors:
                save_file(tensors,str(out/'candidate.safetensors'));report['candidate_capture_sha256']=file_hash(out/'candidate.safetensors')
            persist()
    emit(status=report['status'],cases=report['completed_cases'])


if __name__=='__main__':main()
