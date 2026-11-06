"""One fixed, gradient-free reference-weight repair and paired existing-data test."""
import argparse
import json
import time
from pathlib import Path
from qwen_one_two.training.inputs import canonical,digest,check,SLOTS
from qwen_one_two.training.release import load_release
from qwen_one_two.training.reference_data import catalogue
from qwen_one_two.training.weight_repair import calibration_views,deflate_affine,PROTOCOL
from scripts.fit_reference import TRAIN_SHA,DEV_SHA,source_snapshot,export_reference

PARENTS=Path('data/reference-encoder-20261106/run')
RECEIPT=Path('docs/evidence/reference-encoder-v2.json')
CONTRACT=Path('docs/reference-weight-repair-v1.md')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--snapshot',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();output=Path(args.output);output.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();receipt=json.loads(RECEIPT.read_bytes());parents={};baselines={}
    releases=Path('data/training-collections-20261105/releases')
    release_hashes={split:digest((releases/split/'manifest.json').read_bytes())
                    for split in ('train','development','locked')}
    check(release_hashes==dict(train=TRAIN_SHA,development=DEV_SHA,
        locked='d13594f18738056ee179b3498d54d8261f7f7654aad633fabb776810681aa1cf'),'release manifest drift')
    for n in (900,1000):
        path=PARENTS/f'reference-{n:04d}';raw=(path/'manifest.json').read_bytes()
        check(digest(raw)==receipt['assets'][str(path/'manifest.json')],'parent manifest drift')
        m=json.loads(raw)
        check(m['format']=='qot-frozen-reference-v1' and m['pooler_kind']=='target-v2' and m['reference_frozen'],'parent structure')
        check(digest((path/'weights.safetensors').read_bytes())==m['files']['weights.safetensors'],'parent weights drift')
        qpath=PARENTS/f'qualification-{n:04d}.json';qraw=qpath.read_bytes()
        check(digest(qraw)==receipt['assets'][str(qpath)],'baseline report drift')
        q=json.loads(qraw)
        check(q['reference_sha256']==q['reference_after_sha256']==m['reference_id'],'baseline identity')
        check(q['reference_bundle_sha256']==digest(raw),'baseline/export link')
        parents[n]=(path,m);baselines[n]=q
    train=load_release('data/training-collections-20261105/releases/train',expected_hash=TRAIN_SHA)
    views=calibration_views(catalogue(train));check(len(views)==71,'pinned calibration size')
    calibration=[dict(id=v.identity,role=v.role,annotation_sha256=v.sha256,sources=v.sources) for v in views]
    (output/'calibration.json').write_bytes(canonical(calibration))
    sources=source_snapshot(output)
    for name in ('scripts/repair_reference.py','docs/reference-weight-repair-v1.md','tests/test_weight_repair.py','tests/test_weight_repair_gpu.py'):
        body=Path(name).read_bytes();sources[name]=digest(body)
        target=output/'source-snapshot'/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(body)
    (output/'source-snapshot/manifest.json').write_bytes(canonical(sources))
    import torch
    from safetensors.torch import load_file,save_file
    from torch.nn.attention import sdpa_kernel,SDPBackend
    from qwen_one_two.g1_proof import configure
    from qwen_one_two.codec.qualify import load
    from qwen_one_two.g1_evidence import frozen_check
    from qwen_one_two.training.reference import Reference,state_fingerprint
    from qwen_one_two.training.reference_qualification import evaluate,load_reference_evidence,source_hashes
    from qwen_one_two.training.manifest import BASE_SHA,file_hash
    check(file_hash(Path(args.snapshot)/'model.safetensors-00001-of-00001.safetensors')==BASE_SHA,'base hash')
    configure();torch.cuda.reset_peak_memory_stats()
    with sdpa_kernel(SDPBackend.MATH),torch.no_grad():
        b,c,_=load(args.snapshot,'data/g1-proof/gate-repair-v2-20261105/diagnostic.safetensors')
        live_before=state_fingerprint({**dict(b.named_parameters()),**{'codec.'+n:p for n,p in c.state_dict().items()}})
        refs={};originals={};sums={};counts={role:0 for role in SLOTS};examples={};payloads={}
        for n,(path,m) in parents.items():
            ref=Reference(b,c,pooler_kind='target-v2')
            state={k.removeprefix('reference.'):v for k,v in load_file(str(path/'weights.safetensors'),device='cuda').items()}
            ref.load_state_dict(state);ref.freeze(m['reference_id']);refs[n]=ref;originals[n]=state
            check(baselines[n]['source_hashes']==source_hashes(),'baseline model-source drift')
            sums[n]={role:torch.zeros(1024,device='cuda',dtype=torch.float64) for role in SLOTS}
            payloads[n]={'shared':[],'target':[]}
        for i,view in enumerate(views,1):
            text=view.content.decode()
            rendered=b.tokenizer.apply_chat_template([{'role':'user','content':text}],tokenize=False,add_generation_prompt=True,enable_thinking=True)
            states=refs[1000].annotation_states(b.tokenizer.encode(rendered,add_special_tokens=False))
            branch='target' if view.role=='target' else 'shared'
            counts[view.role]+=1
            for n,ref in refs.items():
                payload=ref.pooler(states,*SLOTS[view.role])
                sums[n][view.role].add_(payload.double().mean(0));payloads[n][branch].append(payload.detach())
            if branch not in examples:examples[branch]=(view.role,states)
            if i%16==0 or i==len(views):print(json.dumps(dict(event='calibration',views=i,total=len(views))),flush=True)
        directions={};edits={};export_hashes={}
        for n,ref in refs.items():
            check(ref.fingerprint()==parents[n][1]['reference_id'],'calibration mutated reference')
            role_means={role:sums[n][role]/counts[role] for role in SLOTS}
            branch_means={'target':role_means['target'],'shared':torch.stack([v for role,v in role_means.items() if role!='target']).mean(0)}
            edits[n]={};directions[n]={}
            for branch,mean in branch_means.items():
                check(bool(mean.isfinite().all()) and float(mean.norm())>1e-12,'degenerate common mode')
                u=mean/mean.norm();directions[n][branch]=u
                lift=getattr(ref.pooler,branch).lift
                rw,rb=deflate_affine(lift.weight,lift.bias,u)
                edits[n][branch]=dict(calibration_mean_norm=float(mean.norm()),
                    mean_absolute_payload_cosine_before=float(torch.cat(payloads[n][branch]).double().matmul(u).abs().mean()/32),
                    weight_relative_delta=float((rw.double()-lift.weight.double()).norm()/lift.weight.double().norm()),
                    bias_delta_norm=float((rb.double()-lift.bias.double()).norm()),
                    weight_orthogonal_residual=float((u@rw.double()).norm()/rw.double().norm()),
                    bias_orthogonal_residual=float(abs(u@rb.double())/rb.double().norm().clamp_min(1e-30)))
                check(edits[n][branch]['weight_orthogonal_residual']<1e-6 and edits[n][branch]['bias_orthogonal_residual']<1e-6,'projection integrity')
                lift.weight.copy_(rw);lift.bias.copy_(rb)
            allowed={f'pooler.{branch}.lift.{field}' for branch in ('shared','target') for field in ('weight','bias')}
            changed=[];scalars=0
            for name,value in ref.state_dict().items():
                old=originals[n][name];left=value.contiguous().reshape(-1).view(torch.uint8);right=old.contiguous().reshape(-1).view(torch.uint8)
                if not torch.equal(left,right):
                    check(name in allowed,'nonlift reference mutation');changed.append(name)
                    scalars+=int((left.reshape(-1,4)!=right.reshape(-1,4)).any(-1).sum())
            check(set(changed)==allowed,'unexpected changed tensor set')
            for branch,(role,states) in examples.items():
                actual=ref.pooler(states,*SLOTS[role]);u=directions[n][branch]
                residual=float((actual.double()@u).norm()/actual.double().norm())
                check(residual<1e-5,'postedit pooling orthogonality')
                edits[n][branch]['actual_output_orthogonal_residual']=residual
            ref.freeze()
            edits[n].update(changed_tensors=sorted(changed),changed_scalar_values=scalars,
                parent_reference_id=parents[n][1]['reference_id'],reference_id=ref.artifact_id,
                all_nonedited_reference_tensors_byte_identical=True)
            export_hashes[n]=export_reference(output/f'reference-{n:04d}',ref,
                dict(weight_repair_protocol=PROTOCOL,weight_repair_contract=digest(CONTRACT.read_bytes()),
                     parent_manifest_sha256=digest((parents[n][0]/'manifest.json').read_bytes()),
                     parent_reference_id=parents[n][1]['reference_id'],parent_window=n,
                     optimizer_updates=0,weight_repair_optimizer_updates=0,training_release=train.identity,
                     calibration_sha256=digest(canonical(calibration)),scope='one programmed repair; no training or live promotion'))
            saved=load_file(str(output/f'reference-{n:04d}'/'weights.safetensors'),device='cuda')
            check(state_fingerprint({k.removeprefix('reference.'):v for k,v in saved.items()})==ref.artifact_id,'export readback identity')
            del saved
        save_file({f'{n}.{branch}':u.contiguous() for n,values in directions.items() for branch,u in values.items()},str(output/'directions.safetensors'))
        (output/'edits.json').write_bytes(canonical(edits))
        print(json.dumps(dict(event='one_candidate_fixed',changed_tensors_per_artifact=4,optimizer_updates=0)),flush=True)
        # Both repaired artifacts are fixed/exported before any development feedback.
        dev=load_release('data/training-collections-20261105/releases/development',expected_hash=DEV_SHA)
        results=[]
        for n,ref in refs.items():
            print(json.dumps(dict(event='evaluation_started',window=n)),flush=True)
            q=evaluate(ref,dev);q.update(reference_bundle_sha256=export_hashes[n],weight_repair_protocol=PROTOCOL,
                parent_window=n,parent_reference_id=parents[n][1]['reference_id'])
            path=output/f'qualification-{n:04d}.json';path.write_bytes(canonical(q))
            accepted=False;rejection=None
            try:
                evidence=load_reference_evidence(path,dev,expected_hash=digest(path.read_bytes()));evidence.assert_reference(ref);accepted=True
            except ValueError as error:
                check(str(error)=='reference qualification did not pass','unexpected evidence rejection');rejection=str(error)
            record=dict(window=n,status=q['status'],recovery=q['recovery'],contrast=q['contrast'],
                parent_recovery=baselines[n]['recovery'],parent_contrast=baselines[n]['contrast'],
                loader_accepted=accepted,loader_rejection=rejection,report_sha256=digest(path.read_bytes()),reference_id=ref.artifact_id)
            results.append(record);print(json.dumps(dict(event='evaluation_finished',**record)),flush=True)
        live_after=state_fingerprint({**dict(b.named_parameters()),**{'codec.'+n:p for n,p in c.state_dict().items()}})
        base=frozen_check(b,args.snapshot)
        check(live_before==live_after and base['pass'] and base['base_tied_storage'],'live/base mutation')
        check(all(p.grad is None for ref in refs.values() for p in ref.parameters()) and
              all(p.grad is None for p in list(b.parameters())+list(c.parameters())),'gradient contamination')
        for n,ref in refs.items():check(ref.fingerprint()==edits[n]['reference_id'],'postevaluation reference mutation')
        for path,m in parents.values():check(digest((path/'weights.safetensors').read_bytes())==m['files']['weights.safetensors'],'parent file mutation')
        check(all(digest((releases/split/'manifest.json').read_bytes())==sha for split,sha in release_hashes.items()),'release mutation')
        check(all(digest(Path(name).read_bytes())==sha for name,sha in sources.items()),'run source mutation')
        passed=all(q['loader_accepted'] for q in results)
        report=dict(protocol=PROTOCOL,status='PASSED' if passed else 'FAILED',paired_recovery_criterion_passed=passed,
            candidate_rules_tried=1,repair_applications=2,optimizer_updates=0,capability_optimizer_updates=0,
            gpu_uuid='GPU-225509c3-d702-6c27-de03-76b5e99b33b1',profile='v3/graph-v1',release_hashes=release_hashes,
            gradient_updates=0,new_source_records=0,agents=0,calibration_views=len(views),calibration_counts=counts,
            annotation_W_prefills=len(views),calibration_pooler_forwards=2*len(views),postedit_pooler_checks=4,
            live_before_sha256=live_before,live_after_sha256=live_after,live_unchanged=True,frozen_base=base,
            parent_artifacts_unchanged=True,all_nonedited_reference_tensors_byte_identical=True,owner_gradients_none=True,
            results=results,peak_allocated=torch.cuda.max_memory_allocated(),elapsed_seconds=time.monotonic()-started,
            source_hashes=sources,contract_sha256=digest(CONTRACT.read_bytes()),
            scope='one fixed gradient-free reference repair; no learned-flow certification, live initialization or training')
        (output/'report.json').write_bytes(canonical(report));print(json.dumps(report),flush=True)
        return 0 if passed else 2


if __name__=='__main__':raise SystemExit(main())
