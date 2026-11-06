"""Bounded A4 reference-only fit. No capability phases or data generation."""
import argparse
import json
import time
from pathlib import Path

from qwen_one_two.training.inputs import canonical,digest,check,SLOTS
from qwen_one_two.training.release import load_release

TRAIN_SHA='c70d4f44135a34c00e7fa0348833b1ec56613def15f5753231589492ed35478f'
DEV_SHA='5b8f4cffdc4f33e255d3a34306cac03a68c50dcd9dab905ae8d8adcf89c81e77'


def source_snapshot(output):
    root=Path(__file__).resolve().parents[1]
    paths=sorted(set((root/'qwen_one_two/training').glob('*.py'))|
                 set((root/'qwen_one_two/codec').glob('*.py'))|
                 set((root/'qwen_one_two').glob('g1_*.py'))|
                 {root/'scripts/fit_reference.py',root/'scripts/qualify_reference.py',root/'scripts/qualify_recurrence.py'})
    hashes={}
    for path in paths:
        name=str(path.relative_to(root));body=path.read_bytes();hashes[name]=digest(body)
        target=output/'source-snapshot'/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(body)
    (output/'source-snapshot/manifest.json').write_bytes(canonical(hashes))
    return hashes


def export_reference(path,reference,provenance):
    from safetensors.torch import save_file
    check(reference.frozen and reference.artifact_id==reference.fingerprint(),'frozen export identity')
    check(not path.exists(),'never overwrite a reference export')
    path.mkdir(parents=True)
    state={'reference.'+name:value.detach().contiguous().clone() for name,value in reference.state_dict().items()}
    save_file(state,str(path/'weights.safetensors'))
    manifest=dict(provenance,format='qot-frozen-reference-v1',reference_frozen=True,
                  reference_id=reference.artifact_id,capability_training=False,
                  files={'weights.safetensors':digest((path/'weights.safetensors').read_bytes())})
    (path/'manifest.json').write_bytes(canonical(manifest))
    return digest((path/'manifest.json').read_bytes())


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--snapshot',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--resume')
    parser.add_argument('--profile',choices=('legacy-v1','recovery-v1'),default='recovery-v1')
    parser.add_argument('--initial-reference',help='Explicit unqualified reference-only warm start; fresh optimizer')
    args=parser.parse_args()
    output=Path(args.output);output.mkdir(parents=True,exist_ok=False)
    train=load_release('data/training-collections-20261105/releases/train',expected_hash=TRAIN_SHA)
    dev=load_release('data/training-collections-20261105/releases/development',expected_hash=DEV_SHA)
    import torch
    from torch.nn.attention import sdpa_kernel,SDPBackend
    from qwen_one_two.g1_proof import configure
    from qwen_one_two.codec.qualify import load
    from qwen_one_two.g1_evidence import frozen_check
    from qwen_one_two.training.reference import Reference,state_fingerprint
    from qwen_one_two.training.binding import wire_payload
    from qwen_one_two.training.component_producer import reference_record
    from qwen_one_two.training.engine import Window,reference_groups
    from qwen_one_two.training.sampler import Sampler
    from qwen_one_two.training.reference_data import catalogue,ViewSampler
    from qwen_one_two.training.checkpoint import save,bundle_tensors,restore_model
    from qwen_one_two.training.reference_qualification import evaluate,load_reference_evidence
    from qwen_one_two.training.manifest import BASE_SHA,file_hash
    check(file_hash(Path(args.snapshot)/'model.safetensors-00001-of-00001.safetensors')==BASE_SHA,'base hash mismatch')
    configure();started=time.monotonic()
    with sdpa_kernel(SDPBackend.MATH):
        b,c,_=load(args.snapshot,'data/g1-proof/gate-repair-v2-20261105/diagnostic.safetensors')
        ref=Reference(b,c);ref.assert_isolated(c)
        warm_start=None
        if args.initial_reference:
            check(not args.resume,'warm start and exact resume are distinct')
            from safetensors.torch import load_file
            path=Path(args.initial_reference);manifest=json.loads((path/'manifest.json').read_bytes())
            check(manifest['format']=='qot-frozen-reference-v1','reference-only warm start required')
            check(digest((path/'weights.safetensors').read_bytes())==manifest['files']['weights.safetensors'],'warm-start file hash')
            ref.load_state_dict({k.removeprefix('reference.'):v for k,v in load_file(str(path/'weights.safetensors'),device='cuda').items()})
            check(ref.fingerprint()==manifest['reference_id'],'warm-start tensor identity')
            warm_start={'manifest_sha256':digest((path/'manifest.json').read_bytes()),'reference_id':ref.fingerprint(),
                        'prior_optimizer_updates':manifest['optimizer_updates'],'optimizer_reset':True}
        live_before=state_fingerprint({**dict(b.named_parameters()),**{'codec.'+n:p for n,p in c.state_dict().items()}})
        sources=source_snapshot(output)
        views=catalogue(train) if args.profile=='recovery-v1' else None
        sampler=ViewSampler(views,train.identity) if views is not None else Sampler.from_release(train,'reference')
        groups=reference_groups(ref)
        for group in groups:
            group['named_parameters']=[('reference.'+name,p) for name,p in group['named_parameters']]
        window=Window(groups,phase='reference')
        live_ids={id(p) for p in b.parameters()}|{id(p) for p in c.parameters()}
        check(not live_ids & {id(p) for p in window.parameters},'live/reference optimizer overlap')
        identities={'base':BASE_SHA,'qev':digest(Path('data/qev/release_manifest.json').read_bytes()),
                    'protocol':digest(Path('docs/neural-input-encoding-v1.md').read_bytes()),
                    'precision':digest(Path('docs/codec-feedback-precision-v3.md').read_bytes()),
                    'reference':ref.fingerprint(),'dataset':train.identity,
                    'sources':digest(canonical({'files':sources,'profile':args.profile,'catalogue':sampler.catalogue_hash,
                        'revision':digest(Path('docs/reference-recovery-revision-v1.md').read_bytes()) if views is not None else None}))}
        prepared={};feature_cache={}
        # Frozen-W feature extraction only; pooler/T remain live and are never
        # cached. Duplicate canonical meanings still retain all scheduled IDs.
        if views is not None:
            entries=[(view.identity,view) for view in views]
            (output/'reference-views.json').write_bytes(canonical([dict(id=v.identity,role=v.role,operation=v.operation,
                annotation_sha256=digest(v.content),sources=v.sources) for v in views]))
        else:
            entries=[(entry['id'],reference_record(train,train.record(entry['id']))[0])
                     for entry in train.sampler_rows() if entry['category']=='reference']
        for key,annotation in entries:
            text=annotation.content.decode()
            targets=b.tokenizer.encode(text,add_special_tokens=False)+[248046]
            if annotation.sha256 not in feature_cache:
                rendered=b.tokenizer.apply_chat_template([{'role':'user','content':text}],tokenize=False,
                    add_generation_prompt=True,enable_thinking=True)
                ids=b.tokenizer.encode(rendered,add_special_tokens=False)
                feature_cache[annotation.sha256]=ref.annotation_states(ids)
            states=feature_cache[annotation.sha256]
            check(len(targets)+SLOTS[annotation.role][1]<=512,'reference reconstruction exceeds ceiling')
            prepared[key]=(annotation,targets,states,states._version)
        check(len(prepared)==(3376 if views is not None else 500),'pinned reference catalogue size drift')
        if args.resume:
            old=json.loads((Path(args.resume)/'manifest.json').read_bytes())
            # Initial reference identity is run lineage, not current tensor hash.
            identities['reference']=old['identities']['reference']
            restored=restore_model(args.resume,b,c,ref,identities,window=window)
            sampler.load_state_dict(restored['sampler'])
        else:
            # Exercise the actual full reference ownership/export/resume path
            # before the first update, not after 100 windows of fitting.
            initial=output/'checkpoint-0000'
            save(initial,window,bundle_tensors(b,c,ref),identities,sampler.state_dict(),
                 reference_frozen=False,reference_id=None)
            restored=restore_model(initial,b,c,ref,identities,window=window)
            sampler.load_state_dict(restored['sampler'])
        run={'scope':'reference only; no capability training','train_release':train.identity,'development_release':dev.identity,
             'profile':args.profile,'warm_start':warm_start,'catalogue_hash':sampler.catalogue_hash,
             'target_history':'alternate full preceding targets / zero preceding-target vectors' if views is not None else 'full teacher forcing',
             'reference_revision_contract':digest(Path('docs/reference-recovery-revision-v1.md').read_bytes()) if views is not None else None,
             'recurrence_backend':b.recurrence_backend,
             'scheduled_windows':1000,'examples_per_window':8,'training_seed':17,'reference_records':len(prepared),
             'frozen_W_feature_cache_entries':len(feature_cache),'cache_device':'cuda',
             'contrast_training':'masked: no explicit reviewed training reference contrast relations supplied',
             'source_hashes':sources,'identities':identities,'resume':args.resume}
        (output/'run.json').write_bytes(canonical(run))
        print(json.dumps(dict(event='prepared',**run)),flush=True)
        qualifications=[]
        torch.cuda.reset_peak_memory_stats()
        with (output/'windows.jsonl').open('x') as log:
            while window.scheduled_windows<1000:
                ids=sampler.next_window();loss_values=[];histories=[];step_start=time.monotonic()
                for ordinal,key in enumerate(ids):
                    annotation,targets,states,version=prepared[key]
                    check(states._version==version,'mutated frozen feature cache')
                    payload=ref.pooler(states,*SLOTS[annotation.role])
                    prefix=wire_payload(ref,payload,annotation.role)
                    use_history=views is None or (ordinal+window.scheduled_windows)%2==0
                    histories.append(use_history)
                    loss=ref.reconstruct_teacher_forced(prefix,targets,target_history=use_history)
                    loss_values.append(float(loss.detach()))
                    window.add_episode({'reconstruction':loss},complete=True,positions=len(prefix)+len(targets))
                    del loss,prefix,payload
                report=window.step()
                check(not report['failed_exposures'] and report['eligible_examples']=={'reconstruction':8},'reference exposure eligibility')
                check(all(p.grad is None for p in b.parameters()) and all(p.grad is None for p in c.parameters()),'live gradient contamination')
                record=dict(report,samples=ids,target_history=histories,ablated_history_exposures=histories.count(False),
                            mean_reconstruction=sum(loss_values)/8,seconds=time.monotonic()-step_start,
                            elapsed_seconds=time.monotonic()-started,peak_allocated=torch.cuda.max_memory_allocated(),
                            optimizer_updates=window.optimizer_updates)
                log.write(json.dumps(record)+'\n');log.flush()
                if window.scheduled_windows==1 or window.scheduled_windows%10==0:
                    print(json.dumps({k:record[k] for k in ('window','mean_reconstruction','seconds','elapsed_seconds','peak_allocated')}),flush=True)
                if window.scheduled_windows%100:continue
                number=window.scheduled_windows
                save(output/f'checkpoint-{number:04d}',window,bundle_tensors(b,c,ref),identities,sampler.state_dict(),
                     reference_frozen=False,reference_id=None)
                # Independent frozen evaluation copy; preserve the next update's RNG.
                with torch.random.fork_rng(devices=[0]):
                    evaluated=Reference(b,c);evaluated.load_state_dict(ref.state_dict());evaluated.freeze()
                    export_hash=export_reference(output/f'reference-{number:04d}',evaluated,
                        dict(window=number,optimizer_updates=window.optimizer_updates,training_release=train.identity,
                             development_release=dev.identity,sources=identities['sources']))
                    evaluation=evaluate(evaluated,dev)
                    evaluation['reference_bundle_sha256']=export_hash
                    evaluation['reference_training_updates']=window.optimizer_updates
                    path=output/f'qualification-{number:04d}.json';path.write_bytes(canonical(evaluation))
                    if evaluation['status']=='PASSED':
                        evidence=load_reference_evidence(path,dev,expected_hash=digest(path.read_bytes()))
                        evidence.assert_reference(evaluated)
                    qualifications.append({'window':number,'status':evaluation['status'],
                        'recovery':evaluation['recovery'],'contrast':evaluation['contrast'],
                        'report_sha256':digest(path.read_bytes()),'reference_sha256':evaluated.artifact_id})
                    print(json.dumps(dict(event='qualification',**qualifications[-1])),flush=True)
                    del evaluated
        live_after=state_fingerprint({**dict(b.named_parameters()),**{'codec.'+n:p for n,p in c.state_dict().items()}})
        frozen=frozen_check(b,args.snapshot)
        check(live_before==live_after and frozen['pass'],'live/base weights changed during reference fitting')
        final_two=[q for q in qualifications if q['window'] in (900,1000)]
        promoted=len(final_two)==2 and all(q['status']=='PASSED' for q in final_two)
        result=dict(status='PASSED' if promoted else 'BLOCKED',phase='reference',qualifications=qualifications,
            final_two_passed=promoted,optimizer_updates=window.optimizer_updates,scheduled_windows=window.scheduled_windows,
            live_parameters_unchanged=live_before==live_after,live_before_sha256=live_before,live_after_sha256=live_after,
            frozen_base=frozen,capability_optimizer_updates=0,reference_fitting_only=True,
            peak_allocated=torch.cuda.max_memory_allocated(),elapsed_seconds=time.monotonic()-started)
        (output/'report.json').write_bytes(canonical(result));print(json.dumps(result),flush=True)
        return 0 if promoted else 2


if __name__=='__main__':raise SystemExit(main())
