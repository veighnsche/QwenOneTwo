"""One bounded block-replacement search; existing data, no gradient training."""
import argparse
from collections import Counter
import json
from pathlib import Path
import time
from qwen_one_two.training.inputs import canonical,digest,check,SLOTS
from qwen_one_two.training.diagnostics import (PROTOCOL,BLOCKS,CONTROL_IDS,MASKS,choose_cases,
    ranking,eligible,BlockBank,Evaluator,load_resume)
from scripts.fit_reference import TRAIN_SHA,DEV_SHA,source_snapshot,export_reference

ROOT=Path('data/reference-encoder-20261106')
CONTRACT=Path('docs/reference-block-search-resume-v1.md')
LOCKED_SHA='d13594f18738056ee179b3498d54d8261f7f7654aad633fabb776810681aa1cf'


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--snapshot',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--resume');parser.add_argument('--resume-receipt')
    args=parser.parse_args();check(bool(args.resume)==bool(args.resume_receipt),'resume requires its evidence receipt')
    output=Path(args.output);output.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    resumed=load_resume(args.resume,args.resume_receipt) if args.resume else None
    def save(name,value):(output/name).write_bytes(canonical(value))
    log=(output/'events.jsonl').open('x')
    def emit(event):
        event=dict(event,elapsed_seconds=time.monotonic()-started)
        log.write(json.dumps(event)+'\n');log.flush()
        if event['event'] not in ('sensitivity','recovery'):print(json.dumps(event),flush=True)
    receipt=json.loads(Path('docs/evidence/reference-encoder-v2.json').read_bytes())
    bundles={'baseline':ROOT/'run/reference-1000','donor':ROOT/'partition-probe/reference-0064',
             'paired':ROOT/'run/reference-0900'};manifests={};assets={}
    for label,path in bundles.items():
        raw=(path/'manifest.json').read_bytes();m=json.loads(raw)
        check(digest(raw)==receipt['assets'][str(path/'manifest.json')],'pinned parent manifest')
        check(m['format']=='qot-frozen-reference-v1' and m['reference_frozen'] and m['pooler_kind']=='target-v2','parent structure')
        check(digest((path/'weights.safetensors').read_bytes())==m['files']['weights.safetensors'],'parent weight file')
        manifests[label]=m;assets[str(path/'manifest.json')]=digest(raw)
    windows=ROOT/'run/windows.jsonl';check(digest(windows.read_bytes())==receipt['assets'][str(windows)],'exposure evidence hash')
    exposures=Counter();window_count=0
    for line in windows.read_text().splitlines():
        record=json.loads(line);exposures.update(record['samples']);window_count+=1
    check(window_count==1000 and sum(exposures.values())==8000,'prior exposure accounting')
    assets[str(windows)]=digest(windows.read_bytes())
    releases=Path('data/training-collections-20261105/releases')
    release_hashes={split:digest((releases/split/'manifest.json').read_bytes()) for split in ('train','development','locked')}
    check(release_hashes==dict(train=TRAIN_SHA,development=DEV_SHA,locked=LOCKED_SHA),'release manifest drift')
    from qwen_one_two.training.release import load_release
    from qwen_one_two.training.component_producer import reference_record
    train=load_release(releases/'train',expected_hash=TRAIN_SHA);records=train.records;rows=[];annotations={}
    for entry in train.manifest['records']:
        if not entry['selected']:continue
        row=records[entry['id']]
        if row.get('training_category')!='reference':continue
        a,_=reference_record(train,row);annotations[row['id']]=a
        rows.append(dict(id=row['id'],role=a.role,family=row['family'],annotation_sha256=a.sha256,
                         source_sha256=digest(canonical(row)),source_path=['model_inputs','meaning'],
                         prior_broad_exposures=exposures[a.sha256]))
    splits=choose_cases(rows);donor_counts=Counter(CONTROL_IDS[i%6] for i in range(512))
    for items in splits.values():
        for row in items:row['prior_donor_exposures']=donor_counts[row['id']]
    save('cases.json',dict(training_release=train.identity,splits=splits,scope='two disjoint search sets; both previously training data'))
    if resumed:
        check(resumed['cases']==json.loads((output/'cases.json').read_bytes()),'resume cases changed')
        check(resumed['report']['parent_assets']==assets and resumed['report']['release_hashes']==release_hashes,'resume source/data changed')
        save('resume.json',dict(path=args.resume,receipt=args.resume_receipt,receipt_sha256=resumed['receipt_sha256'],
            report_sha256=resumed['report_sha256'],completed_masks=sorted(resumed['search']),
            partial_recovery_counts={m:len(v) for m,v in resumed['partial'].items()},
            previous_source_hashes=resumed['report']['source_hashes']))
    sources=source_snapshot(output)
    for name in ('scripts/diagnose_reference.py',str(CONTRACT),'tests/test_reference_diagnostics.py','tests/test_reference_diagnostics_gpu.py'):
        body=Path(name).read_bytes();sources[name]=digest(body);p=output/'source-snapshot'/name
        p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(body)
    save('source-snapshot/manifest.json',sources)
    emit(dict(event='preflight_finished',cases={k:len(v) for k,v in splits.items()},candidate_masks=list(MASKS)))
    import torch
    from safetensors.torch import load_file
    from torch.nn.attention import sdpa_kernel,SDPBackend
    from qwen_one_two.g1_proof import configure
    from qwen_one_two.codec.qualify import load
    from qwen_one_two.g1_evidence import frozen_check
    from qwen_one_two.training.manifest import BASE_SHA,file_hash
    from qwen_one_two.training.reference import Reference,state_fingerprint
    from qwen_one_two.training.reference_qualification import evaluate,load_reference_evidence,source_hashes
    check(file_hash(Path(args.snapshot)/'model.safetensors-00001-of-00001.safetensors')==BASE_SHA,'base file hash')
    configure();torch.cuda.reset_peak_memory_stats()
    with sdpa_kernel(SDPBackend.MATH),torch.no_grad():
        b,c,_=load(args.snapshot,'data/g1-proof/gate-repair-v2-20261105/diagnostic.safetensors')
        live=lambda:state_fingerprint({**dict(b.named_parameters()),**{'codec.'+n:p for n,p in c.state_dict().items()}})
        live_before=live();ref=Reference(b,c,pooler_kind='target-v2')
        def state(label):return {k.removeprefix('reference.'):v for k,v in load_file(str(bundles[label]/'weights.safetensors'),device='cuda').items()}
        baseline=state('baseline');donor=state('donor')
        check(state_fingerprint(baseline)==manifests['baseline']['reference_id'] and
              state_fingerprint(donor)==manifests['donor']['reference_id'],'loaded source fingerprints')
        bank=BlockBank(baseline,donor);ref.load_state_dict(baseline);ref.freeze(manifests['baseline']['reference_id'])
        decoder_before=ref.decoder_fingerprint();prepared={}
        annotation_positions=0
        for row in splits['search']+splits['screen']:
            text=annotations[row['id']].content.decode()
            rendered=b.tokenizer.apply_chat_template([{'role':'user','content':text}],tokenize=False,
                add_generation_prompt=True,enable_thinking=True)
            ids=b.tokenizer.encode(rendered,add_special_tokens=False)
            prepared[row['id']]=dict(states=ref.annotation_states(ids),targets=b.tokenizer.encode(text,add_special_tokens=False)+[248046],
                                     encoding_ids_sha256=digest(canonical(ids)),annotation_positions=len(ids))
            annotation_positions+=len(ids)
        save('feature-identities.json',dict(base_sha256=BASE_SHA,profile='v3/graph-v1',
            chat_template_sha256=digest(canonical(b.tokenizer.chat_template)),
            source_hashes=source_hashes(),cases={k:{n:v for n,v in p.items() if n!='states'} for k,p in prepared.items()}))
        if resumed:check(resumed['features']==json.loads((output/'feature-identities.json').read_bytes()),'resume feature/model identity changed')
        evaluator=Evaluator(ref,prepared,emit);search={} if not resumed else dict(resumed['search']);screen={}
        edits={} if not resumed else {int(k):v for k,v in resumed['edits'].items()}
        fingerprints={r['reference_id']:m for m,r in search.items()};deduplicated=[]
        reused_masks=set(search)
        def install(mask):
            change=bank.apply(ref.state_dict(),mask);ref.freeze();edits[mask]=dict(change,reference_id=ref.artifact_id)
            return ref.artifact_id
        for mask in MASKS:
            if mask in reused_masks:
                (output/f'search-{mask:02d}.json').write_bytes((Path(args.resume)/f'search-{mask:02d}.json').read_bytes())
                emit(dict(event='candidate_reused',mask=mask,stage='search',reference_id=search[mask]['reference_id'],
                          report_sha256=digest((output/f'search-{mask:02d}.json').read_bytes())))
            else:
                identity=install(mask)
                if identity in fingerprints:
                    other=fingerprints[identity];search[mask]=dict(search[other],mask=mask)
                    deduplicated.append(dict(mask=mask,duplicate_of=other));emit(dict(event='duplicate',mask=mask,duplicate_of=other))
                else:
                    fingerprints[identity]=mask;emit(dict(event='candidate_started',mask=mask,stage='search',blocks=edits[mask]['blocks']))
                    partial={} if not resumed else resumed['partial'].get(mask,{})
                    if partial:check(identity==resumed['edits'][str(mask)]['reference_id'],'partial candidate identity changed')
                    search[mask]=evaluator.measure(mask,'search',splits['search'],recoveries=partial)
                save(f'search-{mask:02d}.json',search[mask])
            save('edits.json',edits)
            controls=[x for x in search[mask]['cases'] if x['id'] in CONTROL_IDS]
            if mask==0:check(sum(x['recovery']['exact'] for x in controls)==2,'baseline six-case evidence drift')
            if mask==31:
                check(search[mask]['reference_id']==manifests['donor']['reference_id'] and all(x['recovery']['exact'] for x in controls),
                      'donor six-case positive evidence failed')
        # Shortlist by the complete search set only, preserving every measured result.
        unique_masks=[m for m in MASKS if m and m not in {d['mask'] for d in deduplicated}]
        shortlist=sorted(unique_masks,key=lambda m:ranking(search[m]))[:3]
        save('shortlist.json',dict(masks=shortlist,selection='search exact, margin, swap effect, fewer blocks, mask'))
        for mask in (0,*shortlist):
            install(mask);emit(dict(event='candidate_started',mask=mask,stage='screen'))
            screen[mask]=evaluator.measure(mask,'screen',splits['screen']);save(f'screen-{mask:02d}.json',screen[mask])
        complete=len(search)==32 and len(screen)==4
        check(complete,'finish the entire agreed candidate and screening set')
        eligible_masks=[m for m in shortlist if complete and eligible(search[m],screen[m],search[0],screen[0])]
        winner=min(eligible_masks,key=lambda m:ranking(screen[m])) if eligible_masks else None
        selection=dict(complete=complete,shortlist=shortlist,eligible=eligible_masks,winner=winner,
                       reason='screen gain without any baseline-recovery loss' if winner is not None else
                              'no shortlisted candidate met retention and strict screen-gain criteria')
        save('selection.json',selection);emit(dict(event='selection',**selection))
        # Export the search winner before loading development labels. No next-best retry.
        qualifications=[];export_hashes={}
        if winner is not None:
            install(winner)
            for n in (1000,900):
                if n==900:
                    paired=state('paired');check(state_fingerprint(paired)==manifests['paired']['reference_id'],'paired parent identity')
                    pairbank=BlockBank(paired,donor);ref.load_state_dict(paired)
                    paired_edit=pairbank.apply(ref.state_dict(),winner);ref.freeze();save('paired-edit.json',dict(paired_edit,reference_id=ref.artifact_id))
                export_hashes[n]=export_reference(output/f'reference-{n:04d}',ref,dict(
                    protocol=PROTOCOL,mask=winner,parent_window=n,parent_reference_id=manifests['baseline' if n==1000 else 'paired']['reference_id'],
                    donor_reference_id=manifests['donor']['reference_id'],gradient_updates=0,optimizer_updates=0,
                    search_cases_sha256=digest((output/'cases.json').read_bytes()),contract_sha256=sources[str(CONTRACT)],
                    scope='selected by existing training data; unqualified until original paired gate passes'))
            dev=load_release(releases/'development',expected_hash=DEV_SHA)
            for n in (900,1000):
                bundle=output/f'reference-{n:04d}'
                ref.load_state_dict({k.removeprefix('reference.'):v for k,v in load_file(str(bundle/'weights.safetensors'),device='cuda').items()});ref.freeze()
                q=evaluate(ref,dev);q.update(reference_bundle_sha256=export_hashes[n],weight_search_protocol=PROTOCOL,mask=winner)
                path=output/f'qualification-{n:04d}.json';save(path.name,q);accepted=False;rejection=None
                try:
                    evidence=load_reference_evidence(path,dev,expected_hash=digest(path.read_bytes()));evidence.assert_reference(ref);accepted=True
                except ValueError as error:
                    check(str(error)=='reference qualification did not pass','unexpected qualification rejection');rejection=str(error)
                qualifications.append(dict(window=n,status=q['status'],recovery=q['recovery'],contrast=q['contrast'],
                    loader_accepted=accepted,loader_rejection=rejection,report_sha256=digest(path.read_bytes()),reference_id=ref.artifact_id))
                emit(dict(event='qualification',**qualifications[-1]))
        # Finish on the exact original baseline.
        bank.apply(ref.state_dict(),0);ref.freeze(manifests['baseline']['reference_id'])
        check(ref.decoder_fingerprint()==decoder_before,'decoder drift')
        check(state_fingerprint(bank.baseline)==manifests['baseline']['reference_id'] and
              state_fingerprint(bank.donor)==manifests['donor']['reference_id'],'immutable source mutation')
        frozen=frozen_check(b,args.snapshot);live_after=live()
        check(frozen['pass'] and frozen['base_tied_storage'] and live_before==live_after,'base/live mutation')
        check(all(p.grad is None for p in list(ref.parameters())+list(b.parameters())+list(c.parameters())),'gradient contamination')
        for label,path in bundles.items():check(digest((path/'weights.safetensors').read_bytes())==manifests[label]['files']['weights.safetensors'],'parent file mutation')
        check(all(digest((releases/k/'manifest.json').read_bytes())==v for k,v in release_hashes.items()),'release mutation')
        check(all(digest(Path(k).read_bytes())==v for k,v in sources.items()),'source drift')
        status=('INCOMPLETE' if not complete else 'NO_FINALIST' if winner is None else
                'QUALIFIED' if all(q['loader_accepted'] for q in qualifications) else 'QUALIFICATION_FAILED')
        result=dict(protocol=PROTOCOL,status=status,search_complete=complete,candidate_budget=32,
            unique_candidates=len(fingerprints),completed_search_candidates=len(search),completed_screen_candidates=len(screen),
            reused_search_candidates=len(reused_masks),new_search_candidates=len(search)-len(reused_masks),
            reused_partial_recovery_cases=evaluator.reused_recoveries,
            prior_report_sha256=None if not resumed else resumed['report_sha256'],
            deduplicated=deduplicated,selection=selection,qualifications=qualifications,
            gpu_uuid='GPU-225509c3-d702-6c27-de03-76b5e99b33b1',profile='v3/graph-v1',
            search_seconds_limit=None,source_hashes=sources,parent_assets=assets,release_hashes=release_hashes,
            annotation_W_prefills=len(prepared),annotation_positions=annotation_positions,
            diagnostic_sensitivity_prefills=evaluator.sensitivity_calls,diagnostic_recovery_prefills=evaluator.recovery_calls,
            diagnostic_decoder_positions=evaluator.positions,qualification_accounting='separate original full suites in qualification reports',
            gradient_updates=0,optimizer_updates=0,capability_updates=0,new_source_records=0,agents=0,
            search_is_data_driven_optimization=True,baseline_restored=True,base=frozen,decoder_before_sha256=decoder_before,
            decoder_after_sha256=ref.decoder_fingerprint(),live_before_sha256=live_before,live_after_sha256=live_after,
            owner_gradients_none=True,immutable_sources_verified=True,parent_files_unchanged=True,
            peak_allocated=torch.cuda.max_memory_allocated(),elapsed_seconds=time.monotonic()-started,
            scope='bounded diagnostic weight search; no learned-flow certification or live promotion')
        save('edits.json',edits);save('report.json',result);emit(dict(event='finished',status=status,seconds=result['elapsed_seconds']))
        log.close();return 0 if status=='QUALIFIED' else 2


if __name__=='__main__':raise SystemExit(main())
