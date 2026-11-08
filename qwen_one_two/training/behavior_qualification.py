"""Read-only, outcome-blind latent receiver checks over existing reviewed pairs.

This certifies a reference/receiver configuration, not learned invocation, Reason
meaning, native answer consumption, or an entire training flow. No string match,
judge model, gold return, optimizer, or gradient execution is involved.
"""
from pathlib import Path
import math
import struct
from .inputs import canonical, check, digest
from .release import ReviewedRelease, decode, sha256
from .reference_qualification import PROFILE

PROTOCOL='reference-behavior-v1'
GPU='GPU-225509c3-d702-6c27-de03-76b5e99b33b1'
CATEGORIES={1:'Noul',2:'Noul',3:'Noul',4:'Noul',5:'Choice',6:'Choice',7:'Score',8:'Score',10:'Noul'}
PROMPT='Evaluate the internal request using only its supplied latent contents.'


def suite(release):
    check(isinstance(release,ReviewedRelease) and release.manifest['purpose']=='development',
          'behavior evaluation requires reviewed development data')
    pairs={};omitted=[]
    for entry in release.manifest['records']:
        if not entry['selected']:continue
        row=release.record(entry['id'])
        if row.get('evaluation_slice')!='X':continue
        info=row['intervention'];category=info['category']
        protocol='clean_component' if category in (4,9) else (None if category==6 else 'offline_semantic_intervention')
        check(info.get('protocol')==protocol and
              row['construction']['template_version']=='controlled-data-v1','unknown intervention protocol')
        if category==9:
            omitted.append(row['id']);continue  # Supplied-return native reader, a different task.
        check(category in CATEGORIES and row['operation']==CATEGORIES[category],'intervention operation')
        pair=pairs.setdefault(info['pair'],{})
        check(info['side'] in (0,1) and info['side'] not in pair,'duplicate intervention side')
        pair[info['side']]=row
    groups=[]
    for key,pair in sorted(pairs.items()):
        check(set(pair)=={0,1},'incomplete intervention pair')
        a,b=pair[0],pair[1]
        check(a['family']==b['family'] and a['intervention']['category']==b['intervention']['category'],
              'intervention family/category drift')
        groups.append({'id':key,'category':a['intervention']['category'],
                       'ids':[a['id'],b['id']],'operation':a['operation']})
    check({g['category'] for g in groups}==set(CATEGORIES),'missing receiver intervention category')
    return groups,omitted


def expected(row):
    """Reviewed targets stay external; never part of CleanView or Session prompt."""
    gold=row['supervision'].get('result')
    if gold is None:gold=row['labels']
    check(gold['operation']==row['operation'],'reviewed operation mismatch')
    return {'status':gold['status'],'payload':gold['payload']}


def fp32(value):
    # Scalar ABI is FP32: represent reviewed interval endpoints in that ABI,
    # with no added tolerance and no change to the stored reviewed annotation.
    return struct.unpack('f',struct.pack('f',value))[0]


def grade(row,case):
    gold=expected(row);actual=case.get('actual') or {}
    status=actual.get('status')==gold['status']
    valid=case.get('transport') is not None and all(case['transport'].get(k) is True for k in
        ('latent_request','latent_return','bound_snapshot','candidate_binding','consumed','single_prefill','cache_continuity','accounting'))
    value=False;payload=gold['payload'] or {}
    if status:
        if gold['status']!='Assessed':
            value=actual.get('value') is None and actual.get('selected') is None and actual.get('reason_present') is True
        elif row['operation']=='Choice':
            value=(actual.get('selected') in payload['acceptable_selection_ids'] and actual.get('value') is None
                   and actual.get('reason_present') is False)
        else:
            v=actual.get('value')
            if (type(v) in (int,float) and math.isfinite(v) and 0<=v<=1
                    and actual.get('selected') is None and actual.get('reason_present') is False):
                if row['operation']=='Noul':
                    # Truth decision, with calibrated probability reported separately.
                    value=v!=.5 and ((v>.5)==(payload['p_true']>.5))
                else:
                    lo,hi=payload['score_interval'];value=fp32(lo)<=v<=fp32(hi)
    return {'status_correct':status,'value_correct':bool(value),'transport_valid':valid,
            'passed':bool(valid and status and value and not case.get('failure'))}


def summarize(release,cases,*,unchanged):
    groups,omitted=suite(release);ids=[k for g in groups for k in g['ids']]
    check([c['id'] for c in cases]==ids,'missing/reordered/foreign behavioral cases')
    scores={c['id']:grade(release.record(c['id']),c) for c in cases}
    ops={}
    for op in ('Noul','Choice','Score'):
        keys=[k for k in ids if release.record(k)['operation']==op]
        ops[op]={'scheduled':len(keys),'passed':sum(scores[k]['passed'] for k in keys),
                 'status_correct':sum(scores[k]['status_correct'] for k in keys)}
    paired=[dict(g,passed=all(scores[k]['passed'] for k in g['ids'])) for g in groups]
    passed=sum(s['passed'] for s in scores.values());status=sum(s['status_correct'] for s in scores.values())
    transport=sum(s['transport_valid'] for s in scores.values());pairs=sum(g['passed'] for g in paired)
    blockers=[]
    # Existing component/intervention thresholds, not new wording or domain quotas.
    if not unchanged:blockers.append('evaluated weights changed')
    if transport!=len(ids):blockers.append('failed latent transport/binding/cache/accounting checks')
    if passed/len(ids)<.90:blockers.append('receiver semantic/behavior success below 90%')
    if any(v['passed']/v['scheduled']<.85 for v in ops.values()):blockers.append('per-operation success below 85%')
    if status/len(ids)<.95:blockers.append('status accuracy below 95%')
    if ops['Score']['passed']/ops['Score']['scheduled']<.90:blockers.append('reviewed Score agreement below 90%')
    if pairs/len(groups)<.90:blockers.append('paired intervention success below 90%')
    return {'status':'BLOCKED' if blockers else 'PASSED','blockers':blockers,'scheduled':len(ids),
            'passed':passed,'status_correct':status,'transport_valid':transport,'operations':ops,
            'pairs_scheduled':len(groups),'pairs_passed':pairs,'groups':paired,
            'omitted_reader_ids':omitted,'case_scores':scores}


def receiver_hash(backbone,codec):
    from .reference import state_fingerprint
    # W is separately pinned and checked by the CLI. Include all mutable live
    # parameters (R/D/row masters/heads), without rehashing W per case.
    state={'backbone.'+n:p for n,p in backbone.named_parameters() if any(
        x in n for x in ('.R_A','.R_B','.D_A','.D_B','master'))}
    state.update({'codec.'+n:p for n,p in codec.named_parameters()})
    return state_fingerprint(state)


def source_hashes():
    root=Path(__file__).resolve().parents[2]
    files=sorted((root/'qwen_one_two/codec').glob('*.py'))+sorted((root/'qwen_one_two/training').glob('*.py'))
    files+=sorted((root/'qwen_one_two').glob('g1_*.py'))
    files+=[root/'scripts/qualify_behavior.py']
    return {str(p.relative_to(root)):digest(p.read_bytes()) for p in files}


def evaluate_case(backbone,codec,reference,release,row,encoder,*,construction=None,input_profile='d-pooled-role-slots-v1'):
    import torch
    from .component_producer import component_view
    from .clean_packets import CleanPackets
    from .episodes import PORT_NAMES,STATUS_NAMES
    from ..codec.session import Session
    from ..codec.render import Packet
    from ..codec.writer import decode_result
    from ..codec.schema import Kind
    import uuid
    case={'id':row['id'],'record_sha256':digest(canonical(row)),'actual':None,'failure':None}
    s=None
    try:
        from .receiver_profiles import LEGACY,profile
        if input_profile!=LEGACY:
            spec=profile(input_profile,'component')
            check(construction is None and encoder.source_policy==spec['source_policy'] and encoder.role_policy==spec['roles'],
                  'new-profile behavior requires the declared unmodified component encoder')
        view,p,indices=component_view(release,row,intervention=True)
        if construction is not None:
            check(not torch.is_grad_enabled(), 'diagnostic construction requires no_grad')
            view,p,indices=construction.prepare(view,p,indices)
        run=str(uuid.uuid4());encoded=encoder.encode(view,p,run=run)
        factory=CleanPackets(encoded,view,p)
        case['view_sha256']=encoded['input_hash'];case['construction_audit']=encoded['audit']
        # Never include human_input.prompt: it can reveal question/target/evidence.
        rendered=backbone.tokenizer.apply_chat_template([{'role':'user','content':PROMPT}],
            tokenize=False,add_generation_prompt=True,enable_thinking=True)
        ids=backbone.tokenizer.encode(rendered,add_special_tokens=False)
        s=Session.open(backbone,codec,ids,128,override=512,run=run,profile=factory.profile);rt=s.runtime
        if input_profile!=LEGACY:
            case.update(input_profile=input_profile,startup={'raw':float(s.raw_budget.detach()),
                'predicted_budget':s.predicted_budget,'granted_budget':rt.budget,'override':s.override})
        evidence=[]
        for index,ref in enumerate(factory.sources):
            n=len(factory.registry.get(ref).token_ids)
            evidence.extend(factory.span(index,start,min(32,n-start)) for start in range(0,n,32))
        check(len(evidence)<=8,'component evidence exceeds eight spans; never crop')
        port=next(k for k,v in PORT_NAMES.items() if v==row['operation'])
        request=factory.request if construction is None else construction.request(factory,indices)
        packet=request(port,indices['question'],indices['target'],evidence=tuple(evidence),
            candidates=indices['candidates'],rubric=indices['rubric'],direction=indices['direction'],session=s)
        if construction is not None:case['input_construction']=construction.receipt
        _,evaluated,_=s.evaluate(packet)
        # Record unmodified head decisions even if Reason decoding later fails.
        case['predicted_status']=STATUS_NAMES[int(evaluated['status_logits'].argmax())+1]
        if 'value' in evaluated:case['predicted_value']=float(evaluated['value'])
        returned=decode_result(codec,rt,evaluated)
        result=rt.registry.get(rt.call.result,Kind.RESULT)
        snapshot=rt.registry.get(packet.snapshot,Kind.SNAPSHOT)
        candidates=() if snapshot.bank is None else rt.registry.get(snapshot.bank,Kind.BANK).candidates
        selected=None if result.selected is None else indices['identities'][candidates.index(result.selected)]
        case['actual']={'status':STATUS_NAMES[result.status],
            'value':None if result.scalar is None else float(result.scalar),'selected':selected,
            'reason_present':result.reason is not None}
        consumed=s.deliver(returned)
        summary=s.summary()
        case['transport']={'latent_request':isinstance(packet,Packet),'latent_return':isinstance(returned,Packet),
            'bound_snapshot':returned.snapshot==packet.snapshot==result.snapshot,
            'candidate_binding':result.selected is None or result.selected in candidates,
            'consumed':consumed,'single_prefill':s.prefills==1,'cache_continuity':s.state.lineage==1,
            'accounting':s.state.position==rt.position==sum(summary[k] for k in ('P','B','A','H'))}
        check(len(packet)+len(returned)>0,'empty latent exchange')
        s.close();rt.terminate('Completed',emit_end=False)
    except ValueError as exc:
        case['failure']=str(exc)
        if s is not None and s.runtime.terminal is None:s.runtime.terminate('Failed',usable=False,emit_end=False)
    if s is not None:case['runtime']=s.summary()
    case['score']=grade(row,case)
    return case


def evaluate(backbone,codec,reference,release,*,emit=None,reuse=()):
    import torch
    from .clean import CleanEncoder
    from ..codec.neural import cuda_only
    cuda_only()
    check(reference.frozen and all(not p.requires_grad for p in reference.parameters()),'frozen reference required')
    check(backbone.compensated and backbone.precision_parts==2 and backbone.feedback_precision=='fp32'
          and backbone.recurrence_backend=='graph-v1','qualified v3 arithmetic required')
    before=reference.fingerprint();live=receiver_hash(backbone,codec)
    check(before==reference.artifact_id,'reference drift')
    encoder=CleanEncoder(backbone,reference,checkpoint_hash=live,profile_hash=digest(PROFILE.encode()))
    groups,_=suite(release);cases=[]
    cache={c['id']:c for c in reuse}
    check(len(cache)==len(reuse),'duplicate reused cases')
    scheduled={k for g in groups for k in g['ids']}
    check(set(cache)<=scheduled,'foreign reused cases')
    with torch.no_grad():
        for group in groups:
            for key in group['ids']:
                case=cache.get(key)
                if case is None:
                    case=evaluate_case(backbone,codec,reference,release,release.record(key),encoder)
                else:
                    from .component_producer import component_view
                    row=release.record(key);view,p,_=component_view(release,row,intervention=True)
                    _,body=view.materialize(p)
                    check(case['record_sha256']==digest(canonical(row)) and case['view_sha256']==digest(body),
                          'reused record/view drift')
                    check(all(a['checkpoint']==live and a['reference']==before and a['profile']==digest(PROFILE.encode())
                              for a in case['construction_audit']),'reused model/profile drift')
                    check(case['score']==grade(row,case),'reused scoring changed')
                cases.append(case)
                if emit is not None:emit(case)
    after=reference.fingerprint();live_after=receiver_hash(backbone,codec)
    result=summarize(release,cases,unchanged=before==after and live==live_after)
    return dict(result,protocol=PROTOCOL,profile=PROFILE,development_release_sha256=release.identity,
        reference_sha256=before,reference_after_sha256=after,receiver_sha256=live,receiver_after_sha256=live_after,
        source_hashes=source_hashes(),gpu_uuid=GPU,optimizer_updates=0,cases=cases,
        scope='forced isolated latent receiver only; no learned invocation, Reason meaning or native answer qualification',
        exact_reconstruction_required=False,training_authorized=False)


def load_behavior_evidence(path,release,*,expected_hash):
    from .reference_qualification import ReferenceEvidence,_TOKEN
    raw=Path(path).read_bytes();check(digest(raw)==sha256(expected_hash),'behavior report hash')
    report=decode(raw)
    check(report.get('protocol')==PROTOCOL,'semantic/behavioral evidence required; reconstruction is diagnostic only')
    check(report['profile']==PROFILE and report['source_hashes']==source_hashes(),'behavior source/profile drift')
    check(report['development_release_sha256']==release.identity,'behavior release drift')
    check(report['gpu_uuid']==GPU and report['optimizer_updates']==0,'behavior GPU/updates')
    check(report['base_frozen'] is True,'base W changed')
    check(report['reference_sha256']==report['reference_after_sha256'] and
          report['receiver_sha256']==report['receiver_after_sha256'],'behavior weight drift')
    from .component_producer import component_view
    from .manifest import BASE_SHA
    check(report['base_sha256']==BASE_SHA and report['gradients_none'] is True,'base/gradient receipt')
    check(report['exact_reconstruction_required'] is False and report['training_authorized'] is False,'behavior scope drift')
    for case in report['cases']:
        row=release.record(case['id'])
        check(case['record_sha256']==digest(canonical(row)),'behavior annotation drift')
        view,p,_=component_view(release,row,intervention=True)
        excerpts,model_bytes=view.materialize(p)
        audit=case.get('construction_audit',[])
        check(case.get('view_sha256')==digest(model_bytes),'behavior permitted-view drift')
        check(len(audit)==len(excerpts)+len(view.references),'behavior audit coverage')
        for index,excerpt in enumerate(excerpts):
            measured=audit[index];binding=view.evidence[index];parent=p.sources[binding.source]
            check(measured['input_hash']==digest(excerpt) and measured['parent']=={
                'source':binding.source,'version':binding.version,'sha256':parent.sha256,
                'byte_start':binding.start,'byte_end':binding.end},'behavior source binding drift')
        for ref,measured in zip(view.references,audit[len(excerpts):]):
            check(measured['role']==ref.role and measured['input_hash']==ref.sha256,'behavior role binding drift')
        for measured in audit:
            check(measured['checkpoint']==report['receiver_sha256'] and measured['reference']==report['reference_sha256']
                  and measured['profile']==digest(PROFILE.encode()) and measured['fresh_state'] is True,
                  'behavior construction identity drift')
            sha256(measured['tensor_hash'])
        check(case['score']==grade(release.record(case['id']),case),'behavior grade mismatch')
        if case.get('actual') is not None:
            runtime=case['runtime']
            check(runtime['prefills']==1 and runtime['lineage']==1 and runtime['consumed']==1 and
                  runtime['total']==sum(runtime[k] for k in ('P','B','A','H')),'behavior accounting receipt')
    result=summarize(release,report['cases'],unchanged=True)
    check(all(report[k]==v for k,v in result.items()),'behavior summary mismatch')
    check(result['status']=='PASSED','semantic/behavioral checks did not pass')
    return ReferenceEvidence(digest(raw),sha256(report['reference_sha256']),release.identity,
                             sha256(report['receiver_sha256']),_token=_TOKEN)
