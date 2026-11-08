"""Offline formation targets and conservative, content-bound supervision.

No semantic keyword router, similarity threshold or decoded-gloss approval.
Exact reference content can bind; arbitrary learned deviations stay unresolved.
"""
from dataclasses import dataclass
from .inputs import (canonical,check,digest,Node,Provenance,Field,reference_input,SLOTS)
from ..codec.layout import LEGACY,DIRECT,semantic_lengths


class UnsupportedNativeTargets(ValueError):
    """Explicit metadata exclusion, not a swallowed GPU/model failure."""


def annotation(role,fields,operation):
    if isinstance(fields,str):fields={'meaning':fields}
    fields={key:value if isinstance(value,str) else canonical(value).decode() for key,value in fields.items()}
    kind='reason' if role=='reason' else 'request_semantics'
    p=Provenance({key:Node(kind,digest(value.encode())) for key,value in fields.items()},{})
    return reference_input(role,{key:Field(value,key) for key,value in fields.items()},p,operation=operation)


def encode_annotation(reference,item):
    import torch
    rendered=reference.backbone.tokenizer.apply_chat_template([{'role':'user','content':item.content.decode()}],
        tokenize=False,add_generation_prompt=True,enable_thinking=True)
    ids=reference.backbone.tokenizer.encode(rendered,add_special_tokens=False)
    with torch.no_grad():return reference.pooler(reference.annotation_states(ids),*SLOTS[item.role])


def wire_payload(reference,payload,role):
    import torch
    from ..codec.schema import Role
    roles={'question':Role.QUESTION,'target':Role.TARGET,'alternative':Role.CANDIDATE,
           'dimension':Role.DIMENSION,'conditions':Role.CONDITIONS,'zero':Role.ZERO,'one':Role.ONE,
           'midpoint':Role.MIDPOINT,'key':Role.KEY,'reason':Role.REASON}
    return torch.stack([reference.codec.wire(v,roles[role],i,0,0 if role=='key' else None) for i,v in enumerate(payload)])


@dataclass(frozen=True)
class Targets:
    release:str
    record:str
    record_hash:str
    operation:str
    references:tuple
    payloads:tuple
    names:tuple
    identities:tuple
    direction:int
    midpoint:bool
    labels:bytes
    reason:object
    source:bytes
    reference:str
    payload_hashes:tuple
    source_ref:object
    source_hash:str | None
    reason_hash:str | None
    profile:str = LEGACY
    field_lengths:tuple = ()


def request_targets(reference,release,row,source,*,call_index=0,source_ref=None,source_record=None,
                    profile=LEGACY,source_encoder=None,source_tokens=None):
    """Reviewed role meanings only enter the independent offline encoder."""
    check(canonical(row)==canonical(release.record(row['id'])),'target record/release drift')
    check(release.purpose!='locked','locked supervision prohibited')
    check(row['id'] in release.selected_ids,'unselected supervision')
    check(row['training_category'] in {'writer','interleaved'},'native request targets')
    if row['training_category']=='interleaved':
        call=row['trajectory']['calls'][call_index]
        meanings,labels,operation=call['request_target'],call['result_target'],call['operation']
    else:
        check(call_index==0,'single writer target')
        meanings,labels,operation=row['supervision']['request_semantics'],row['supervision']['result'],row['operation']
    check(labels['operation']==operation,'result/request operation mismatch')
    items=[('question',meanings['question']),('target',meanings['target'])]
    candidates=meanings.get('alternatives',[])
    items += [('alternative',value) for value in candidates]
    rubric=meanings.get('rubric',{})
    items += [(key,rubric[key]) for key in ('dimension','conditions','zero','one','midpoint') if key in rubric]
    refs=tuple(annotation(role,value,operation) for role,value in items)
    check(profile in (LEGACY,DIRECT),'native target profile')
    plans=None
    if profile==DIRECT:
        from .native import target_layout
        plans=target_layout(refs,operation,source_tokens,reference.backbone.tokenizer)
        if callable(source_encoder):source_encoder=source_encoder()
        check(source_encoder is not None and source_encoder.source_policy=='frozen-w-source-v1'
              and source_encoder.role_policy==DIRECT,'direct training targets require declared frozen W encoder')
        payloads=tuple(source_encoder._body_states(ref.content,p['ids'],p['selected'],p['offsets'])[0]
                       for ref,p in zip(refs,plans))
    else:payloads=tuple(encode_annotation(reference,ref) for ref in refs)
    reason=row['supervision'].get('reason') if row['training_category']=='writer' else row['supervision'].get('reason_sequence',[None]*len(row['trajectory']['calls']))[call_index]
    # Some collections store only a single call's Reason; never reuse it for
    # a later call lacking an explicitly bound annotation.
    if row['training_category']=='interleaved' and call_index==0 and reason is None:
        reason=row['supervision'].get('reason')
    reason_payload=None if reason is None or profile==DIRECT else encode_annotation(reference,annotation('reason',reason,operation))
    check(reference.frozen and reference.artifact_id==reference.fingerprint(),'frozen unchanged target reference')
    return Targets(release.identity,row['id'],digest(canonical(row)),operation,refs,payloads,
                   tuple(role for role,_ in items),tuple(c['identity'] for c in candidates),
                   2 if rubric.get('direction')=='lower' else 1,'midpoint' in rubric,canonical(labels),
                   reason_payload,source,reference.artifact_id,tuple(tensor_hash(x) for x in payloads),
                   source_ref,None if source_record is None else source_hash(source_record),
                   None if reason_payload is None else tensor_hash(reason_payload),profile,
                   tuple(len(p['selected']) for p in plans) if plans is not None else ())


def tensor_hash(value):
    check(value.is_cuda,'CUDA semantic targets required')
    return digest(value.detach().contiguous().cpu().numpy().tobytes())


def source_hash(source):
    return digest(canonical({'text':digest(source.text),'tokens':source.token_ids,'offsets':source.offsets,
                             'states':tensor_hash(source.states)}))


def request_fields(registry,snapshot):
    """Actual immutable snapshot, never a writer's mutable draft/proxy IDs."""
    from ..codec.schema import Kind
    spec=registry.get(snapshot,Kind.SNAPSHOT)
    candidates=() if spec.bank is None else registry.get(spec.bank,Kind.BANK).candidates
    entries=[('question',spec.question),('target',spec.target)]+[('alternative',ref) for ref in candidates]
    fields=[];extras=False
    for role,ref in entries:
        value=registry.get(ref)
        extras |= bool(value.attributes)
        fields.append((role,value.slots))
    if spec.rubric is not None:
        rubric=registry.get(spec.rubric,Kind.RUBRIC)
        extras |= bool(rubric.attributes)
        sizes=semantic_lengths(Kind.RUBRIC,spec.profile,rubric.lengths,midpoint=rubric.midpoint)
        start=0
        for role,n in zip(('dimension','conditions','zero','one','midpoint'),sizes):
            if role!='midpoint' or rubric.midpoint:fields.append((role,rubric.slots[start:start+n]))
            start+=n
    return spec,candidates,tuple(fields),extras


def canonical_payload(payload):
    mean=payload.square().mean(-1,keepdim=True)
    return payload/(mean.where(mean>1e-12,mean.new_full((),1e-12))).sqrt()


def candidate_mapping(registry,snapshot,targets):
    """Bind reviewed identities to stable actual Refs, permitting permutation."""
    import torch
    from ..codec.schema import Kind
    _,candidates,_,_=request_fields(registry,snapshot)
    expected=[payload for role,payload in zip(targets.names,targets.payloads) if role=='alternative']
    if len(candidates)!=len(expected) or len(set(targets.identities))!=len(expected):return None
    mapping={}
    for identity,payload in zip(targets.identities,expected):
        matches=[ref for ref in candidates if torch.equal(registry.get(ref,Kind.ALTERNATIVE).slots,canonical_payload(payload))]
        if len(matches)!=1 or matches[0] in mapping.values():return None
        mapping[identity]=matches[0]
    return mapping


def compare_request(registry,snapshot,targets):
    """Exact canonical payloads plus complete grounded source coverage.

    A matching identity, same argmax, small loss or similar vectors never bind.
    Extra attributes, partial evidence and changed meanings reject. Candidate
    order may change when an exact, unambiguous identity-to-Ref match exists.
    """
    if targets.profile==DIRECT:
        return False,'native direct writer semantics unverified; alignment is not outcome admission'
    import torch
    from ..codec.schema import Kind
    from ..codec.records import Span
    from .episodes import PORT_NAMES
    spec,candidates,fields,extras=request_fields(registry,snapshot)
    if spec.profile!=targets.profile:return False,'native layout mismatch'
    check(tuple(tensor_hash(x) for x in targets.payloads)==targets.payload_hashes,'semantic target mutation')
    if PORT_NAMES[spec.port]!=targets.operation:return False,'operation mismatch'
    if extras:return False,'unreviewed attributes'
    if tuple(role for role,_ in fields)!=targets.names:return False,'field layout mismatch'
    for (role,actual),expected in zip(fields,targets.payloads):
        # Registry publication canonicalizes each semantic payload once. Apply
        # precisely that operation to reference targets, including tiny floors.
        if role=='alternative':continue
        expected=canonical_payload(expected)
        if not torch.equal(actual,expected):return False,'unresolved '+role+' content'
    if candidate_mapping(registry,snapshot,targets) is None:return False,'unresolved/ambiguous candidate content'
    if spec.rubric is not None:
        rubric=registry.get(spec.rubric,Kind.RUBRIC)
        if rubric.direction!=targets.direction or rubric.midpoint!=targets.midpoint:return False,'rubric metadata mismatch'
    spans=[]
    for item in spec.evidence:
        if not isinstance(item,Span):return False,'unreviewed history evidence'
        if targets.source_ref is None or item.source!=targets.source_ref:return False,'source lineage mismatch'
        _,body,clip_start,clip_end=registry.span(item)
        source=registry.get(item.source,Kind.SOURCE)
        if source_hash(source)!=targets.source_hash:return False,'source states/coordinates mismatch'
        start=source.offsets[item.start][0]+clip_start
        end=source.offsets[item.start+item.count-1][0]+clip_end
        if source.text!=targets.source:return False,'source content mismatch'
        spans.append((start,end,body))
    spans.sort(key=lambda x:x[0])
    cursor=0
    for start,end,body in spans:
        if start!=cursor or body!=targets.source[start:end]:return False,'evidence gap/overlap'
        cursor=end
    if cursor!=len(targets.source):return False,'incomplete evidence'
    return True,'exact reviewed reference content and evidence'


def formation_losses(reference,registry,snapshot,targets,book):
    """Valid formation targets survive divergent outcomes; no joint substitution."""
    from .losses import alignment
    check(tuple(tensor_hash(x) for x in targets.payloads)==targets.payload_hashes,'semantic target mutation')
    spec,_,fields,_=request_fields(registry,snapshot)
    if spec.profile!=targets.profile:return
    if tuple(role for role,_ in fields)!=targets.names:return
    for index,((role,predicted),item,expected) in enumerate(zip(fields,targets.references,targets.payloads)):
        if targets.profile==DIRECT:
            if predicted.shape==expected.shape:
                book.add('alignment',str(index),alignment(predicted,expected),role=role)
            continue  # no unsupported variable-layout reference reconstruction
        book.add('alignment',str(index),alignment(predicted,expected),role=role)
        tokens=reference.backbone.tokenizer.encode(item.content.decode(),add_special_tokens=False)+[248046]
        book.add('reconstruction',str(index),reference.reconstruct(wire_payload(reference,predicted,role),tokens),role=role)


def bind_labels(reference,evidence,registry,snapshot,targets,*,runtime=None):
    if targets.profile==DIRECT:
        return None,{'supported':False,'reason':'native direct writer semantics unverified; no reference/answer-based waiver',
                     'qualification':None,'receiver_behavior_accepted':False}
    from .reference_qualification import ReferenceEvidence
    from .episodes import DecisionLabels,STATUS_NAMES
    from .release import decode
    check(evidence is None or isinstance(evidence,ReferenceEvidence),'load measured reference evidence first')
    check(reference.frozen and all(not p.requires_grad for p in reference.parameters())
          and reference.artifact_id==reference.fingerprint(),'frozen reference changed')
    if evidence is not None:evidence.assert_reference(reference)
    if runtime is None or runtime.registry is not registry or runtime.state!='Evaluating' or runtime.call.snapshot!=snapshot:
        return None,{'supported':False,'reason':'no current admitted request'}
    check(targets.reference==reference.artifact_id,'target reference mismatch')
    check(targets.reason is None or tensor_hash(targets.reason)==targets.reason_hash,'Reason target mutation')
    supported,reason=compare_request(registry,snapshot,targets)
    if not supported:return None,{'supported':False,'reason':reason}
    _,candidates,_,_=request_fields(registry,snapshot)
    mapping=candidate_mapping(registry,snapshot,targets)
    gold=decode(targets.labels);payload=gold.get('payload') or {}
    status=next(k for k,v in STATUS_NAMES.items() if v==gold['status'])
    chosen=tuple(mapping[key] for key in payload.get('acceptable_selection_ids',[]))
    feasibility=tuple((mapping[key],0 if value is True else 1)
                      for key,value in gold.get('feasibility',{}).items() if type(value) is bool)
    labels=DecisionLabels(snapshot,status,noul=payload.get('p_true'),choice=chosen,
        score_interval=tuple(payload['score_interval']) if 'score_interval' in payload else None,
        feasibility=feasibility,reason_slots=targets.reason)
    return labels,{'supported':True,'reason':reason,'snapshot':snapshot.identity,
                  'record_hash':targets.record_hash,'qualification':None if evidence is None else evidence.identity,
                  'receiver_behavior_accepted':evidence is not None}


def compatible_return(registry,result,labels,targets):
    if labels is None:return False
    import torch
    from ..codec.schema import Kind,Port,Status
    value=registry.get(result,Kind.RESULT)
    if value.snapshot!=labels.snapshot or value.status!=labels.status:return False
    spec=registry.get(value.snapshot,Kind.SNAPSHOT)
    if value.status==Status.ASSESSED:
        if spec.port==Port.CHOICE:return len(labels.choice)==1 and value.selected in labels.choice
        raw=value.scalar.detach()
        if not bool(raw.isfinite().all()):return False
        if spec.port==Port.NOUL:return labels.noul is not None and bool(raw==labels.noul)
        return labels.score_interval is not None and bool((raw>=labels.score_interval[0]) & (raw<=labels.score_interval[1]))
    if targets.reason is None or value.reason is None:return False
    reason=registry.get(value.reason,Kind.REASON)
    expected=canonical_payload(targets.reason)
    return not reason.attachments and torch.equal(reason.slots,expected)
