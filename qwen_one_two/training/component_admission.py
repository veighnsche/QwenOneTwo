"""Explicit experimental prepared-component binding; never a native oracle.

Only the reviewed clean constructor can establish this provenance. No outcome,
predicted answer, similarity threshold or reference-qualification flag admits it.
Tensor fingerprints are CUDA readbacks, not CPU model computation.
"""
from dataclasses import dataclass
from .inputs import canonical,check,digest
from .receiver_profiles import LEGACY,profile
from .binding import tensor_hash,source_hash,canonical_payload
from ..codec.layout import DIRECT,LEGACY as WIRE_LEGACY,semantic_lengths
from ..codec.schema import Kind,Port
from ..codec.records import Span

PROTOCOL='prepared-receiver-component-v1'
GPU_CHECKS={
    'RD05-GPU-GRADIENT-ACCUMULATION':'test_joint_VJPs_and_eligible_accumulation',
    'RD05-GPU-RESUME':'test_complete_window_resume_and_frozen_owners',
}


def _fields(registry,snapshot):
    spec=registry.get(snapshot,Kind.SNAPSHOT)
    candidates=() if spec.bank is None else registry.get(spec.bank,Kind.BANK).candidates
    fields=[]
    for role,ref in [('question',spec.question),('target',spec.target)]+[('alternative',r) for r in candidates]:
        item=registry.get(ref)
        check(not item.attributes,'experimental component has unreviewed attributes')
        fields.append((role,tensor_hash(item.slots)))
    if spec.rubric is not None:
        item=registry.get(spec.rubric,Kind.RUBRIC)
        check(not item.attributes,'experimental rubric attributes')
        sizes=semantic_lengths(Kind.RUBRIC,spec.profile,item.lengths,midpoint=item.midpoint)
        start=0
        for role,n in zip(('dimension','conditions','zero','one','midpoint'),sizes):
            if role!='midpoint' or item.midpoint:fields.append((role,tensor_hash(item.slots[start:start+n])))
            start+=n
    return spec,candidates,tuple(fields)


def _signature(registry,snapshot):
    spec,candidates,fields=_fields(registry,snapshot)
    rubric=None if spec.rubric is None else registry.get(spec.rubric)
    evidence=[]
    for span in spec.evidence:
        check(isinstance(span,Span),'experimental component requires exact source spans')
        registry.span(span)
        evidence.append((span.source.identity,span.start,span.count,span.clip_start,span.clip_end,
                         source_hash(registry.get(span.source,Kind.SOURCE))))
    return digest(canonical({'snapshot':snapshot.identity,'profile':spec.profile,'port':int(spec.port),
        'question':spec.question.identity,'target':spec.target.identity,
        'bank':None if spec.bank is None else spec.bank.identity,
        'rubric':None if spec.rubric is None else (spec.rubric.identity,rubric.direction,rubric.midpoint,rubric.lengths),
        'candidates':[r.identity for r in candidates],'fields':fields,'evidence':evidence}))


@dataclass(frozen=True)
class PreparedReceiver:
    runtime:object
    snapshot:object
    signature:str
    release:str
    record:str
    record_hash:str
    input_profile:str
    input_hash:str
    candidates:tuple

    def validate(self,runtime):
        check(runtime is self.runtime and runtime.call is not None and runtime.call.snapshot==self.snapshot,
              'foreign/divergent prepared snapshot')
        check(_signature(runtime.registry,self.snapshot)==self.signature,'prepared snapshot contents diverged')

    def receipt(self):
        return {'protocol':PROTOCOL,'experimental':True,'release':self.release,'record':self.record,
            'record_sha256':self.record_hash,'input_profile':self.input_profile,'input_sha256':self.input_hash,
            'snapshot':self.snapshot.identity,'contents_sha256':self.signature,
            'reference_qualification':None,'receiver_behavior_accepted':False,
            'reason_supported':False,'reason_mask':'new-profile Reason semantics unsupported; no target encoding'}

    def labels(self,runtime,release,row,*,reason_slots=None):
        self.validate(runtime)
        check(runtime.state=='Evaluating' and runtime.call.q_remaining==0,'labels require complete actual request forward')
        check(release.purpose=='training' and release.identity==self.release,'never train on development/foreign release')
        check(row['id']==self.record and digest(canonical(row))==self.record_hash
              and canonical(row)==canonical(release.record(self.record)),'prepared label record drift')
        check(reason_slots is None,'unsupported new-profile Reason target')
        from .episodes import DecisionLabels,STATUS_NAMES
        gold=row['supervision']['result'];payload=gold['payload'] or {};mapping=dict(self.candidates)
        check(gold['operation']==row['operation'],'prepared result operation mismatch')
        status=next(k for k,v in STATUS_NAMES.items() if v==gold['status'])
        return DecisionLabels(self.snapshot,status,noul=payload.get('p_true'),
            choice=tuple(mapping[k] for k in payload.get('acceptable_selection_ids',[])),
            score_interval=tuple(payload['score_interval']) if 'score_interval' in payload else None,
            feasibility=tuple((mapping[k],0 if v is True else 1) for k,v in gold.get('feasibility',{}).items() if type(v) is bool))


def admit_prepared_component(release,row,view,provenance,encoded,index,runtime,packet,*,input_profile):
    """Compare the actual sealed snapshot against independently encoded inputs.

    This is intentionally limited to reviewed receiver training components;
    writer/replay/reader paths keep their existing conservative admission masks.
    """
    check(input_profile!=LEGACY,'experimental admission requires an explicit new profile')
    spec=profile(input_profile,'component')
    check(release.purpose=='training','never train on development/locked records')
    from .component_producer import component_view
    original,p,indices=component_view(release,row)
    check(row['training_category']=='receiver' and view==original and index==indices
          and view.materialize(provenance)==original.materialize(p),'unreviewed/divergent permitted component view')
    excerpts,input_bytes=view.materialize(provenance)
    check(encoded['input_hash']==digest(input_bytes),'component input identity drift')
    check(runtime.state=='Evaluating' and runtime.call.snapshot==packet.snapshot and runtime.call.request is packet,
          'component is not the actual admitted request')
    wire=DIRECT if spec['roles']==DIRECT else WIRE_LEGACY
    actual,candidates,fields=_fields(runtime.registry,packet.snapshot)
    check(actual.profile==wire==encoded.get('role_profile',WIRE_LEGACY),'component layout mismatch')
    check({Port.NOUL:'Noul',Port.CHOICE:'Choice',Port.SCORE:'Score'}[actual.port]==row['operation'],'component operation drift')
    check(len(encoded['slots'])==len(view.references),'component role coverage')
    expected=tuple((ref.role,tensor_hash(value if wire==DIRECT else canonical_payload(value)))
                   for ref,value in zip(view.references,encoded['slots']))
    check(fields==expected,'actual role/candidate content differs from reviewed component')
    check(len(index['identities'])==len(candidates) and len(set(index['identities']))==len(candidates),'ambiguous candidate identities')
    rubric=None if actual.rubric is None else runtime.registry.get(actual.rubric)
    check((rubric is not None)==bool(index['rubric']) and
          (rubric is None or (rubric.direction==index['direction'] and rubric.midpoint==(len(index['rubric'])==5))),
          'component rubric metadata drift')
    audits=encoded['audit'];sources=encoded['sources']
    check(len(sources)==len(excerpts) and len(audits)==len(sources)+len(view.references),
          'complete component source/role audit required')
    for ref,value,audit in zip(view.references,encoded['slots'],audits[len(sources):]):
        check(audit['role']==ref.role and audit['input_hash']==ref.sha256 and audit['tensor_hash']==tensor_hash(value),
              'component role input/tensor audit drift')
    spans=[]
    for ref,audit,excerpt in zip(sources,audits,excerpts):
        source=encoded['registry'].get(ref,Kind.SOURCE)
        check(source.text==excerpt and audit['offsets']==source.offsets,'component source byte mapping drift')
        check(source_hash(source)==source_hash(runtime.registry.get(ref,Kind.SOURCE)),'component source states/coordinates drift')
        check(audit['source_policy']==spec['source_policy'] and audit['input_hash']==digest(source.text)
              and audit['tensor_hash']==tensor_hash(source.states),'component frozen source audit drift')
        spans.extend(Span(ref,j,min(32,len(source.token_ids)-j)) for j in range(0,len(source.token_ids),32))
    check(tuple(spans)==actual.evidence,'component evidence gap/overlap/foreign source')
    return PreparedReceiver(runtime,packet.snapshot,_signature(runtime.registry,packet.snapshot),release.identity,
        row['id'],digest(canonical(row)),input_profile,encoded['input_hash'],tuple(zip(index['identities'],candidates)))
