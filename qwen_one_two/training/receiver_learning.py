"""Isolated supervised receiver objectives; no native training or gold returns."""
import uuid
from .inputs import check,digest,canonical,Exposure
from .release import ReviewedRelease
from .behavior_qualification import PROMPT,PROFILE,receiver_hash
from .receiver_profiles import receiver_parameter, profile, LEGACY

PROTOCOL='receiver-learning-v1'


def training_order(release):
    check(isinstance(release,ReviewedRelease) and release.purpose=='training','receiver training release')
    ids=[e['id'] for e in release.manifest['records'] if e['selected']
         and release.record(e['id'])['training_category']=='receiver']
    check(len(ids)==1500,'pinned complete receiver catalogue')
    ids.sort(key=lambda k:digest((PROTOCOL+':17:'+k).encode()))
    # Eight-example windows: wrap four actual records; never invent padding data.
    return tuple(ids+ids[:(-len(ids))%8])


def groups(backbone,codec,reference):
    from ..codec.ownership import parameter_groups
    original=parameter_groups(backbone,codec)
    selected=[]
    for group in original:
        entries=[(n,p) for n,p in group['named_parameters'] if receiver_parameter(group['name'],n)]
        if entries:selected.append(dict(group,named_parameters=entries))
    backbone.requires_grad_(False);codec.requires_grad_(False)
    for group in selected:
        for _,p in group['named_parameters']:p.requires_grad_(True)
    reference.assert_isolated(codec)
    check(reference.frozen and all(not p.requires_grad for p in reference.parameters()),'frozen reference owner')
    check(all(not p.requires_grad for p in backbone.adapter_parameters('R')),'R must stay frozen')
    check(not codec.budget_row.requires_grad,'budget row must stay frozen')
    return selected


class ReceiverProducer:
    """A complete request-to-heads component, ending before feedback decoding.

    A malformed predicted feedback packet cannot suppress an otherwise valid
    classifier objective. No truncated failed forward is accepted; any real
    construction/forward/resource failure still has zero eligible losses.
    """
    def __init__(self,backbone,codec,reference,release,*,input_profile=LEGACY,cache_sources=False):
        from .clean import CleanEncoder
        check(release.purpose=='training','never train on development/locked records')
        spec=profile(input_profile,'component')
        check(spec['roles']=='frozen-reference-pooled-v1', 'direct role implementation belongs to RD-04')
        self.b,self.c,self.ref,self.release=backbone,codec,reference,release
        self.input_profile=input_profile
        self.encoder=CleanEncoder(backbone,reference,checkpoint_hash=receiver_hash(backbone,codec),
            profile_hash=digest(PROFILE.encode()),cache_references=True,
            source_policy=spec['source_policy'],cache_sources=cache_sources)

    def __call__(self,row):
        import torch
        from .component_producer import component_view
        from .clean_packets import CleanPackets
        from .binding import annotation
        from .episodes import PORT_NAMES,STATUS_NAMES,DecisionLabels,decision_losses
        from .losses import ce
        from ..codec.schema import Kind
        from ..codec.session import Session
        from ..codec.render import Packet
        view,p,index=component_view(self.release,row)
        check(row['training_category']=='receiver','receiver records only')
        # Identity follows the current D/codec; source states are never cached.
        self.encoder.identity['checkpoint']=receiver_hash(self.b,self.c)
        run=str(uuid.uuid4());encoded=self.encoder.encode(view,p,run=run);factory=CleanPackets(encoded,view,p)
        rendered=self.b.tokenizer.apply_chat_template([{'role':'user','content':PROMPT}],
            tokenize=False,add_generation_prompt=True,enable_thinking=True)
        s=Session.open(self.b,self.c,self.b.tokenizer.encode(rendered,add_special_tokens=False),128,override=512,run=run)
        rt=s.runtime;report={'producer':PROTOCOL,'record':row['id'],'record_sha256':digest(canonical(row)),
            'objective':'complete isolated request-to-heads; no feedback/native objective',
            'reference':self.ref.artifact_id,'receiver_behavior_accepted':False,'diagnostic_reference':False,
            'text_tokens_for_invocation':0,'gold_return':False,'input_sha256':encoded['input_hash'],
            'source_audit':encoded['audit'][:len(factory.sources)]}
        s.episode_report=report
        if self.input_profile!=LEGACY:
            report.update(input_profile=self.input_profile,source_policy=self.encoder.source_policy,
                          frozen_source=self.encoder.identity['frozen_source'])
        debit=rt.debit
        def bounded(kind,count,label):
            check(rt.position+count<=512,'complete receiver objective exceeds 512 positions')
            return debit(kind,count,label)
        rt.debit=bounded;losses={}
        try:
            spans=[]
            for i,ref in enumerate(factory.sources):
                n=len(factory.registry.get(ref).token_ids)
                spans.extend(factory.span(i,j,min(32,n-j)) for j in range(0,n,32))
            check(len(spans)<=8,'receiver evidence exceeds eight spans; never crop')
            port=next(k for k,v in PORT_NAMES.items() if v==row['operation'])
            packet=factory.request(port,index['question'],index['target'],evidence=tuple(spans),
                candidates=index['candidates'],rubric=index['rubric'],direction=index['direction'],session=s)
            _,evaluated,_=s.evaluate(packet)
            check(isinstance(packet,Packet),'latent receiver request')
            # Observe actual heads before reading outcome labels. This is report
            # telemetry only; no predicted value is substituted into the packet.
            head={'status':next(v for k,v in STATUS_NAMES.items() if int(k)==int(evaluated['status_logits'].argmax())+1)}
            if row['operation']=='Choice':
                head['selected']=index['identities'][int(evaluated['pointer_logits'].argmax())]
            else:
                head['value']=float(evaluated['value'].detach())
                head['raw']=float(evaluated['raw'].detach())
            report['head']=head
            # Outcome labels are accessed only after the actual request forward.
            gold=row['supervision']['result'];payload=gold['payload'] or {}
            status=next(k for k,v in STATUS_NAMES.items() if v==gold['status'])
            actual=next(k for k in STATUS_NAMES if int(k)==int(evaluated['status_logits'].argmax())+1)
            snapshot=rt.registry.get(packet.snapshot,Kind.SNAPSHOT)
            candidates=() if snapshot.bank is None else rt.registry.get(snapshot.bank,Kind.BANK).candidates
            mapping=dict(zip(index['identities'],candidates))
            reason=None
            if row['supervision'].get('reason') is not None:
                target=annotation('reason',row['supervision']['reason'],row['operation'])
                reason=self.encoder.reference_payload(target)
                report['reason_target_sha256']=target.sha256
            labels=DecisionLabels(packet.snapshot,status,noul=payload.get('p_true'),
                choice=tuple(mapping[k] for k in payload.get('acceptable_selection_ids',[])),
                score_interval=tuple(payload['score_interval']) if 'score_interval' in payload else None,
                feasibility=tuple((mapping[k],0 if v is True else 1) for k,v in gold.get('feasibility',{}).items()
                                  if type(v) is bool),reason_slots=reason)
            losses,eligibility=decision_losses(rt.registry,packet.snapshot,actual,evaluated,labels)
            if eligibility['masks']['reason']:
                # Canonical approved Reason target is slots only, with no
                # attachments, exactly as the existing clean reader constructs.
                losses['structure']=ce(self.c.heads['child_count'](evaluated['feature']),0)
            report.update(eligibility=eligibility,predicted_status=STATUS_NAMES[actual],
                          reference_status=gold['status'],losses={k:float(v.detach()) for k,v in losses.items()})
            # Ending this declared component releases unused return reservations;
            # no result is supplied or claimed consumed, and no native loss exists.
            rt.terminate('Completed',emit_end=False)
        except ValueError as exc:
            losses={};rt.terminate('Failed',usable=False,emit_end=False);report['failure']=str(exc)
        exposure=Exposure('isolated_receiver',packet_origin='reviewed_component',decision_forcing=True,
                          semantic_substitution=True,budget_override=512)
        del rt.debit  # discard the closure over this completed runtime
        report['summary']=s.summary()
        return s,losses,exposure
