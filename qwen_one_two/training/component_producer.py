"""Offline reference materialization and isolated receiver/reader episodes.

English glosses are encoded only by the independent, frozen offline reference.
Live Qev invocation uses Session.evaluate(Packet), never annotation text.
"""
import uuid
from .inputs import (Binding,Source,Node,Provenance,Field,reference_input,canonical,digest,check,Exposure)
from .clean import CleanView,CleanEncoder
from .producer import PrerequisiteError,EpisodeLengthError


def component_view(release,row,*,intervention=False):
    """Build per-role inputs with byte-bound facts; Reason stays separate."""
    check(canonical(row)==canonical(release.record(row['id'])),'record/release drift')
    check(release.purpose!='locked','locked collection is not an episode-development source')
    check(row['id'] in release.selected_ids,'unselected component record')
    category=row['training_category']
    if intervention:
        check(row.get('evaluation_slice') in {'E','X'} and category is None,'reviewed receiver evaluation view')
        category='intervention'
    check(category in {'receiver','reader','decision_replay','intervention'},'isolated component category')
    reading=category=='reader'
    inputs=row['model_inputs']
    if reading:
        check(set(inputs)=={'bound_return','permitted_prefix'},'reader input whitelist')
        semantics=inputs['bound_return']['request_semantics']
    elif category=='receiver':
        check(set(inputs)=={'request','evidence'},'receiver input whitelist')
        semantics=inputs['request']
    elif category=='intervention':
        check(set(inputs)<= {'prompt','evidence','request','supplied_return'},'intervention input whitelist')
        semantics=row['semantics']
        check(inputs.get('request',semantics)==semantics,'intervention semantics drift')
    else:
        check(set(inputs)<= {'prompt','evidence'},'decision replay input whitelist')
        semantics=row['supervision']['request_semantics']
    nodes={};sources={};bindings=[];refs=[];indices={}
    def role(name,fields,kind='request_semantics'):
        if isinstance(fields,str):fields={'meaning':fields}
        fields={k:(v if isinstance(v,str) else canonical(v).decode()) for k,v in fields.items()}
        bound={}
        for key,text in fields.items():
            node=f'role-{len(refs)}-{key}'
            nodes[node]=Node(kind,digest(text.encode()))
            bound[key]=Field(text,node)
        p=Provenance(nodes,sources)
        ref=reference_input(name,bound,p,operation=row['operation'])
        index=len(refs);refs.append(ref)
        return index
    indices['question']=role('question',semantics['question'])
    indices['target']=role('target',semantics['target'])
    alternatives=semantics.get('alternatives',[])
    if reading and row['operation']=='Choice':
        # Bank membership is binding metadata, not nonselected evidence. Expose
        # only identity for unselected entries and only the declared selected
        # contents from the reviewed bound return. Never import their properties.
        selected={x['identity']:x for x in inputs['bound_return']['selected_contents']}
        alternatives=[selected.get(x['identity'],{'identity':x['identity']})
                      for x in row['semantics']['alternatives']]
    indices['candidates']=tuple(role('alternative',x) for x in alternatives)
    indices['identities']=tuple(x['identity'] for x in alternatives)
    rubric=semantics.get('rubric')
    indices['rubric']=()
    if rubric:
        indices['rubric']=tuple(role(name,rubric[name]) for name in ('dimension','conditions','zero','one','midpoint') if name in rubric)
    indices['direction']=2 if rubric and rubric.get('direction')=='lower' else 1
    indices['reason']=None
    if reading and inputs['bound_return'].get('reason') is not None:
        indices['reason']=role('reason',inputs['bound_return']['reason'],'reason')
    if not reading:
        evidence=inputs.get('evidence',{})
        check(set(evidence)<= {'facts','rules'},'component evidence scope')
        for descriptor in row['sources']:
            body=release.asset(descriptor['path'])
            source=Source(descriptor['id'],descriptor['version'],body,descriptor['sha256'])
            source.validate();sources[source.identity]=source
        for text in evidence.get('facts',[])+evidence.get('rules',[]):
            body=text.encode()
            found=next(((key,source.content.find(body)) for key,source in sources.items() if body in source.content),None)
            check(found is not None,'component fact has no source bytes')
            key,start=found
            binding=Binding(key,sources[key].version,start,start+len(body))
            node='fact-'+str(len(bindings));bindings.append(binding)
            nodes[node]=Node('source_fact',digest(body),binding=binding)
    p=Provenance(nodes,sources)
    view=CleanView(tuple(refs),tuple(bindings),frozenset(nodes),purpose='reader' if reading else 'receiver',operation=row['operation'])
    view.materialize(p)
    return view,p,indices


def reference_record(release,row):
    """Select one existing reference annotation; no live model/request context."""
    check(canonical(row)==canonical(release.record(row['id'])),'record/release drift')
    check(release.purpose!='locked','locked reference material')
    check(row['id'] in release.selected_ids,'unselected reference record')
    check(row['training_category']=='reference' or
          (row['split']=='development' and row.get('evaluation_slice')=='RF'),'reference category')
    check(set(row['model_inputs'])=={'role','meaning'},'reference input whitelist')
    role=row['model_inputs']['role'];meaning=row['model_inputs']['meaning']
    check(role==row['reference_role'],'reference role drift')
    if isinstance(meaning,str):meaning={'meaning':meaning}
    fields={key:(text if isinstance(text,str) else canonical(text).decode()) for key,text in meaning.items()}
    kind='reason' if role=='reason' else 'request_semantics'
    p=Provenance({key:Node(kind,digest(text.encode())) for key,text in fields.items()},{})
    return reference_input(role,{key:Field(text,key) for key,text in fields.items()},p,operation=row['operation']),p


def reference_record_objective(reference,release,row):
    """Future reference-stage callback; returns graph only, never fits/steps."""
    from .reference import reference_objective
    annotation,p=reference_record(release,row)
    return reference_objective(reference,annotation,p,operation=row['operation'])


class ComponentProducer:
    def __init__(self,backbone,codec,release,reference,*,checkpoint_hash,profile_hash,
                 reference_evidence=None,diagnostic_reference=False):
        if reference is None:
            raise PrerequisiteError('isolated components require a frozen reference artifact; no reference fitting is authorized')
        self.backbone,self.codec,self.release=backbone,codec,release
        from .reference_qualification import ReferenceEvidence
        check(reference_evidence is None or isinstance(reference_evidence,ReferenceEvidence),
              'load measured reference evidence first')
        check(not (reference_evidence is not None and diagnostic_reference),'qualified/diagnostic reference modes are distinct')
        if isinstance(reference_evidence,ReferenceEvidence):
            reference_evidence.assert_reference(reference)
            reference_evidence.assert_receiver(backbone,codec)
        self.evidence,self.diagnostic=reference_evidence,diagnostic_reference
        self.encoder=CleanEncoder(backbone,reference,checkpoint_hash=checkpoint_hash,profile_hash=profile_hash)
        self.last_report=None

    def __call__(self,row,phase,window):
        import torch
        from .clean_packets import CleanPackets
        from .episodes import DecisionLabels,decision_losses,STATUS_NAMES,PORT_NAMES
        from ..codec.runtime import Runtime
        from ..codec.session import Session
        from ..codec.schema import Port,Status,Kind
        from ..codec.writer import decode_result
        check(phase!='reference','native component phase')
        if self.evidence is not None:self.evidence.assert_reference(self.encoder.reference)
        view,p,indices=component_view(self.release,row)
        reading=view.purpose=='reader'
        run=str(uuid.uuid4())
        encoded=self.encoder.encode(view,p,run=run)
        factory=CleanPackets(encoded,view,p)
        # The isolated reader withholds original question/evidence history. Its
        # prefix is explicitly reviewed; the request/return are latent packets.
        prompt=row['model_inputs']['permitted_prefix'] if reading else row['human_input']['prompt']
        rendered=self.backbone.tokenizer.apply_chat_template([{'role':'user','content':prompt}],
            tokenize=False,add_generation_prompt=True,enable_thinking=True)
        ids=self.backbone.tokenizer.encode(rendered,add_special_tokens=False)
        override=512 if phase in {'components','guided_joint'} else None
        limit=512 if phase=='components' else 1024
        check(len(ids)+6<limit,'component prompt exceeds ceiling')
        s=Session.open(self.backbone,self.codec,ids,128,override=override,run=run)
        rt=s.runtime;report={'producer':'clean-component-episodes-v1','record':row['id'],
            'reference':self.encoder.reference.artifact_id,'receiver_behavior_accepted':self.evidence is not None,
            'diagnostic_reference':self.diagnostic,
            'construction_audit':encoded['audit'],'gold_return':reading,'text_tokens_for_invocation':0}
        self.last_report=report;s.episode_report=report
        # A total-position guard runs before every ledger mutation/forward.
        original=rt.debit
        def bounded(kind,count,label):
            if rt.position+count>limit:raise EpisodeLengthError('complete component exceeds ceiling')
            return original(kind,count,label)
        rt.debit=bounded
        losses={}
        try:
            evidence=[]
            for index,ref in enumerate(factory.sources):
                n=len(factory.registry.get(ref).token_ids)
                evidence.extend(factory.span(index,start,min(32,n-start)) for start in range(0,n,32))
            check(len(evidence)<=8,'component evidence exceeds eight spans; never crop')
            port=next(k for k,v in PORT_NAMES.items() if v==row['operation'])
            packet=factory.request(port,indices['question'],indices['target'],evidence=tuple(evidence),
                candidates=indices['candidates'],rubric=indices['rubric'],direction=indices['direction'],session=s)
            inputs,evaluated,_=s.evaluate(packet)
            gold=row['model_inputs']['bound_return'] if reading else row['supervision']['result']
            status=next(k for k,v in STATUS_NAMES.items() if v==gold['status'])
            payload=gold['payload'] or {}
            snapshot=rt.registry.get(packet.snapshot,Kind.SNAPSHOT)
            candidates=() if snapshot.bank is None else rt.registry.get(snapshot.bank,Kind.BANK).candidates
            acceptable=payload.get('acceptable_selection_ids',[])
            choice=tuple(candidates[indices['identities'].index(key)] for key in acceptable)
            if reading:
                returned=factory.supplied_return(status,
                    scalar=torch.tensor(payload['p_true'] if port==Port.NOUL else sum(payload['score_interval'])/2,device='cuda')
                        if status==Status.ASSESSED and port!=Port.CHOICE else None,
                    selected_index=indices['identities'].index(acceptable[0]) if choice else None,reason=indices['reason'])
            else:
                returned=decode_result(self.codec,rt,evaluated)
                actual=rt.registry.get(rt.call.result,Kind.RESULT).status
                feasible=gold.get('feasibility',{})
                # The annotation's generic "unknown" does not distinguish
                # Insufficient from Conflicting. Do not invent the fourth-class
                # label from an overall request status or a prose keyword.
                labels=DecisionLabels(packet.snapshot,status,noul=payload.get('p_true'),choice=choice,
                    score_interval=tuple(payload['score_interval']) if 'score_interval' in payload else None,
                    feasibility=tuple((candidates[indices['identities'].index(k)],0 if v is True else 1)
                                      for k,v in feasible.items() if type(v) is bool))
                losses,mask=decision_losses(rt.registry,packet.snapshot,actual,evaluated,labels)
                report['eligibility']=mask
            check(s.deliver(returned),'component return not consumed')
            s.close()
            if reading and len(choice)<=1:
                targets=self.backbone.tokenizer.encode(row['supervision']['continuations'][0],add_special_tokens=False)
                check(len(targets)<=rt.answer,'reader target exceeds answer allowance')
                values,_=s.native(targets,True)
                if values:losses['native']=torch.stack(values).mean()
            elif reading:
                report['native_continuation_mask']='set-valued reviewed selection is not a single bound return'
            # Receiver-only episodes terminate without pretending to supply an
            # annotated native answer. Isolated-reader native targets are the
            # one explicitly permitted supplied-return supervision branch.
            rt.debit('H',1,'component_chat_end')
            s.step(self.backbone.embedding([248046])[0],'R','component_chat_end',False)
            rt.terminate('Completed',emit_end=False)
        except (ValueError,EpisodeLengthError) as exc:
            rt.terminate('Failed',usable=False,emit_end=False);losses={};report['failure']=str(exc)
        exposure=Exposure('isolated_reader' if reading else 'isolated_receiver',packet_origin='reviewed_component',
            native_teacher_forcing=reading,decision_forcing=True,semantic_substitution=True,gold_return=reading,budget_override=override)
        exposure.validate()
        report.update(summary=s.summary(),eligible_components=sorted(losses))
        return s,losses,exposure
