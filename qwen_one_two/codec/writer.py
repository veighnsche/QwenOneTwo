"""Bounded, learned head decoding. Exact rules validate, never choose semantics."""
from dataclasses import dataclass, replace
import torch
from .schema import Role, Kind, Port, Status, ValueKind, Mode, Invalid, require, decode_atoms, Atom, parse_numeric
from .records import Semantic, Attribute, Value, Span, History, Reason
from .render import Renderer
from .neural import finite
from .layout import LEGACY,DIRECT,COMPACT_DIRECT


@dataclass(frozen=True)
class Draft:
    port: Port
    question: object = None
    target: object = None
    evidence: tuple = ()
    candidates: tuple = ()
    rubric: object = None
    bank: object = None  # unpublished until FINISH; no bank tag in draft view
    profile: str = LEGACY


class Decoder:
    def __init__(self, codec, registry, catalogue=None,*,profile=LEGACY):
        self.codec=codec;self.registry=registry
        self.profile=profile;codec.require_profile(profile)
        self.catalogue=tuple(registry.refs() if catalogue is None else catalogue)

    def candidates(self, predicate):
        return tuple(r for r in self.catalogue if predicate(r,self.registry.get(r)))

    def summary(self,ref):
        x=self.registry.get(ref)
        if ref.kind==Kind.SOURCE:return x.states.mean(0)
        return self.codec.encode(Renderer(self.registry).object(ref)).mean(0)

    def pointer(self,name,f,refs,optional=False):
        require(bool(refs) or optional,'no structurally compatible pointer entries')
        keys=torch.stack([self.summary(r) for r in refs]) if refs else f.new_empty(0,1024)
        i=int(self.codec.pointer(name,f,keys,optional).argmax())
        return None if optional and i==0 else refs[i-int(optional)]

    def span(self,f):
        sources=self.candidates(lambda r,x:r.kind==Kind.SOURCE)
        ref=self.pointer('source',f,sources)
        s=self.registry.get(ref)
        start=int(self.codec.pointer('span_start',f,s.states).argmax())
        length=self.codec.heads['span_length'](f)
        mask=torch.arange(33,device='cuda')
        length=length.masked_fill((mask==0)|(mask>min(32,len(s.token_ids)-start)),-torch.inf)
        count=int(length.argmax());last=start+count-1
        def clip(index,name,minimum=None):
            a,b=s.offsets[index]
            offsets=[]
            for offset in range(b-a+1):
                if minimum is not None and offset<=minimum:continue
                if name=='clip_start' and offset==b-a:continue
                if name=='clip_end' and offset==0:continue
                try:s.text[:a+offset].decode('utf8')
                except UnicodeError:continue
                offsets.append(offset)
            require(bool(offsets),'no legal source boundary')
            k=self.codec.clipping_keys(self.codec.pointer_key(s.states[index]),offsets)
            logits=(k*self.codec.pointers[name](f)).sum(-1)/16
            finite(logits,'clipping pointer logits')
            return offsets[int(logits.argmax())]
        a=clip(start,'clip_start');b=clip(last,'clip_end',a if count==1 else None)
        span=Span(ref,start,count,a,b);self.registry.span(span)
        return span

    def attributes(self,f,owner_role):
        count=int(self.codec.heads['child_count'](f).argmax())
        return tuple(self.attribute(f,i,owner_role) for i in range(count))

    def attribute(self,f,index,owner_role=Role.TARGET):
        c=self.codec;u=c.condition(f,Role.ATTRIBUTE_DESC,attribute=index)
        kind=ValueKind(int(c.heads['attribute_type'](u).argmax())+1)
        present=bool(c.heads['presence'](u).squeeze()>0)
        keys=c.emit(f,Role.KEY,c.semantic_count(f,Role.KEY) if self.profile==DIRECT else 2,index)
        refs=self.catalogue
        unit=self.pointer('unit',u,refs,optional=True)
        if not present:return Attribute(keys,Value(kind,False,unit=unit),self.profile)
        mode_logits=c.heads['source_mode'](u)
        if kind not in (ValueKind.INT64,ValueKind.DECIMAL):
            mode_logits=mode_logits.clone();mode_logits[1]=-torch.inf
        copy=[]
        for r in refs:
            x=self.registry.get(r)
            for i,a in enumerate(getattr(x,'attributes',())):
                if a.profile==self.profile and a.value.kind==kind and a.value.present:copy.append((r,i,a))
        if not copy:mode_logits=mode_logits.clone();mode_logits[0]=-torch.inf
        mode=Mode(int(mode_logits.argmax())+1)
        if mode==Mode.COPY:
            summaries=torch.stack([c.encode_attribute_view(self.registry,r,i).mean(0) for r,i,a in copy])
            j=int(c.pointer('copy',u,summaries).argmax());r,i,a=copy[j]
            value=replace(a.value,mode=Mode.COPY,copied_owner=r,copied_index=i,parsed_span=None)
            return Attribute(keys,value,self.profile)
        if mode==Mode.PARSE:
            span=self.span(u);_,literal,_,_=self.registry.span(span)
            return Attribute(keys,Value(kind,True,parse_numeric(literal,kind),mode,span.source,unit=unit,parsed_span=span),self.profile)
        if kind==ValueKind.SOURCE_STRING:
            span=self.span(u)
            return Attribute(keys,Value(kind,True,span,mode,span.source,unit=unit),self.profile)
        if kind==ValueKind.REFERENCE:
            data=self.pointer('reference',u,refs)
        elif kind==ValueKind.BOOLEAN:data=bool(c.heads['boolean'](u).squeeze()>0)
        elif kind==ValueKind.INT64:
            raw=bytes(c.byte_logits(u,8,index).argmax(-1).tolist())
            data=decode_atoms(kind,(Atom(Role.INT64,raw),))
        elif kind==ValueKind.DECIMAL:
            raw=bytes(c.byte_logits(u,8,index).argmax(-1).tolist())
            coefficient=int.from_bytes(raw,'big',signed=True)
            data=(coefficient,int(c.heads['decimal_exponent'](u).argmax())-18)
        else:
            length=int(c.heads['utf8_length'](u).argmax())
            data=bytes(c.byte_logits(u,length,index).argmax(-1).tolist())
        value=Value(kind,True,data,mode,unit=unit)
        attr=Attribute(keys,value,self.profile);self.registry.attribute(attr,self.catalogue)
        return attr

    def semantic(self,f,kind,*,allow_attributes=True):
        c=self.codec
        require(self.profile!=COMPACT_DIRECT,'native v2 role emission unsupported; component-only profile')
        if self.profile==DIRECT:
            if kind==Kind.RUBRIC:
                logits=c.heads['direction'](f).clone();logits[0]=-torch.inf
                direction=int(logits.argmax());midpoint=bool(c.heads['presence'](f).squeeze()>0)
                roles=(Role.DIMENSION,Role.CONDITIONS,Role.ZERO,Role.ONE)+( (Role.MIDPOINT,) if midpoint else () )
                lengths=tuple(c.semantic_count(f,r) for r in roles)
                return Semantic(torch.cat([c.emit(f,r,n) for r,n in zip(roles,lengths)]),direction=direction,
                    midpoint=midpoint,profile=DIRECT,lengths=lengths if midpoint else lengths+(0,))
            role={Kind.QUESTION:Role.QUESTION,Kind.TARGET:Role.TARGET,Kind.ALTERNATIVE:Role.CANDIDATE}[kind]
            count=c.semantic_count(f,role)
            return Semantic(c.emit(f,role,count),self.attributes(f,role) if allow_attributes and kind!=Kind.QUESTION else (),
                            profile=DIRECT,lengths=(count,))
        if kind==Kind.RUBRIC:
            logits=c.heads['direction'](f).clone();logits[0]=-torch.inf
            direction=int(logits.argmax());midpoint=bool(c.heads['presence'](f).squeeze()>0)
            parts=[c.emit(f,r,2) for r in (Role.DIMENSION,Role.CONDITIONS,Role.ZERO,Role.ONE)]
            parts.append(c.emit(f,Role.MIDPOINT,2) if midpoint else f.new_zeros(2,1024))
            return Semantic(torch.cat(parts),direction=direction,midpoint=midpoint)
        role,count={Kind.QUESTION:(Role.QUESTION,8),Kind.TARGET:(Role.TARGET,2),Kind.ALTERNATIVE:(Role.CANDIDATE,4)}[kind]
        return Semantic(c.emit(f,role,count),self.attributes(f,role) if kind in (Kind.TARGET,Kind.ALTERNATIVE) else ())

    def reason(self,feature):
        c=self.codec;count=int(c.heads['child_count'](feature).argmax());attachments=[]
        for i in range(count):
            f=c.condition(feature,Role.REASON,attribute=i)
            kind=int(c.heads['attachment_kind'](f).argmax())
            if kind==0:attachments.append(self.attribute(f,i,Role.REASON))
            elif kind==1:attachments.append(self.span(f))
            else:
                refs=self.candidates(lambda r,x:r.kind==Kind.RESULT and x.presentation is not None)
                attachments.append(History(self.pointer('history',f,refs)))
        return Reason(c.emit(feature,Role.REASON,4),tuple(attachments),self.profile)


class Writer:
    """One start and seven learned events; caller supplies the one causal state."""
    def __init__(self,runtime,codec):
        self.runtime=runtime;self.codec=codec;self.draft=Draft(runtime.call.port,profile=runtime.profile)

    def feature(self,h):
        view=Renderer(self.runtime.registry).draft(self.draft)
        memory=self.codec.encode(view) if len(view) else h.new_empty(0,1024)
        return self.codec.feature(h,self.runtime.counters(),memory)

    def record(self,f,kind):
        decoder=Decoder(self.codec,self.runtime.registry,profile=self.runtime.profile)
        refs=decoder.candidates(lambda r,x:r.kind==kind and x.profile==self.runtime.profile)
        logits=self.codec.heads['record_mode'](f)
        if not refs:logits=logits.clone();logits[1]=-torch.inf
        if int(logits.argmax())==1:return decoder.pointer('reuse',f,refs)
        return self.runtime.publish(kind,decoder.semantic(f,kind))

    def initialise(self,h):
        require(self.runtime.call.writer==1,'initialization is WRITER_START only')
        f=self.feature(h)
        q=self.record(f,Kind.QUESTION);t=self.record(f,Kind.TARGET)
        self.draft=replace(self.draft,question=q,target=t)

    def control_vector(self):
        rt=self.runtime;c=self.codec
        rt.writer_position()  # current position is charged before counters sampled
        role=Role.WRITER_START if rt.call.writer==1 else Role.WRITER_TICK
        view=Renderer(rt.registry).draft(self.draft)
        length=len(view)+2 if len(view) else 0
        payload=c.meta(torch.tensor(rt.control_bits(role,length=length,candidates=len(self.draft.candidates),
                            evidence=len(self.draft.evidence)),device='cuda',dtype=torch.float32))
        return role,payload

    def tick_vector(self,h):
        role,payload=self.control_vector()
        if role==Role.WRITER_TICK:payload=payload+self.codec.feedback(self.feature(h))
        return self.codec.wire(payload,role,previous=self.runtime.call.previous if role==Role.WRITER_TICK else None)

    def execute(self,h):
        rt=self.runtime;f=self.feature(h);logits=self.codec.writer_action(f).clone()
        if self.draft.port!=Port.CHOICE or len(self.draft.candidates)>=16:logits[1]=-torch.inf
        if self.draft.port!=Port.SCORE:logits[2]=-torch.inf
        if len(self.draft.evidence)>=8:logits[0]=-torch.inf
        action=int(logits.argmax())
        old=self.draft;packet=None
        try:
            d=Decoder(self.codec,rt.registry,profile=rt.profile)
            if action==0:
                require(len(old.evidence)<8,'evidence cap')
                kind=int(self.codec.heads['evidence_kind'](f).argmax())
                refs=d.candidates(lambda r,x:r.kind==Kind.RESULT and x.presentation is not None)
                e=d.span(f) if kind==0 else History(d.pointer('history',f,refs))
                proposal=replace(old,evidence=old.evidence+(e,))
            elif action==1:
                require(old.port==Port.CHOICE and len(old.candidates)<16,'candidate operation/cap')
                r=self.record(f,Kind.ALTERNATIVE);proposal=replace(old,candidates=old.candidates+(r,))
            elif action==2:
                require(old.port==Port.SCORE,'rubric operation')
                proposal=replace(old,rubric=self.record(f,Kind.RUBRIC))
            elif action==3:proposal=replace(old,question=self.record(f,Kind.QUESTION))
            elif action==4:proposal=replace(old,target=self.record(f,Kind.TARGET))
            elif action==5:
                packet=rt.seal(old.question,old.target,old.evidence,old.rubric,old.candidates);proposal=old
            else:proposal=old
            if action<5:
                view=Renderer(rt.registry).draft(proposal)
                require(len(view)+2<=256,'prospective packet bound')
            self.draft=proposal
            if rt.state=='Writer':rt.writer_action(action)
            else:rt.call.previous=action
        except Invalid as exc:
            self.draft=old
            if rt.state=='Writer':
                rt.call.previous=action;rt.call.outcome=2
                if rt.call.writer==32:rt.end_attempt('WriterLimit',5)
            return None,exc
        return packet,None


def decode_result(codec,runtime,evaluated):
    status=Status(int(evaluated['status_logits'].argmax())+1)
    if status==Status.ASSESSED:
        if runtime.call.port==Port.CHOICE:
            return runtime.prepare_result(runtime.call.snapshot,status,selected_index=int(evaluated['pointer_logits'].argmax()))
        return runtime.prepare_result(runtime.call.snapshot,status,scalar=evaluated['value'])
    reason=Decoder(codec,runtime.registry,runtime.call.catalogue,profile=runtime.profile).reason(evaluated['feature'])
    return runtime.prepare_result(runtime.call.snapshot,status,reason=reason)
