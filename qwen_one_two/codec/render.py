"""One deterministic position builder shared by training and serving."""
from dataclasses import dataclass, replace
from types import SimpleNamespace
from .schema import Role, Kind, Port, Status, ValueKind, Mode, Metadata, Atom, encode_atoms, require, Invalid
from .records import Ref, Span, History, Attribute, Result


@dataclass(frozen=True)
class Position:
    role: Role
    payload: object
    owner: int | None = None
    slot: int | None = None
    attribute: int | None = None
    previous: int | None = None


@dataclass(frozen=True)
class Packet:
    positions: tuple
    handles: tuple
    snapshot: Ref | None
    candidate_ends: tuple = ()
    result: Ref | None = None

    def __len__(self):
        return len(self.positions)


class Handles:
    def __init__(self, registry, prefix=()):
        self.registry=registry
        self.refs=list(prefix)
        require(len(self.refs)<=256 and len(set(self.refs))==len(self.refs),'handle prefix')
        for r in self.refs:require(registry.contains(r),'handle Ref')

    def intern(self, ref):
        if ref is None:return None
        require(self.registry.contains(ref),'unpublished/foreign handle')
        if ref not in self.refs:
            require(len(self.refs)<256,'handle capacity')
            self.refs.append(ref)
        return self.refs.index(ref)


class Renderer:
    def __init__(self, registry):
        self.registry=registry

    def _build(self, snapshot, result=None, prefix=(), historical=False, sealed=True, draft=None, record=None):
        reg=self.registry
        s=draft if draft is not None else (reg.get(snapshot,Kind.SNAPSHOT) if record is None else
            SimpleNamespace(port=Port.NONE,target=None,question=None,bank=None,rubric=None))
        t=Handles(reg,prefix);out=[];candidate_ends=[]
        target=reg.get(s.target) if s.target is not None else None
        question=reg.get(s.question) if s.question is not None else None
        bank=(SimpleNamespace(candidates=draft.candidates) if draft is not None and draft.candidates else
              (reg.get(s.bank) if s.bank is not None else None))
        rubric=reg.get(s.rubric) if s.rubric is not None else None

        def meta(role, owner, kind, following=0, **kw):
            # B1 traversal: owner then the header's Q/T/B/U, or descriptor P/S/U.
            refs={name:kw.pop(name,None) for name in ('question','target','bank','rubric','primary','secondary','unit')}
            owner_h=t.intern(owner)
            order=('question','target','bank','rubric') if role in (Role.REQUEST,Role.RESULT) else ('primary','secondary','unit')
            encoded={name:t.intern(refs[name]) for name in order}
            require(all(refs[name] is None for name in refs if name not in order),'inapplicable reference')
            md=Metadata(s.port,kind=kind,owner=owner_h,following=following,historical=historical,sealed=sealed,**encoded,**kw)
            md.encode(role)
            out.append(Position(role,md,owner_h,attribute=kw.get('_attribute')))

        def slots(role, values, owner, attribute=None):
            h=t.intern(owner)
            for i,v in enumerate(values):out.append(Position(role,v,h,i,attribute))

        def attribute(a, owner, index, catalogue=None):
            reg.attribute(a,catalogue);v=a.value
            n=v.body_length();pa=pb=0
            if v.present and v.kind==ValueKind.SOURCE_STRING:
                _,_,pa,pb=reg.span(v.data)
            if v.present and v.kind==ValueKind.UTF8:pa=len(v.data)
            # Attribute index is a tag, never the descriptor's item index.
            h=t.intern(owner)
            primary=t.intern(v.provenance);secondary=t.intern(v.copied_owner);unit=t.intern(v.unit)
            md=Metadata(s.port,kind=owner.kind,present=v.present,owner=h,following=2+n,
                        value_kind=v.kind,mode=v.mode,primary=primary,secondary=secondary,unit=unit,
                        parameter_a=pa,parameter_b=pb,historical=historical,sealed=sealed)
            md.encode(Role.ATTRIBUTE_DESC)
            out.append(Position(Role.ATTRIBUTE_DESC,md,h,attribute=index))
            slots(Role.KEY,a.keys,owner,index)
            if not v.present:return
            if v.kind==ValueKind.SOURCE_STRING:
                values,_,_,_=reg.span(v.data)
                slots(Role.ATTRIBUTE_SOURCE,values,owner,index)
            else:
                handle=t.intern(v.data) if v.kind==ValueKind.REFERENCE else None
                for i,a0 in enumerate(encode_atoms(v.kind,v.data,handle)):
                    out.append(Position(a0.role,a0,h,i,index))

        def history(packet):
            # Translate exact identities in actual first-encounter order. Do not
            # intern the whole old sidecar or detach its continuous values.
            old=packet.handles
            def translate(h):return None if h is None else t.intern(old[h])
            for p in packet.positions:
                if isinstance(p.payload,Metadata):
                    md=p.payload
                    names=('owner','question','target','bank','rubric') if p.role in (Role.REQUEST,Role.RESULT,Role.BUDGET_HEADER) else ('owner','primary','secondary','unit')
                    changed={n:translate(getattr(md,n)) for n in names}
                    changed['historical']=True
                    md=replace(md,**changed);md.encode(p.role)
                    out.append(replace(p,payload=md,owner=changed['owner']))
                elif p.role==Role.REFERENCE:
                    h=translate(p.owner);refh=translate(p.payload.data[7])
                    out.append(replace(p,owner=h,payload=Atom(Role.REFERENCE,bytes(7)+bytes([refh]))))
                else:out.append(replace(p,owner=translate(p.owner)))

        def evidence(e, index, catalogue=None):
            if isinstance(e,Span):
                if catalogue is not None:require(e.source in catalogue,'source outside catalogue')
                values,_,pa,pb=reg.span(e)
                meta(Role.SOURCE_DESC,e.source,Kind.SOURCE,len(values),present=True,item=index,parameter_a=pa,parameter_b=pb)
                slots(Role.SOURCE,values,e.source)
            else:
                require(isinstance(e,History),'evidence kind')
                reg.evidence(e)
                if catalogue is not None:require(e.result in catalogue,'history outside catalogue')
                previous=reg.get(e.result)
                packet=previous.presentation
                if catalogue is not None:require(set(packet.handles)<=set(catalogue),'historical contents outside catalogue')
                meta(Role.HISTORY_DESC,e.result,Kind.RESULT,len(packet),present=True,item=index,secondary=previous.snapshot)
                history(packet)

        def candidate(ref,index,bank_ref):
            c=reg.get(ref,Kind.ALTERNATIVE)
            cost=sum(a.cost for a in c.attributes)
            require(cost<=46,'candidate return bound')
            meta(Role.CANDIDATE_DESC,ref,Kind.ALTERNATIVE,4+cost,present=True,item=index,primary=bank_ref,children=len(c.attributes))
            slots(Role.CANDIDATE,c.slots,ref)
            for i,a in enumerate(c.attributes):attribute(a,ref,i)
            candidate_ends.append((index,len(out)-1))

        if record is not None:
            obj=reg.get(record)
            if record.kind==Kind.QUESTION:slots(Role.QUESTION,obj.slots,record)
            elif record.kind==Kind.TARGET:
                meta(Role.TARGET_DESC,record,Kind.TARGET,2+sum(a.cost for a in obj.attributes),present=True,children=len(obj.attributes))
                slots(Role.TARGET,obj.slots,record)
                for i,a in enumerate(obj.attributes):attribute(a,record,i)
            elif record.kind==Kind.ALTERNATIVE:candidate(record,0,None)
            elif record.kind==Kind.RUBRIC:
                meta(Role.RUBRIC_DESC,record,Kind.RUBRIC,10,present=True,direction=obj.direction,midpoint=obj.midpoint)
                for j,r in enumerate((Role.DIMENSION,Role.CONDITIONS,Role.ZERO,Role.ONE,Role.MIDPOINT)):
                    slots(r,obj.slots[j*2:j*2+2],record)
            elif record.kind==Kind.RESULT:history(obj.presentation)
            elif record.kind==Kind.REASON:
                start=len(out)
                meta(Role.REASON_DESC,record,Kind.REASON,0,present=True,children=len(obj.attachments))
                slots(Role.REASON,obj.slots,record)
                for i,a in enumerate(obj.attachments):
                    if isinstance(a,Attribute):attribute(a,record,i)
                    else:evidence(a,i)
                out[start]=replace(out[start],payload=replace(out[start].payload,following=len(out)-start-1))
            elif record.kind==Kind.BANK:
                for i,r in enumerate(obj.candidates):candidate(r,i,record)
            else:raise Invalid('object has no compatible bounded presentation')
            require(len(out)<=256,'bounded object view')
            return Packet(tuple(out),tuple(t.refs),None,tuple(candidate_ends))

        owner=snapshot if result is None else result
        rr=None if result is None else reg.get(result,Kind.RESULT)
        require(rr is None or rr.snapshot==snapshot,'result snapshot mismatch')
        assessed=rr is not None and rr.status==Status.ASSESSED
        value_kind=ValueKind.NONE if rr is None or not assessed else (ValueKind.REFERENCE if s.port==Port.CHOICE else ValueKind.CONTINUOUS)
        # Header size is finalized after building the complete position list.
        if draft is None:meta(Role.REQUEST if rr is None else Role.RESULT,owner,Kind.SNAPSHOT if rr is None else Kind.RESULT,
             following=0,present=assessed,status=Status.NONE if rr is None else rr.status,value_kind=value_kind,
             question=s.question,target=s.target,bank=s.bank,rubric=s.rubric,
             evidence=len(s.evidence) if rr is None else 0,candidates=len(bank.candidates) if bank else 0,
             children=len(target.attributes) if rr is None else 0,attempt=s.attempt,complete=1,
             direction=rubric.direction if rubric else 0,midpoint=rubric.midpoint if rubric else False)
        slots(Role.QUESTION,question.slots,s.question)
        at=sum(a.cost for a in target.attributes) if rr is None else 0
        meta(Role.TARGET_DESC,s.target,Kind.TARGET,2+at,present=True,children=len(target.attributes) if rr is None else 0)
        slots(Role.TARGET,target.slots,s.target)
        if rr is None:
            for i,a in enumerate(target.attributes):attribute(a,s.target,i)
            for i,e in enumerate(s.evidence):evidence(e,i)
        if rubric:
            meta(Role.RUBRIC_DESC,s.rubric,Kind.RUBRIC,10,present=True,direction=rubric.direction,midpoint=rubric.midpoint)
            for j,role in enumerate((Role.DIMENSION,Role.CONDITIONS,Role.ZERO,Role.ONE,Role.MIDPOINT)):
                values=rubric.slots[j*2:j*2+2]
                slots(role,values,s.rubric)
        if rr is None:
            if bank:
                for i,r in enumerate(bank.candidates):candidate(r,i,s.bank)
        elif assessed:
            if s.port==Port.CHOICE:candidate(rr.selected,rr.selected_index,s.bank)
            else:out.append(Position(Role.NOUL if s.port==Port.NOUL else Role.SCORE,rr.scalar,t.intern(result),0))
        else:
            z=reg.get(rr.reason,Kind.REASON)
            start=len(out)
            meta(Role.REASON_DESC,rr.reason,Kind.REASON,0,present=True,primary=result,children=len(z.attachments))
            slots(Role.REASON,z.slots,rr.reason)
            catalogue=getattr(self,'catalogue',None)
            for i,a in enumerate(z.attachments):
                if isinstance(a,Attribute):attribute(a,rr.reason,i,catalogue)
                else:evidence(a,i,catalogue)
            extra=len(out)-start-5
            require(extra<=32,'reason aggregate bound')
            out[start]=replace(out[start],payload=replace(out[start].payload,following=4+extra))
        if draft is None:meta(Role.EVALUATE if rr is None else Role.RETURN_END,owner,Kind.SNAPSHOT if rr is None else Kind.RESULT)
        n=len(out)
        require(n<=(256 if rr is None else 64),'packet length')
        if draft is None:out[0]=replace(out[0],payload=replace(out[0].payload,following=n-1,complete=n))
        for p in out:
            if isinstance(p.payload,Metadata):p.payload.encode(p.role)
        return Packet(tuple(out),tuple(t.refs),snapshot,tuple(candidate_ends),result)

    def request(self, snapshot):
        return self._build(snapshot)

    def draft(self, spec):
        if spec.question is None or spec.target is None:return Packet((),(),None)
        return self._build(None,draft=spec,sealed=False)

    def object(self, ref):
        if ref.kind==Kind.SNAPSHOT:return self.request(ref)
        return self._build(None,record=ref,sealed=False)

    def result(self, result, request_packet, catalogue):
        rr=self.registry.get(result,Kind.RESULT)
        require(rr.snapshot==request_packet.snapshot,'pending result binding')
        # K contains original immutable objects, not selectable completion IDs.
        self.catalogue=frozenset(catalogue)
        try:return self._build(rr.snapshot,result,request_packet.handles)
        finally:self.catalogue=None

    def return_max(self, snapshot):
        s=self.registry.get(snapshot,Kind.SNAPSHOT)
        if s.port==Port.NOUL:return 50
        if s.port==Port.SCORE:return 61
        bank=self.registry.get(s.bank,Kind.BANK)
        maximum=max(sum(a.cost for a in self.registry.get(c).attributes) for c in bank.candidates)
        require(maximum<=46,'all-candidate return bound')
        return max(50,18+maximum)
