"""Owned immutable records. Tensor copies preserve CUDA autograd connections."""
from dataclasses import dataclass, fields, is_dataclass, replace
from typing import Any
from .schema import Kind, Port, Status, ValueKind, Mode, Invalid, require, integer, encode_atoms, parse_numeric
from .layout import LEGACY, DIRECT, valid_profile, semantic_lengths, attribute_cost


@dataclass(frozen=True)
class Ref:
    run: str
    kind: Kind
    identity: int
    version: int = 1

    def __post_init__(self):
        require(type(self.run) is str and bool(self.run) and self.kind != Kind.NONE, 'Ref identity')
        integer(self.identity, 1, (1 << 63) - 1, 'identity')
        integer(self.version, 1, (1 << 63) - 1, 'version')


@dataclass(frozen=True)
class Span:
    source: Ref
    start: int
    count: int
    clip_start: int = 0
    clip_end: int | None = None


@dataclass(frozen=True)
class Source:
    text: bytes
    token_ids: tuple
    offsets: tuple
    states: Any


@dataclass(frozen=True)
class Value:
    kind: ValueKind
    present: bool
    data: Any = None
    mode: Mode = Mode.NONE
    provenance: Ref | None = None
    copied_owner: Ref | None = None
    copied_index: int | None = None
    unit: Ref | None = None
    parsed_span: Span | None = None

    def body_length(self):
        if not self.present:
            return 0
        if self.kind == ValueKind.SOURCE_STRING:
            return self.data.count
        return len(encode_atoms(self.kind, self.data, 0 if self.kind == ValueKind.REFERENCE else None))


@dataclass(frozen=True)
class Attribute:
    keys: Any
    value: Value
    profile: str = LEGACY

    @property
    def cost(self):
        return attribute_cost(self.profile,2 if self.profile==LEGACY else len(self.keys),self.value.body_length())


@dataclass(frozen=True)
class Semantic:
    slots: Any
    attributes: tuple = ()
    direction: int = 0
    midpoint: bool = False
    profile: str = LEGACY
    lengths: tuple = ()


@dataclass(frozen=True)
class Bank:
    candidates: tuple
    profile: str = LEGACY


@dataclass(frozen=True)
class History:
    result: Ref


@dataclass(frozen=True)
class Reason:
    slots: Any
    attachments: tuple = ()
    profile: str = LEGACY


@dataclass(frozen=True)
class Snapshot:
    port: Port
    question: Ref
    target: Ref
    evidence: tuple = ()
    bank: Ref | None = None
    rubric: Ref | None = None
    attempt: int = 1
    frontier: int = 0
    profile: str = 'b1-full-v1-compensated'


@dataclass(frozen=True)
class Result:
    snapshot: Ref
    status: Status
    scalar: Any = None
    selected: Ref | None = None
    selected_index: int | None = None
    reason: Ref | None = None
    presentation: Any = None


def tensor(x, shape):
    require(hasattr(x, 'is_cuda') and x.is_cuda, 'tensor payload requires CUDA; no CPU fallback')
    require(tuple(x.shape) == tuple(shape), 'tensor shape')
    require(x.dtype.__str__() == 'torch.float32', 'protocol tensor must be FP32')
    require(bool(x.isfinite().all()), 'nonfinite tensor')


def owned(x):
    if hasattr(x, 'is_cuda'):
        require(x.is_cuda, 'owned tensor requires CUDA')
        return x.clone()  # not detach(): old versions own storage AND keep gradients
    if is_dataclass(x):
        return replace(x, **{f.name: owned(getattr(x, f.name)) for f in fields(x)})
    if isinstance(x, tuple):
        return tuple(owned(v) for v in x)
    require(x is None or isinstance(x, (str, bytes, int, float, bool)), 'mutable/unsupported record field')
    return x


class Registry:
    def __init__(self, run):
        require(type(run) is str and bool(run), 'run identity')
        self.run = run
        self._records = {}
        self._next = 1

    def contains(self, ref, kind=None):
        return isinstance(ref, Ref) and ref.run == self.run and ref in self._records and (kind is None or ref.kind == kind)

    def get(self, ref, kind=None):
        require(self.contains(ref, kind), 'foreign/missing/wrong-kind Ref')
        return owned(self._records[ref])

    def refs(self):
        return tuple(self._records)

    def fork(self):
        other=Registry(self.run)
        other._records=self._records.copy();other._next=self._next
        return other

    def adopt(self, other):
        require(other.run==self.run and all(other._records.get(r) is x for r,x in self._records.items()),
                'transaction changed an existing immutable record')
        self._records=other._records;self._next=other._next

    def publish(self, kind, record, previous=None):
        kind = Kind(kind)
        self.validate(kind, record)
        if previous is not None:
            require(self.contains(previous, kind), 'invalid previous version')
            ref = Ref(self.run, kind, previous.identity, previous.version + 1)
            require(ref not in self._records, 'version already exists')
        else:
            ref = Ref(self.run, kind, self._next)
        record = owned(record)
        # Canonicalize semantic contents once; raw source states and continuous
        # scalars are deliberately not normalized.
        def canon(t):
            m=t.square().mean(-1,keepdim=True)
            return t / (m.where(m > 1e-12, m.new_full((), 1e-12))).sqrt()
        def attrs(items):
            return tuple(replace(a, keys=a.keys if a.profile==DIRECT else canon(a.keys)) for a in items)
        if isinstance(record, Semantic):
            slots=record.slots if record.profile==DIRECT else canon(record.slots)
            if kind == Kind.RUBRIC and not record.midpoint and record.profile==LEGACY:
                slots = __import__('torch').cat((slots[:8], slots.new_zeros(2,1024)))
            record=replace(record,slots=slots,attributes=attrs(record.attributes))
        elif isinstance(record, Reason):
            record=replace(record,slots=canon(record.slots),attachments=tuple(
                replace(a,keys=a.keys if a.profile==DIRECT else canon(a.keys)) if isinstance(a,Attribute) else a for a in record.attachments))
        self._records[ref] = record
        if previous is None:
            self._next += 1
        return ref

    def span(self, span):
        require(isinstance(span, Span), 'source span')
        source=self.get(span.source, Kind.SOURCE)
        integer(span.start,0,len(source.token_ids)-1,'span start')
        integer(span.count,1,32,'span count')
        require(span.start+span.count <= len(source.token_ids),'span end')
        a,b=source.offsets[span.start];z,w=source.offsets[span.start+span.count-1]
        end=w-z if span.clip_end is None else span.clip_end
        integer(span.clip_start,0,b-a,'clip start');integer(end,0,w-z,'clip end')
        left,right=a+span.clip_start,z+end
        require(left < right and left < b and right > z,'empty/noncovered span')
        try:
            source.text[:left].decode('utf8');source.text[:right].decode('utf8')
            literal=source.text[left:right];literal.decode('utf8')
        except UnicodeError as exc:
            raise Invalid('illegal UTF8 clipping boundary') from exc
        return source.states[span.start:span.start+span.count], literal, span.clip_start, end

    def attribute(self, attr, catalogue=None):
        require(isinstance(attr, Attribute),'attribute')
        valid_profile(attr.profile)
        length=2 if attr.profile==LEGACY else integer(len(attr.keys),1,32,'direct key length')
        tensor(attr.keys,(length,1024));v=attr.value
        require(v.kind in tuple(ValueKind)[1:7] and type(v.present) is bool,'attribute type/presence')
        def ref(r):
            require(self.contains(r),'attribute reference')
            require(catalogue is None or r in catalogue,'reference outside frozen catalogue')
        if v.unit is not None:ref(v.unit)
        if not v.present:
            require(v.data is None and v.mode == Mode.NONE and v.provenance is v.copied_owner is v.parsed_span is None
                    and v.copied_index is None,'absent value fields')
            return
        require(v.mode in (Mode.COPY,Mode.PARSE,Mode.CONSTRUCT),'creation mode')
        if v.provenance is not None:ref(v.provenance)
        if v.kind == ValueKind.SOURCE_STRING:
            self.span(v.data);ref(v.data.source)
            require(v.provenance == v.data.source,'source-string provenance')
        elif v.kind == ValueKind.REFERENCE:
            ref(v.data)
        else:encode_atoms(v.kind,v.data)
        if v.mode == Mode.COPY:
            ref(v.copied_owner)
            require(v.copied_owner.kind in (Kind.TARGET,Kind.ALTERNATIVE),'copy owner type')
            owner=self.get(v.copied_owner)
            integer(v.copied_index,0,len(owner.attributes)-1,'copied attribute path')
            old=owner.attributes[v.copied_index].value
            require((v.kind,v.present,v.data,v.unit)==(old.kind,old.present,old.data,old.unit),'copy value mismatch')
            require(v.parsed_span is None,'copy parse span')
        else:
            require(v.copied_owner is None and v.copied_index is None,'unexpected copied path')
        if v.mode == Mode.PARSE:
            require(v.kind in (ValueKind.INT64,ValueKind.DECIMAL) and v.parsed_span is not None,'numeric-only parse')
            _,literal,_,_=self.span(v.parsed_span);ref(v.parsed_span.source)
            require(v.provenance == v.parsed_span.source and parse_numeric(literal,v.kind)==v.data,'parse mismatch')
        else:require(v.parsed_span is None,'unexpected parsed span')

    def validate(self, kind, x):
        if kind == Kind.SOURCE:
            require(isinstance(x,Source) and type(x.text) is bytes,'source')
            require(len(x.offsets)==len(x.token_ids)>0,'source mapping')
            tensor(x.states,(len(x.token_ids),1024))
            last=0
            for a,b in x.offsets:
                require(type(a) is int and type(b) is int and a==last and a<b<=len(x.text),'source byte mapping')
                last=b
            require(last==len(x.text),'incomplete source mapping')
            x.text.decode('utf8')
        elif kind in (Kind.QUESTION,Kind.TARGET,Kind.ALTERNATIVE,Kind.RUBRIC):
            require(isinstance(x,Semantic),'semantic record')
            lengths=semantic_lengths(kind,x.profile,x.lengths,midpoint=x.midpoint)
            tensor(x.slots,(sum(lengths),1024))
            require(len(x.attributes)<=8 and (kind in (Kind.TARGET,Kind.ALTERNATIVE) or not x.attributes),'attribute owner/count')
            for a in x.attributes:
                require(a.profile==x.profile,'attribute layout mismatch');self.attribute(a)
            if kind == Kind.RUBRIC:require(x.direction in (1,2) and type(x.midpoint) is bool,'rubric metadata')
            else:require(x.direction==0 and not x.midpoint,'non-rubric fields')
        elif kind == Kind.BANK:
            require(isinstance(x,Bank) and 1<=len(x.candidates)<=16,'candidate bank')
            require(len(set(x.candidates))==len(x.candidates),'duplicate candidates')
            valid_profile(x.profile)
            for r in x.candidates:
                require(self.contains(r,Kind.ALTERNATIVE),'candidate Ref')
                require(self.get(r).profile==x.profile,'bank layout mismatch')
        elif kind == Kind.SNAPSHOT:
            require(isinstance(x,Snapshot) and x.port in (Port.NOUL,Port.CHOICE,Port.SCORE),'snapshot')
            valid_profile(x.profile)
            integer(x.attempt,1,16,'attempt');integer(x.frontier,0,262144,'frontier')
            require(self.contains(x.question,Kind.QUESTION) and self.contains(x.target,Kind.TARGET),'question/target binding')
            require(self.get(x.question).profile==self.get(x.target).profile==x.profile,'snapshot role layout mismatch')
            require((x.bank is not None)==(x.port==Port.CHOICE),'bank applicability')
            require((x.rubric is not None)==(x.port==Port.SCORE),'rubric applicability')
            if x.bank is not None:require(self.contains(x.bank,Kind.BANK),'bank binding')
            if x.rubric is not None:require(self.contains(x.rubric,Kind.RUBRIC),'rubric binding')
            for ref in (x.bank,x.rubric):
                if ref is not None:require(self.get(ref).profile==x.profile,'snapshot layout mismatch')
            require(len(x.evidence)<=8,'evidence count')
            for e in x.evidence:self.evidence(e)
        elif kind == Kind.REASON:
            require(isinstance(x,Reason) and len(x.attachments)<=8,'reason attachments')
            valid_profile(x.profile)
            tensor(x.slots,(4,1024))
            for a in x.attachments:
                if isinstance(a,Attribute):
                    require(a.profile==x.profile,'reason attribute layout mismatch');self.attribute(a)
                else:self.evidence(a)
        elif kind == Kind.RESULT:
            require(isinstance(x,Result) and self.contains(x.snapshot,Kind.SNAPSHOT),'result snapshot')
            snap=self.get(x.snapshot);require(x.status in tuple(Status)[1:],'semantic status')
            require(x.status!=Status.NO_FEASIBLE_CHOICE or snap.port==Port.CHOICE,'Choice-only status')
            if x.status==Status.ASSESSED:
                require(x.reason is None,'assessed reason')
                if snap.port==Port.CHOICE:
                    bank=self.get(snap.bank)
                    integer(x.selected_index,0,len(bank.candidates)-1,'selected index')
                    require(x.selected==bank.candidates[x.selected_index] and x.scalar is None,'selected binding')
                else:
                    tensor(x.scalar,())
                    require(bool((x.scalar>=0)&(x.scalar<=1)) and x.selected is x.selected_index is None,'scalar value')
            else:
                require(x.scalar is x.selected is x.selected_index is None,'non-assessed payload')
                require(self.contains(x.reason,Kind.REASON),'non-assessed reason')
                require(self.get(x.reason).profile==snap.profile,'result Reason layout mismatch')
        else:raise Invalid('unsupported object kind')

    def evidence(self, e):
        if isinstance(e,Span):self.span(e)
        else:
            require(isinstance(e,History) and self.contains(e.result,Kind.RESULT),'historical evidence')
            require(self.get(e.result).presentation is not None,'historical presentation missing')
