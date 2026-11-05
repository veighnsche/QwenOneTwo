"""Small engineering fixtures, not semantic training data or reviewed labels."""
import torch
from .schema import Port, Kind, ValueKind, Mode
from .records import Source, Semantic, Attribute, Value, Span


def slots(n,requires_grad=False):
    return torch.randn(n,1024,device='cuda',dtype=torch.float32,requires_grad=requires_grad)


def source(reg,text=b'1.20 alpha ready',width=1):
    offsets=tuple((i,min(i+width,len(text))) for i in range(0,len(text),width))
    return reg.publish(Kind.SOURCE,Source(text,tuple(range(len(offsets))),offsets,slots(len(offsets),True)))


def base(runtime,port=None,attributes=(),candidate_attributes=(),count=2,evidence=True):
    reg=runtime.registry
    src=source(reg)
    runtime.begin(port or Port.NOUL)
    q=runtime.publish(Kind.QUESTION,Semantic(slots(8,True)))
    t=runtime.publish(Kind.TARGET,Semantic(slots(2,True),tuple(attributes)))
    candidates=tuple(runtime.publish(Kind.ALTERNATIVE,Semantic(slots(4,True),tuple(candidate_attributes)))
                     for _ in range(count)) if runtime.call.port==Port.CHOICE else ()
    rubric=runtime.publish(Kind.RUBRIC,Semantic(slots(10,True),direction=1,midpoint=False)) if runtime.call.port==Port.SCORE else None
    runtime.writer_position();runtime.writer_position()
    packet=runtime.seal(q,t,(Span(src,0,4),) if evidence else (),rubric,candidates)
    return packet,q,t,candidates,rubric,src


def reference_boundary(runtime):
    """The spec's q255/h222/all-completion-handles256 exact fixture."""
    reg=runtime.registry
    def quad():
        s=source(reg,b'x')
        unit=reg.publish(Kind.TARGET,Semantic(slots(2)))
        value=reg.publish(Kind.TARGET,Semantic(slots(2)))
        old=Attribute(slots(2),Value(ValueKind.REFERENCE,True,value,Mode.CONSTRUCT,s,unit=unit))
        owner=reg.publish(Kind.TARGET,Semantic(slots(2),(old,)))
        return Attribute(slots(2),Value(ValueKind.REFERENCE,True,value,Mode.COPY,s,owner,0,unit))
    request_attrs=[quad() for _ in range(53)]
    reason_attrs=tuple(quad() for _ in range(8))
    runtime.begin(Port.CHOICE)
    q=runtime.publish(Kind.QUESTION,Semantic(slots(8,True)))
    t=runtime.publish(Kind.TARGET,Semantic(slots(2,True),tuple(request_attrs[:8])))
    refs=[];offset=8
    for count in (8,8,8,8,8,5):
        refs.append(runtime.publish(Kind.ALTERNATIVE,Semantic(slots(4,True),tuple(request_attrs[offset:offset+count]))))
        offset+=count
    runtime.writer_position();runtime.writer_position()
    packet=runtime.seal(q,t,candidates=tuple(refs))
    return packet,reason_attrs
