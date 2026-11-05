"""Serial lifecycle and independent position/handle/record reservations."""
from dataclasses import dataclass, field, replace
from .schema import Port, Kind, Status, Role, Invalid, require, integer, bits
from .records import Registry, Snapshot, Bank, Result, Reason, Attribute, Span, History, owned
from .render import Renderer


class NotAdmitted(Invalid):
    def __init__(self, violations):
        self.violations=tuple(violations)
        self.code=self.violations[0]
        super().__init__(', '.join(self.violations))


@dataclass
class Call:
    ordinal: int
    port: Port
    frontier: int
    phase: int
    allocated: int = 0
    writer: int = 0
    state: str = 'Writer'
    previous: int | None = None
    outcome: int = 0
    noticed: bool = False
    snapshot: object = None
    request: object = None
    catalogue: tuple = ()
    position_hold: int = 0
    handle_hold: int = 0
    record_hold: int = 2
    q_remaining: int = 0
    r_remaining: int = 0
    result: object = None
    consumed: bool = False
    released_positions: int = 0
    released_records: int = 0
    released_handles: int = 0
    violations: tuple = ()

    @property
    def sealing_hold(self):
        return (2 if self.port==Port.CHOICE else 1) if self.state=='Writer' else 0


class Runtime:
    def __init__(self, run, prompt, budget, answer, context=262144):
        integer(context,1,262144,'context');integer(prompt,0,context,'prompt')
        integer(budget,0,131072,'B');integer(answer,1,262144,'A')
        require(prompt+answer+64<=context,'P+A+H context admission')
        require(budget<=context-prompt-answer-64,'B context grant')
        self.registry=Registry(run)
        self.prompt=prompt;self.budget=budget;self.answer=answer;self.context=context
        self.b=self.a=self.h=0;self.phase=0;self.attempts=self.dispatched=self.consumptions=0
        self.call=None;self.calls=[];self.events=[];self.state='Prefill';self.terminal=None

    @property
    def position(self):return self.prompt+self.b+self.a+self.h

    @property
    def remaining(self):return self.budget-self.b,self.answer-self.a,64-self.h,self.context-self.position

    def debit(self,kind,count,label):
        integer(count,1,262144,'position count')
        require(self.terminal is None,'post-terminal work')
        b,a,h,c=self.remaining
        require(count<={'B':b,'A':a,'H':h}[kind] and count<=c,'position capacity')
        require(kind!='A' or self.phase==1,'answer before closure')
        old=self.position
        if kind=='B':self.b+=count
        elif kind=='A':self.a+=count
        elif kind=='H':self.h+=count
        else:raise Invalid('position kind')
        self.events.append({'position':old,'count':count,'kind':kind,'label':label,'phase':self.phase})

    def startup(self):
        require(self.state=='Prefill','duplicate startup')
        self.debit('H',3,'startup');self.state='Native'

    def begin(self,port):
        require(self.state=='Native' and self.terminal is None,'nested/ineligible call')
        require(port in (Port.NOUL,Port.CHOICE,Port.SCORE),'optional operation')
        require(self.b<self.budget and self.attempts<16,'optional operation masked by resource limit')
        self.attempts+=1
        self.call=Call(self.attempts,port,self.position,self.phase)
        self.calls.append(self.call);self.state='Writer'
        return self.call

    def publish(self,kind,record,previous=None):
        c=self.call;require(c is not None and c.state=='Writer','publication outside writer')
        require(kind in (Kind.QUESTION,Kind.TARGET,Kind.ALTERNATIVE,Kind.RUBRIC),'writer cannot mint source/result/reason objects')
        require(c.allocated+1+c.sealing_hold+c.record_hold<=32,'ObjectLimit')
        ref=self.registry.publish(kind,record,previous)
        c.allocated+=1
        return ref

    def writer_position(self):
        c=self.call;require(self.state=='Writer' and c.writer<32,'WriterLimit')
        self.debit('B',1,'writer_start' if c.writer==0 else 'writer_tick')
        c.writer+=1

    def writer_action(self,action,success=True):
        c=self.call;require(self.state=='Writer' and c.writer>=2,'writer event before tick')
        integer(action,0,6,'writer action');c.previous=action
        if success:c.outcome=0
        if action==6:self.end_attempt('WriterAborted',12)
        elif c.writer==32 and action!=5:self.end_attempt('WriterLimit',5)

    def seal(self,question,target,evidence=(),rubric=None,candidates=()):
        c=self.call;require(self.state=='Writer' and c.writer<=32,'seal outside writer')
        # Structural failure leaves the real draft unsealed and unchanged.
        staged=self.registry.fork()
        bank=staged.publish(Kind.BANK,Bank(tuple(candidates))) if c.port==Port.CHOICE else None
        require(c.port==Port.CHOICE or not candidates,'non-Choice candidates')
        spec=Snapshot(c.port,question,target,tuple(evidence),bank,rubric,c.ordinal,c.frontier)
        snap=staged.publish(Kind.SNAPSHOT,spec)
        renderer=Renderer(staged);packet=renderer.request(snap);rmax=renderer.return_max(snap)
        self.registry.adopt(staged)
        c.allocated+=c.sealing_hold;c.state='Sealed';c.snapshot=snap;c.request=packet
        c.catalogue=self.registry.refs()
        m=len(set(c.catalogue)-set(packet.handles));hhold=2+min(32,m)
        hold=len(packet)+rmax;b,a,h,context=self.remaining
        violations=[]
        if c.allocated+c.record_hold>32:violations.append('ObjectLimit')
        if len(packet.handles)+hhold>256:violations.append('HandleLimit')
        if b<hold:violations.append('BudgetExhausted')
        if context-a-h<hold:violations.append('ContextLimit')
        c.violations=tuple(violations)
        if violations:
            self.end_attempt(violations[0],{'ObjectLimit':7,'HandleLimit':7,'BudgetExhausted':3,'ContextLimit':4}[violations[0]])
            raise NotAdmitted(violations)
        c.position_hold=hold;c.handle_hold=hhold;c.q_remaining=len(packet)
        c.state='Evaluating';self.state='Evaluating';self.dispatched+=1
        return packet

    def semantic_positions(self,count,returning=False):
        c=self.call
        require(self.state==('Delivering' if returning else 'Evaluating'),'semantic phase')
        remaining=c.r_remaining if returning else c.q_remaining
        integer(count,1,remaining,'reserved position count')
        require(count<=c.position_hold,'position hold')
        self.debit('B',count,'return' if returning else 'request')
        c.position_hold-=count
        if returning:c.r_remaining-=count
        else:c.q_remaining-=count

    def prepare_result(self,snapshot,status,scalar=None,selected_index=None,reason=None):
        c=self.call
        # Foreign/duplicate/late completions cannot mutate records, cache or ledger.
        if self.terminal is not None or c is None or c.state!='Evaluating' or c.snapshot!=snapshot or c.result is not None:
            return None
        require(c.q_remaining==0,'completion before request processed')
        staged=self.registry.fork();reason_ref=None
        if status!=Status.ASSESSED:
            require(isinstance(reason,Reason),'missing reason')
            for item in reason.attachments:
                if isinstance(item,Attribute):staged.attribute(item,c.catalogue)
                elif isinstance(item,Span):require(item.source in c.catalogue,'future reason source')
                elif isinstance(item,History):require(item.result in c.catalogue,'future reason history')
            reason_ref=staged.publish(Kind.REASON,reason)
        else:require(reason is None,'assessed reason')
        spec=staged.get(snapshot)
        selected=None
        if spec.port==Port.CHOICE and status==Status.ASSESSED:
            bank=staged.get(spec.bank)
            integer(selected_index,0,len(bank.candidates)-1,'selected index')
            selected=bank.candidates[selected_index]
        result=Result(snapshot,status,scalar,selected,selected_index,reason_ref)
        result_ref=staged.publish(Kind.RESULT,result)
        packet=Renderer(staged).result(result_ref,c.request,c.catalogue)
        require(len(packet)<=c.position_hold,'return exceeded reservation')
        added=len(packet.handles)-len(c.request.handles)
        used=2 if reason_ref is not None else 1
        require(added<=c.handle_hold and used<=c.record_hold and c.allocated+used<=32,'completion resource hold')
        require(packet.handles[:len(c.request.handles)]==c.request.handles,'return handle prefix')
        # Result identity and its presentation become visible atomically. No
        # existing published record is edited or detached.
        staged._records[result_ref]=owned(replace(staged._records[result_ref],presentation=packet))
        self.registry.adopt(staged)
        c.allocated+=used;c.released_records=c.record_hold-used;c.record_hold=0
        c.released_handles=c.handle_hold-added;c.handle_hold=0
        c.result=result_ref;c.r_remaining=len(packet);c.state='Delivering';self.state='Delivering'
        return packet

    def consumed(self,snapshot):
        c=self.call
        if self.terminal is not None or c is None or c.snapshot!=snapshot or c.consumed:return False
        require(self.state=='Delivering' and c.r_remaining==0,'incomplete return')
        c.consumed=True;c.state='Consumed';self.consumptions+=1
        c.released_positions+=c.position_hold;c.position_hold=0
        require(self.phase==c.phase,'return changed native phase')
        self.state='Native'
        return True

    def end_attempt(self,label,code):
        c=self.call;require(c is not None and not c.consumed,'closed attempt')
        c.released_positions+=c.position_hold;c.position_hold=0
        c.released_records+=c.record_hold+c.sealing_hold;c.record_hold=0
        c.released_handles+=c.handle_hold;c.handle_hold=0
        c.state=label;c.outcome=code;self.state='Native'

    def notice(self):
        c=self.call;require(c is not None and c.outcome and not c.noticed and self.state=='Native','duplicate/unneeded notice')
        self.debit('H',1,'notice');c.noticed=True

    def close(self):
        require(self.state=='Native' and self.phase==0,'invalid phase closure')
        self.debit('H',2,'thinking_close');self.phase=1

    def native(self,count=1):
        require(self.state=='Native','native sampling during call')
        if self.phase==0 and self.b==self.budget:self.close()
        self.debit('A' if self.phase else 'B',count,'native')

    def terminate(self,outcome,usable=True,emit_end=True):
        if self.terminal is not None:return False
        require(outcome in ('Completed','Limited','Cancelled','Failed'),'terminal outcome')
        if self.call is not None and self.call.state not in ('Consumed',):
            self.end_attempt(outcome,9 if outcome=='Cancelled' else 8)
        if emit_end and outcome in ('Completed','Limited') and usable and self.h<64 and self.position<self.context:
            self.debit('H',1,'chat_end' if outcome=='Completed' else 'forced_chat_end')
        self.terminal=outcome;self.state='Terminal'
        return True

    def counter_values(self):
        b,a,h,c=self.remaining
        draft=self.call if self.state=='Writer' else None
        return (b,a,h,c,32-draft.writer if draft else 32,16-self.attempts,
                32-draft.allocated if draft else 32,self.phase)

    def counters(self):
        import torch
        from .neural import cuda_only
        cuda_only()
        return torch.tensor(self.counter_values(),device='cuda',dtype=torch.float32)/torch.tensor(
            (131072,262144,64,262144,32,16,32,1),device='cuda',dtype=torch.float32)

    def control_bits(self,role,port=None,length=0,candidates=0,evidence=0,children=0):
        require(role in (Role.WRITER_START,Role.WRITER_TICK,Role.LIMIT),'control role not emitted in v1')
        draft=self.call if self.state=='Writer' or role==Role.LIMIT else None
        b,a,h,c,w,t,o,phase=self.counter_values()
        if draft is not None:w=32-draft.writer;o=32-draft.allocated
        require(0<=length<=256 and 0<=candidates<=16 and 0<=evidence<=8 and 0<=children<=8,'control counts')
        return bits([(0,4,1),(4,3,int(port if port is not None else (draft.port if draft else Port.NONE))),
            (7,1,phase),(8,18,b),(26,19,a),(45,7,h),(52,19,c),(71,6,w),(77,5,t),(82,6,o),
            (88,3,0 if not draft or draft.previous is None else draft.previous+1),(91,9,length if draft else 0),
            (100,6,draft.allocated if draft else 0),(106,5,candidates if draft else 0),
            (111,4,evidence if draft else 0),(115,4,children if draft else 0),
            (119,1,int(bool(draft and draft.snapshot))),(120,1,0),(121,4,draft.outcome if draft else 0)])

    def summary(self):
        return {'P':self.prompt,'B':self.b,'A':self.a,'H':self.h,'total':self.position,
                'attempts':self.attempts,'dispatched':self.dispatched,'consumed':self.consumptions,
                'phase':self.phase,'state':self.state,'terminal':self.terminal,
                'held_positions':sum(c.position_hold for c in self.calls),
                'released_positions':sum(c.released_positions for c in self.calls),
                'new_records':sum(c.allocated for c in self.calls),'events':self.events}
