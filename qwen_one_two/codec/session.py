"""One-backbone adapter execution, shared by prescribed qualification and runtime."""
import torch
from .schema import Role,Kind,Port,Status,require
from .records import Semantic,Source,Span,Reason
from .render import Renderer
from .writer import Writer,decode_result
from .runtime import Runtime
from .neural import thinking_budget
import uuid


class Session:
    def __init__(self,backbone,codec,runtime,cancelled=None):
        self.backbone=backbone;self.codec=codec;self.runtime=runtime
        self.state=backbone.initial();self.hidden=None;self.mode=None
        self.prefills=0;self.switches=[];self.version_checks=0;self.boundaries=[]
        self.calls=[];self.snapshots=[]
        self.cancelled=cancelled or (lambda:False)
        self.raw_budget=None;self.predicted_budget=None;self.override=None

    @classmethod
    def open(cls,backbone,codec,ids,answer,override=None,run=None,cancelled=None):
        rt=Runtime(run or str(uuid.uuid4()),len(ids),0,answer)
        s=cls(backbone,codec,rt,cancelled)
        s.prefill(ids)
        raw=backbone.table.scalar(248077,s.hidden[0,-1])
        predicted=int(thinking_budget(raw));available=rt.context-rt.prompt-rt.answer-64
        if override is not None:
            require(type(override) is int and 0<=override<=131072 and override<=available,'manual budget override')
        rt.budget=min(predicted,available) if override is None else override
        s.predicted_budget=predicted;s.override=override
        s.startup(raw)
        if rt.budget==0:s.close()
        return s

    def step(self,embeds,mode,label,capture=True):
        require(embeds.is_cuda,'CUDA inputs required')
        require(self.runtime.terminal is None,'post-terminal inference')
        if self.cancelled():
            self.runtime.terminate('Cancelled',emit_end=False)
            raise InterruptedError('cancelled before scheduling model work')
        old=self.state;versions=tuple(t._version for pair in old.layers for t in pair)
        if mode!=self.mode:self.switches.append((old.position,self.mode,mode));self.mode=mode
        try:
            self.hidden,self.state=self.backbone(old,mode,embeds)
            checks=((bool(self.hidden.isfinite().all()) and all(bool(t.isfinite().all()) for pair in self.state.layers for t in pair),'nonfinite causal state'),
                    (versions==tuple(t._version for pair in old.layers for t in pair),'mutated prior causal state'),
                    (old.position+len(embeds)==self.state.position==self.runtime.position,'model/ledger position divergence'),
                    (self.state.lineage==old.lineage==1,'cache lineage reset'))
            for valid,message in checks:
                if not valid:raise RuntimeError(message)
        except RuntimeError:
            self.runtime.terminate('Failed',usable=False,emit_end=False)
            raise
        self.version_checks+=len(versions)
        self.boundaries.append((label,self.state.position))
        if capture:self.snapshots.append((label,self.hidden.detach().clone(),self.backbone.snapshot_state(self.state)))
        if self.cancelled():
            self.runtime.terminate('Cancelled',emit_end=False)
            raise InterruptedError('cancelled after charged model work; discard late output')
        return self.hidden

    def prefill(self,ids):
        require(self.prefills==0 and len(ids)==self.runtime.prompt,'single prefill')
        self.prefills+=1
        return self.step(self.backbone.embedding(ids)[0],'D','prefill')

    def startup(self,raw=None):
        if raw is None:raw=self.backbone.table.scalar(248077,self.hidden[0,-1])
        self.raw_budget=raw
        if self.predicted_budget is None:self.predicted_budget=int(thinking_budget(raw))
        self.runtime.startup()
        self.step(self.codec.encode(self.codec.startup(raw)),'R','startup')
        return raw

    def native(self,ids,loss=False):
        losses=[];logits=[]
        for token in ids:
            if self.runtime.phase==0 and self.runtime.b==self.runtime.budget:self.close()
            if loss:
                scores,indices=self.backbone.ordinary_logits(self.hidden[0,-1],self.runtime.phase)
                target=(indices==token).nonzero().flatten()
                require(len(target)==1,'native target masked')
                losses.append(torch.nn.functional.cross_entropy(scores[None],target))
                logits.append(scores.detach().clone())
            self.runtime.native()
            self.step(self.backbone.embedding([token])[0],'R','native',False)
        return losses,logits

    def close(self):
        self.runtime.close()
        self.step(self.backbone.embedding([248069,271])[0],'R','thinking_close')

    def begin_writer(self,port):
        self.runtime.begin(port)
        writer=Writer(self.runtime,self.codec)
        self.step(writer.tick_vector(self.hidden[0,-1])[None],'R','writer_start',False)
        return writer

    def evaluate(self,packet):
        require(packet is self.runtime.call.request and packet.snapshot==self.runtime.call.snapshot,
                'unadmitted or mismatched request presentation')
        # The authoritative snapshot owns the content. A mutable Tensor view in
        # a returned Packet must not be able to rewrite a sealed invocation.
        packet=Renderer(self.runtime.registry).request(self.runtime.call.snapshot)
        self.runtime.semantic_positions(len(packet))
        inputs=self.codec.encode(packet)
        self.backbone.capture_evaluator(torch.is_grad_enabled())
        h=self.step(inputs,'D','request')
        during=[v for dual in self.backbone.duals for v in dual.evaluation_outputs]
        self.backbone.capture_evaluator(False)
        candidate=torch.stack([h[0,i] for _,i in packet.candidate_ends]) if packet.candidate_ends else None
        evaluated=self.codec.evaluate(h[0,-1],self.runtime.call.port,candidate)
        return inputs,evaluated,during

    def deliver(self,packet):
        if packet is None:return False
        if self.runtime.terminal is not None or self.runtime.state!='Delivering' or self.runtime.call.consumed or packet.snapshot!=self.runtime.call.snapshot:
            return False
        require(self.runtime.call.result==packet.result,'return not current')
        packet=self.runtime.registry.get(packet.result,Kind.RESULT).presentation
        self.runtime.semantic_positions(len(packet),True)
        self.step(self.codec.encode(packet),'R','return')
        require(self.runtime.consumed(packet.snapshot),'duplicate return')
        return True

    def learned_call(self,port):
        """Actual head decisions; invalid untrained output follows mechanical exits."""
        writer=self.begin_writer(port)
        try:
            writer.initialise(self.hidden[0,-1])
            while self.runtime.state=='Writer':
                self.step(writer.tick_vector(self.hidden[0,-1])[None],'R','writer_tick',False)
                packet,error=writer.execute(self.hidden[0,-1])
                if packet is not None:
                    _,evaluated,_=self.evaluate(packet)
                    return self.deliver(decode_result(self.codec,self.runtime,evaluated))
            if self.runtime.call.outcome and not self.runtime.call.noticed:self.notice()
            return False
        except (ValueError,StopIteration) as exc:
            if bool(self.hidden.isfinite().all()) and 'nonfinite' not in str(exc):
                code=3 if self.runtime.b==self.runtime.budget else (5 if self.runtime.call.writer>=32 else
                     (7 if 'ObjectLimit' in str(exc) else 2))
                self.runtime.end_attempt({3:'BudgetExhausted',5:'WriterLimit',7:'ObjectLimit',2:'Malformed'}[code],code)
                self.notice()
                return False
            self.runtime.terminate('Failed',usable=False,emit_end=False)
            raise

    def notice(self):
        if self.runtime.h>=64 or self.runtime.position>=self.runtime.context:
            self.runtime.terminate('Limited',usable=False,emit_end=False)
            return
        self.runtime.notice()
        payload=self.codec.meta(torch.tensor(self.runtime.control_bits(Role.LIMIT),device='cuda',dtype=torch.float32))
        self.step(self.codec.wire(payload,Role.LIMIT)[None],'R','mechanical_notice',False)

    def native_step(self,cancelled=False):
        """Unforced learned gate/token step; no semantic keyword rules."""
        rt=self.runtime
        if rt.terminal is not None:return None
        if cancelled:
            rt.terminate('Cancelled',emit_end=False)
            return None
        require(rt.state=='Native','native boundary inside operation')
        if rt.phase==0 and rt.b==rt.budget:self.close()
        if rt.phase==1 and rt.a==rt.answer:
            rt.debit('H',1,'forced_chat_end');self.step(self.backbone.embedding([248046])[0],'R','forced_chat_end',False)
            rt.terminate('Limited',emit_end=False)
            return None
        try:
            logits=self.codec.gate(self.codec.feature(self.hidden[0,-1],rt.counters()))
        except ValueError:
            rt.terminate('Failed',usable=False,emit_end=False)
            raise
        if rt.b==rt.budget or rt.attempts>=16:
            logits=logits.clone();logits[1:]=-torch.inf
        action=int(logits.argmax())
        if action:
            self.learned_call(Port(action))
            return None
        scores,ids=self.backbone.ordinary_logits(self.hidden[0,-1],rt.phase)
        token=int(ids[scores.argmax()])
        if token==248069:self.close()
        elif token==248046:
            rt.debit('H',1,'chat_end');self.step(self.backbone.embedding([token])[0],'R','chat_end',False)
            rt.terminate('Completed',emit_end=False)
        else:self.native([token])
        return token

    def summary(self):
        return {**self.runtime.summary(),'prefills':self.prefills,'lineage':self.state.lineage,
                'switches':self.switches,'version_checks':self.version_checks,'boundaries':self.boundaries,
                'predicted_budget':self.predicted_budget,'granted_budget':self.runtime.budget,'override':self.override}
