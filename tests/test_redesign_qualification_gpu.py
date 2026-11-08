"""RD-07 changed-path numeric/cancellation/1024 qualification, no updates."""
import gc
import json
import os
from pathlib import Path
import unittest

GPU_CHECKS={
    'RD07-GPU-PROJECTION-FP64':'test_projection_forward_and_VJP_FP64_reference',
    'RD07-GPU-LONG-PARITY':'test_direct_1024_functional_checkpoint_serving_FP32',
    'RD07-GPU-CANCELLATION-ENVELOPES':'test_cancellation_and_exact_envelopes',
}


def record(value):
    print(json.dumps(value,allow_nan=False),flush=True)
    if os.environ.get('QOT_RD07_EVIDENCE_DIR'):
        with (Path(os.environ['QOT_RD07_EVIDENCE_DIR'])/'metrics.jsonl').open('a') as stream:
            stream.write(json.dumps(value,allow_nan=False)+'\n')


class RedesignQualificationGPU(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if os.environ.get('QOT_RD07_GPU_QUALIFY')!='1':raise unittest.SkipTest('requires submitted RD-07')
        from qwen_one_two.g1_device import PRIMARY
        if os.environ.get('CUDA_VISIBLE_DEVICES')!=PRIMARY:raise ValueError('exact 3090 UUID required before imports')
        from tests.test_direct_roles_gpu import DirectRolesGPU
        DirectRolesGPU.setUpClass.__func__(cls)
        cls.c.enable_direct_roles()

    def vectors(self,n):
        from tests.test_direct_roles_gpu import DirectRolesGPU
        return DirectRolesGPU.vectors(self,n)

    def packet(self,rt,port):
        from tests.test_direct_roles_gpu import DirectRolesGPU
        return DirectRolesGPU.packet(self,rt,port)

    def ids(self):
        return self.b.tokenizer.encode(self.b.tokenizer.apply_chat_template([{'role':'user','content':self.prompt}],
            tokenize=False,add_generation_prompt=True,enable_thinking=True),add_special_tokens=False)

    def test_projection_forward_and_VJP_FP64_reference(self):
        import torch
        from torch.nn import functional as F
        from qwen_one_two.g1_checks import tensor_error,grad_error
        module=self.c.direct_role_projection
        saved=[p.detach().clone() for p in module.parameters()]
        try:
            # Deterministic nonzero engineering delta; never an optimizer step.
            with torch.no_grad():
                module.weight.copy_((torch.arange(1024*1024,device='cuda').reshape(1024,1024)%17-8).float()*1e-5)
                module.bias.copy_(torch.linspace(-.001,.001,1024,device='cuda'))
            x=self.vectors(32).requires_grad_();direction=torch.linspace(-1,1,x.numel(),device='cuda').reshape_as(x)
            actual=module(x);grads=torch.autograd.grad((actual*direction).sum(),(x,module.weight,module.bias))
            reference_x=x.detach().double().requires_grad_();w=module.weight.detach().double().requires_grad_();bias=module.bias.detach().double().requires_grad_()
            reference=reference_x+F.linear(reference_x,w,bias)
            expected=torch.autograd.grad((reference*direction.double()).sum(),(reference_x,w,bias))
            forward=tensor_error(actual,reference,1e-6,1e-5,1e-5)
            gradients={name:grad_error((a,),(b,),1e-4,.99999) for name,a,b in zip(('input','weight','bias'),grads,expected)}
            record({'check':'RD07-GPU-PROJECTION-FP64','forward':forward,'gradients':gradients,'optimizer_updates':0})
            self.assertTrue(forward['pass']);self.assertTrue(all(g['pass'] for g in gradients.values()))
        finally:
            with torch.no_grad():
                for p,value in zip(module.parameters(),saved):p.copy_(value)

    def trace(self,port,profile):
        import torch
        from qwen_one_two.codec.session import Session
        from qwen_one_two.codec.schema import Status,Port
        from qwen_one_two.codec.layout import DIRECT
        self.b.checkpointing=profile=='checkpoint'
        self.b.set_precision(torch.float32 if profile=='fp32' else torch.bfloat16)
        self.b.recurrence_backend='eager' if profile=='fp32' else 'graph-v1'
        with torch.set_grad_enabled(profile!='serving'):
            s=Session.open(self.b,self.c,self.ids(),1024,override=512,profile=DIRECT,run='RD07-paired-trace')
            rt=s.runtime;rt.begin(port);packet=self.packet(rt,port);wire,evaluated,_=s.evaluate(packet)
            scalar=None if port==Port.CHOICE else evaluated['value']
            returned=rt.prepare_result(packet.snapshot,Status.ASSESSED,scalar=scalar,
                                       selected_index=0 if port==Port.CHOICE else None)
            self.assertTrue(s.deliver(returned));s.close()
            token=self.b.tokenizer.encode(self.prompt,add_special_tokens=False)[0]
            # Explicit prescribed existing-token tail, not sampled/new data.
            remaining=1023-rt.position
            self.assertGreaterEqual(remaining,0)
            while remaining:
                count=min(32,remaining)
                for _ in range(count):rt.native()
                s.step(self.b.embedding([token]*count)[0],'R','qualification_tail',False)
                remaining-=count
            logits,allowed=self.b.ordinary_logits(s.hidden[0,-1],1)
            target=(allowed==token).nonzero().flatten()
            native=torch.nn.functional.cross_entropy(logits[None],target)
            signal=evaluated['pointer_logits'][0] if port==Port.CHOICE else evaluated['raw']
            loss=evaluated['status_logits'][0]+.1*signal+.1*native
            vectors={}
            if profile!='serving':
                groups={'D':list(self.b.adapter_parameters('D')),'projection':list(self.c.direct_role_projection.parameters())}
                params=[p for values in groups.values() for p in values]
                gradients=torch.autograd.grad(loss,params,allow_unused=True)
                offset=0
                for name,values in groups.items():vectors[name]=tuple(g.detach() if g is not None else None for g in gradients[offset:offset+len(values)]);offset+=len(values)
            rt.debit('H',1,'chat_end');s.step(self.b.embedding([248046])[0],'R','end',True);rt.terminate('Completed',emit_end=False)
            self.assertEqual(rt.position,1024);self.assertEqual(rt.position,s.state.position)
            self.assertEqual(s.prefills,1);self.assertEqual(s.state.lineage,1);self.assertEqual(s.summary()['held_positions'],0)
            call={'status':int(Status.ASSESSED),'predicted_status':int(evaluated['status_logits'].argmax())+1,
                  'q':len(packet),'r':len(returned),'handles':len(returned.handles),'selected':0 if port==Port.CHOICE else None,
                  'scalar':None if scalar is None else float(scalar.detach())}
            result={'summary':s.summary(),'snapshots':s.snapshots,'vectors':vectors,'logits':[logits.detach()],
                    'calls':[call],'budget_raw':float(s.raw_budget.detach())}
        record({'event':'long_trace','port':int(port),'profile':profile,'positions':rt.position,'q':len(packet),'r':len(returned),
                'prefills':s.prefills,'lineage':s.state.lineage,'optimizer_updates':0,'forcing':'existing prompt/token, direct engineering role states, Assessed, Choice index0, budget512, 1024 total'})
        return result

    def test_direct_1024_functional_checkpoint_serving_FP32(self):
        import torch
        from qwen_one_two.codec.qualify import compare
        from qwen_one_two.codec.schema import Port
        original_R=[p.requires_grad for p in self.b.adapter_parameters('R')]
        backend=self.b.recurrence_backend
        # Keep FP32 adapter masters out of the base-only precision conversion.
        for p in self.b.adapter_parameters('R'):p.requires_grad_(True)
        try:
            for port in (Port.NOUL,Port.CHOICE,Port.SCORE):
                base=self.trace(port,'functional')
                for profile in ('checkpoint','serving','fp32'):
                    test=self.trace(port,profile);metrics=compare(test,base,profile=='fp32')
                    record({'check':'RD07-GPU-LONG-PARITY','port':int(port),'profile':profile,'comparison':metrics})
                    self.assertTrue(metrics['pass'],metrics)
                    del test;gc.collect();torch.cuda.empty_cache()
                del base;gc.collect();torch.cuda.empty_cache()
        finally:
            self.b.set_precision(torch.bfloat16);self.b.checkpointing=False;self.b.recurrence_backend=backend
            for p,value in zip(self.b.adapter_parameters('R'),original_R):p.requires_grad_(value)

    def test_cancellation_and_exact_envelopes(self):
        import torch
        from qwen_one_two.codec.schema import Port,Kind,Status,Invalid
        from qwen_one_two.codec.records import Semantic,Source,Span,Reason
        from qwen_one_two.codec.runtime import Runtime
        from qwen_one_two.codec.session import Session
        from qwen_one_two.codec.layout import DIRECT
        with torch.no_grad():
            for where in ('rpc_before_request','callback_before_request','callback_in_request','callback_before_return'):
                with self.subTest(where=where):
                    s=Session.open(self.b,self.c,self.ids(),64,override=512,profile=DIRECT);rt=s.runtime;rt.begin(Port.NOUL)
                    packet=self.packet(rt,Port.NOUL);before=rt.position
                    if where=='rpc_before_request':s.native_step(cancelled=True)
                    elif where=='callback_before_request':
                        s.cancelled=lambda:True
                        with self.assertRaises(InterruptedError):s.evaluate(packet)
                    elif where=='callback_in_request':
                        s.cancelled=lambda:s.state.position>before
                        with self.assertRaises(InterruptedError):s.evaluate(packet)
                    else:
                        _,heads,_=s.evaluate(packet)
                        result=rt.prepare_result(packet.snapshot,Status.ASSESSED,scalar=heads['value'])
                        before=rt.position;s.cancelled=lambda:True
                        with self.assertRaises(InterruptedError):s.deliver(result)
                    self.assertEqual(rt.terminal,'Cancelled');self.assertEqual(rt.position,s.state.position)
                    self.assertEqual(rt.summary()['held_positions'],0)
                    self.assertIsNone(rt.prepare_result(packet.snapshot,Status.ASSESSED,scalar=torch.tensor(.5,device='cuda')))
                    record({'check':'RD07-GPU-CANCELLATION-ENVELOPES','cancellation':where,'summary':s.summary()})
            # Exact256 request and one position over, using explicit engineering spans.
            for extra in (0,1):
                rt=Runtime('RD07-q-boundary',16,1024,64,profile=DIRECT);rt.startup()
                body=self.prompt.encode();src=rt.registry.publish(Kind.SOURCE,Source(body,tuple(range(len(body))),tuple((i,i+1) for i in range(len(body))),self.vectors(len(body))))
                rt.begin(Port.CHOICE)
                q=rt.publish(Kind.QUESTION,Semantic(self.vectors(1),profile=DIRECT,lengths=(1,)))
                t=rt.publish(Kind.TARGET,Semantic(self.vectors(1),profile=DIRECT,lengths=(1,)))
                cs=tuple(rt.publish(Kind.ALTERNATIVE,Semantic(self.vectors(14),profile=DIRECT,lengths=(14,))) for _ in range(16))
                refs=rt.registry.refs()
                if extra:
                    with self.assertRaises(Invalid):rt.seal(q,t,(Span(src,0,11),),candidates=cs)
                    self.assertEqual(rt.registry.refs(),refs);self.assertEqual(rt.dispatched,0)
                else:
                    packet=rt.seal(q,t,(Span(src,0,10),),candidates=cs)
                    self.assertEqual(len(packet),256);self.assertEqual(len(self.c.encode(packet)),256)
                rt.terminate('Cancelled',emit_end=False)
            rt=Runtime('RD07-r-boundary',16,1024,64,profile=DIRECT);rt.startup()
            body=self.prompt.encode();src=rt.registry.publish(Kind.SOURCE,Source(body,tuple(range(len(body))),tuple((i,i+1) for i in range(len(body))),self.vectors(len(body))))
            rt.begin(Port.NOUL)
            q=rt.publish(Kind.QUESTION,Semantic(self.vectors(23),profile=DIRECT,lengths=(23,)))
            t=rt.publish(Kind.TARGET,Semantic(self.vectors(1),profile=DIRECT,lengths=(1,)))
            packet=rt.seal(q,t);rt.semantic_positions(len(packet))
            returned=rt.prepare_result(packet.snapshot,Status.INSUFFICIENT,reason=Reason(self.vectors(4),(Span(src,0,31),),DIRECT))
            self.assertEqual(len(returned),64);self.assertEqual(len(self.c.encode(returned)),64)
            rt.semantic_positions(64,True);self.assertTrue(rt.consumed(packet.snapshot));self.assertEqual(rt.summary()['held_positions'],0)
            record({'check':'RD07-GPU-CANCELLATION-ENVELOPES','exact_request':256,'rejected_request':257,'exact_return':64})


if __name__=='__main__':unittest.main()
