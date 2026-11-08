"""RD-04 engineering assertions pending RD-07; no model imports at collection."""
import os
import unittest


class DirectRolesGPU(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if os.environ.get('QOT_RD04_GPU_QUALIFY')!='1':
            raise unittest.SkipTest('direct role assertions require submitted RD-07')
        from qwen_one_two.g1_device import PRIMARY
        if os.environ.get('CUDA_VISIBLE_DEVICES')!=PRIMARY:raise ValueError('exact 3090 UUID required before imports')
        import torch
        from torch.nn.attention import sdpa_kernel,SDPBackend
        from qwen_one_two.g1_proof import configure
        from qwen_one_two.codec.qualify import load
        from qwen_one_two.training.reference import Reference
        from qwen_one_two.training.receiver_learning import groups
        from qwen_one_two.training.checkpoint import restore_model
        from qwen_one_two.training.receiver_plan import initial_state,TRAIN_SHA
        from qwen_one_two.training.release import load_release
        from qwen_one_two.training.two_gpu import ROOT,CHECKPOINT
        configure();cls.math=sdpa_kernel(SDPBackend.MATH);cls.math.__enter__();cls.addClassCleanup(cls.math.__exit__,None,None,None)
        cls.b,cls.c,_=load(os.environ['QOT_QUALIFY_SNAPSHOT'],ROOT/'data/g1-proof/gate-repair-v2-20261105/diagnostic.safetensors')
        cls.ref=Reference(cls.b,cls.c,pooler_kind='target-v2');cls.ref.freeze();groups(cls.b,cls.c,cls.ref)
        saved,_=initial_state();restore_model(ROOT/CHECKPOINT,cls.b,cls.c,cls.ref,saved['identities'])
        release=load_release(ROOT/'data/training-collections-20261105/releases/train',expected_hash=TRAIN_SHA)
        cls.prompt=release.record(sorted(release.selected_ids)[0])['human_input']['prompt']
        cls.reference_id=cls.ref.fingerprint()
        from qwen_one_two.codec.runtime import Runtime
        from qwen_one_two.codec.fixtures import base
        from qwen_one_two.codec.schema import Port
        rt=Runtime('rd04-before-enable',16,1024,16);rt.startup()
        cls.legacy_packet,*_=base(rt,Port.NOUL)
        cls.legacy_wire=cls.c.encode(cls.legacy_packet).detach().clone()

    def enable(self):
        if not hasattr(self.c,'direct_role_projection'):self.c.enable_direct_roles()

    def vectors(self,n):
        # Explicit engineering tensors, not annotations, source features or labels.
        import torch
        return (torch.arange(n*1024,device='cuda',dtype=torch.float32).reshape(n,1024)%257)/257+.01

    def packet(self,rt,port):
        from qwen_one_two.codec.records import Semantic,Attribute,Value
        from qwen_one_two.codec.schema import Kind,ValueKind,Mode
        from qwen_one_two.codec.layout import DIRECT
        q=rt.publish(Kind.QUESTION,Semantic(self.vectors(2),profile=DIRECT,lengths=(2,)))
        t=rt.publish(Kind.TARGET,Semantic(self.vectors(3),profile=DIRECT,lengths=(3,)))
        attrs=(Attribute(self.vectors(3),Value(ValueKind.INT64,True,123,Mode.CONSTRUCT),DIRECT),)
        candidates=tuple(rt.publish(Kind.ALTERNATIVE,Semantic(self.vectors(n),attrs,profile=DIRECT,lengths=(n,)))
                         for n in (4,5)) if int(port)==2 else ()
        rubric=rt.publish(Kind.RUBRIC,Semantic(self.vectors(6),direction=1,profile=DIRECT,lengths=(1,2,1,2,0))) if int(port)==3 else None
        return rt.seal(q,t,rubric=rubric,candidates=candidates)

    def test_identity_projection_and_legacy_wire(self):
        import torch
        self.enable()
        self.assertTrue(torch.equal(self.legacy_wire,self.c.encode(self.legacy_packet)))
        x=self.vectors(32);self.assertTrue(torch.equal(self.c.direct_role_projection(x),x))
        self.assertEqual(self.c.direct_role_projection.weight.dtype,torch.float32)
        self.assertEqual(self.c.semantic_count(torch.zeros(256,device='cuda'),__import__('qwen_one_two.codec.schema',fromlist=['Role']).Role.QUESTION),1)

    def test_direct_packets_returns_and_immutable_bindings(self):
        import torch
        from qwen_one_two.codec.runtime import Runtime
        from qwen_one_two.codec.records import Reason
        from qwen_one_two.codec.schema import Port,Status,Role
        from qwen_one_two.codec.layout import DIRECT
        self.enable();seen=set()
        for port in (Port.NOUL,Port.CHOICE,Port.SCORE):
            for status in tuple(Status)[1:]:
                if status==Status.NO_FEASIBLE_CHOICE and port!=Port.CHOICE:continue
                rt=Runtime('rd04-roundtrip',16,1024,16,profile=DIRECT);rt.startup();rt.begin(port)
                packet=self.packet(rt,port);encoded=self.c.encode(packet)
                self.assertEqual(len(encoded),len(packet));self.assertTrue(bool(encoded.isfinite().all()))
                rt.semantic_positions(len(packet))
                result=rt.prepare_result(packet.snapshot,status,
                    scalar=torch.tensor(.25,device='cuda') if status==Status.ASSESSED and port!=Port.CHOICE else None,
                    selected_index=1 if status==Status.ASSESSED and port==Port.CHOICE else None,
                    reason=Reason(self.vectors(4),profile=DIRECT) if status!=Status.ASSESSED else None)
                self.assertLessEqual(len(result),64);self.assertTrue(bool(self.c.encode(result).isfinite().all()))
                self.assertEqual(result.handles[:len(packet.handles)],packet.handles)
                self.assertFalse(any(p.role==Role.MIDPOINT for p in packet.positions))
                snapshot=rt.registry.get(packet.snapshot);external=rt.registry.get(snapshot.question)
                with torch.no_grad():external.slots.add_(1)
                self.assertTrue(torch.equal(rt.registry.get(snapshot.question).slots,self.vectors(2)))
                rt.semantic_positions(len(result),True);self.assertTrue(rt.consumed(packet.snapshot));seen.add((port,status))
        self.assertEqual(len(seen),13)

    def test_live_projection_and_consumer_gradients(self):
        import torch
        from qwen_one_two.codec.session import Session
        from qwen_one_two.codec.schema import Port
        from qwen_one_two.codec.layout import DIRECT
        from qwen_one_two.g1_evidence import frozen_check
        self.enable()
        ids=self.b.tokenizer.encode(self.b.tokenizer.apply_chat_template([{'role':'user','content':self.prompt}],
            tokenize=False,add_generation_prompt=True,enable_thinking=True),add_special_tokens=False)
        s=Session.open(self.b,self.c,ids,16,override=512,profile=DIRECT);s.runtime.begin(Port.NOUL)
        packet=self.packet(s.runtime,Port.NOUL);_,outputs,_=s.evaluate(packet)
        d=list(self.b.adapter_parameters('D'));projection=list(self.c.direct_role_projection.parameters())
        gradients=torch.autograd.grad(outputs['status_logits'][0],d+projection,allow_unused=True)
        self.assertTrue(any(g is not None and g.norm().item()>0 for g in gradients[:len(d)]))
        self.assertTrue(all(g is not None and bool(g.isfinite().all()) and g.norm().item()>0 for g in gradients[len(d):]))
        self.assertEqual(s.summary()['prefills'],1);self.assertEqual(s.summary()['lineage'],1)
        self.assertTrue(frozen_check(self.b,os.environ['QOT_QUALIFY_SNAPSHOT'])['pass'])
        self.assertEqual(self.reference_id,self.ref.fingerprint())
        s.runtime.terminate('Completed',emit_end=False)


if __name__=='__main__':unittest.main()
