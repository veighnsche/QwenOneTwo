"""Full-codec CUDA qualification. Never fall back to CPU model execution."""
import os
import unittest
from dataclasses import replace
from pathlib import Path
import torch
import json
from torch.nn.attention import sdpa_kernel,SDPBackend
from safetensors.torch import load_file
from qwen_one_two.g1_proof import configure
from qwen_one_two.g1_backbone import Backbone,GPU_UUID
from qwen_one_two.codec.schema import *
from qwen_one_two.codec.records import *
from qwen_one_two.codec.runtime import Runtime,NotAdmitted
from qwen_one_two.codec.render import Renderer
from qwen_one_two.codec.neural import Codec,norm,Block,ReferencePooler,thinking_budget
from qwen_one_two.codec.session import Session
from qwen_one_two.codec.ownership import parameter_groups
from qwen_one_two.codec.fixtures import slots,source,base,reference_boundary
from qwen_one_two.codec.backbone import Backbone as ExpandedBackbone
from qwen_one_two.codec.precision import linear as three_part_linear


@unittest.skipUnless(os.environ.get('CUDA_VISIBLE_DEVICES')==GPU_UUID,'explicit RTX 3090 required')
class FullCodecGPU(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        configure();cls.math=sdpa_kernel(SDPBackend.MATH);cls.math.__enter__()
        snapshot=os.environ['QOT_QUALIFY_SNAPSHOT']
        cls.backbone=Backbone(snapshot,'data/qev')
        with torch.device('cuda'):
            cls.codec=Codec(cls.backbone.sigma,cls.backbone.model.model.embed_tokens.weight,
                            load_file('data/qev/pointer.safetensors',device='cuda'))
        cls.backbone.bind_rows(cls.codec);cls.backbone.set_compensated(True)
        cls.vector_dir=Path(os.environ['QOT_CODEC_VECTOR_OUTPUT']) if os.environ.get('QOT_CODEC_VECTOR_OUTPUT') else None
        if cls.vector_dir:
            from safetensors.torch import save_file
            cls.vector_dir.mkdir(parents=True,exist_ok=False)
            save_file({k:v.contiguous() for k,v in cls.codec.state_dict().items()},str(cls.vector_dir/'encoder-state.safetensors'))

    def persist(self,name,packet):
        if self.vector_dir is None:return
        from safetensors.torch import save_file
        tensors={'wire':self.codec.encode(packet).detach().contiguous()};rows=[]
        for i,p in enumerate(packet.positions):
            row={'role':int(p.role),'owner':p.owner,'slot':p.slot,'attribute':p.attribute}
            if isinstance(p.payload,Metadata):row['bits']=p.payload.encode(p.role)
            elif isinstance(p.payload,Atom):
                row.update(atom_hex=p.payload.data.hex(),coefficient=p.payload.coefficient,exponent=p.payload.exponent)
            else:tensors[f'payload_{i}']=p.payload.detach().contiguous()
            rows.append(row)
        save_file(tensors,str(self.vector_dir/(name+'.safetensors')))
        (self.vector_dir/(name+'.json')).write_text(json.dumps({'scope':'untrained engineering wire vectors, not semantic labels',
             'positions':rows,'handles':[{'run':r.run,'kind':int(r.kind),'id':r.identity,'version':r.version} for r in packet.handles]},indent=2))

    @classmethod
    def tearDownClass(cls):
        del cls.codec,cls.backbone;cls.math.__exit__(None,None,None);torch.cuda.empty_cache()

    def rt(self,name='fixture',budget=2048):
        r=Runtime(name,16,budget,32);r.startup();return r

    def test_all_thirteen_cells_presence_lengths_and_encoder_gradients(self):
        seen=set()
        for port in (Port.NOUL,Port.CHOICE,Port.SCORE):
            for status in tuple(Status)[1:]:
                if status==Status.NO_FEASIBLE_CHOICE and port!=Port.CHOICE:continue
                rt=self.rt();packet,*_=base(rt,port)
                rt.semantic_positions(len(packet))
                v=torch.tensor(0.,device='cuda',requires_grad=True)
                reason=None if status==Status.ASSESSED else Reason(slots(4,True))
                result=rt.prepare_result(rt.call.snapshot,status,
                    scalar=v if status==Status.ASSESSED and port!=Port.CHOICE else None,
                    selected_index=1 if status==Status.ASSESSED and port==Port.CHOICE else None,reason=reason)
                expected=(14 if port==Port.NOUL else 25 if port==Port.SCORE else 18) if status==Status.ASSESSED else (29 if port==Port.SCORE else 18)
                self.assertEqual(len(result),expected)
                self.assertEqual(result.positions[0].payload.present,status==Status.ASSESSED)
                self.assertEqual(result.positions[0].payload.children,0)
                values=[p for p in result.positions if p.role in (Role.NOUL,Role.SCORE)]
                self.assertEqual(len(values),int(status==Status.ASSESSED and port!=Port.CHOICE))
                x=self.codec.encode(result);self.assertTrue(bool(x.isfinite().all()))
                self.persist(f'{port.name}_{status.name}',result)
                if values:
                    grad=torch.autograd.grad(x[12].sum() if port==Port.NOUL else x[-2].sum(),v,allow_unused=True)[0]
                    self.assertIsNotNone(grad);self.assertGreater(grad.abs().item(),1e-10)
                rt.semantic_positions(len(result),True);self.assertTrue(rt.consumed(rt.call.snapshot))
                self.assertFalse(rt.consumed(rt.call.snapshot));self.assertEqual(rt.summary()['held_positions'],0)
                seen.add((port,status))
        self.assertEqual(len(seen),13)

    def test_full_typed_roles_absence_empty_zero_and_source_clipping(self):
        rt=self.rt();reg=rt.registry;src=source(reg,b'1.20 \xce\xb1 ready')
        unit=reg.publish(Kind.TARGET,Semantic(slots(2)))
        vals=(Value(ValueKind.BOOLEAN,True,False,Mode.CONSTRUCT),
              Value(ValueKind.INT64,True,-(1<<63),Mode.CONSTRUCT),
              Value(ValueKind.DECIMAL,True,(120,-2),Mode.PARSE,src,parsed_span=Span(src,0,4)),
              Value(ValueKind.REFERENCE,True,unit,Mode.CONSTRUCT),
              Value(ValueKind.SOURCE_STRING,True,Span(src,5,2),Mode.CONSTRUCT,src),
              Value(ValueKind.UTF8,True,b'',Mode.CONSTRUCT),
              Value(ValueKind.BOOLEAN,False),Value(ValueKind.UTF8,True,b'x'*128,Mode.CONSTRUCT))
        attrs=tuple(Attribute(slots(2),v) for v in vals)
        packet,*_=base(rt,Port.SCORE,attrs)
        roles={p.role for p in packet.positions}
        self.assertTrue({Role.BOOLEAN,Role.INT64,Role.COEFFICIENT,Role.EXPONENT,Role.REFERENCE,Role.ATTRIBUTE_SOURCE,Role.UTF8}<=roles)
        desc=[p for p in packet.positions if p.role==Role.ATTRIBUTE_DESC]
        self.assertEqual((desc[5].payload.present,desc[5].payload.following),(True,2))
        self.assertEqual((desc[6].payload.present,desc[6].payload.following),(False,2))
        self.assertEqual(desc[2].payload.mode,Mode.PARSE)
        self.assertTrue(torch.equal(self.codec.atom_features(encode_atoms(ValueKind.INT64,-(1<<63))[0])[-3:],
                                    torch.tensor([-1.,1.,0.],device='cuda')))
        features=self.codec.atom_features(encode_atoms(ValueKind.DECIMAL,(120,-2))[1])
        self.assertEqual(features[-2].item(),63/512)
        self.assertEqual(features[-1].item(),torch.tensor(-1/9,device='cuda').item())
        self.assertTrue(bool(self.codec.encode(packet).isfinite().all()))
        self.persist('all_typed_request',packet)
        with self.assertRaises(Invalid):reg.span(Span(src,5,1,0,1))

    def test_immutable_versions_historical_scalar_connection_and_serial_accounting(self):
        rt=self.rt();p,q,t,*_=base(rt,Port.NOUL,evidence=False)
        old=rt.registry.get(q);new=rt.registry.publish(Kind.QUESTION,Semantic(slots(8)),previous=q)
        self.assertEqual(new.identity,q.identity);self.assertEqual(new.version,2)
        with torch.no_grad():old.slots.zero_()
        self.assertGreater(rt.registry.get(q).slots.abs().sum().item(),0)
        v=torch.tensor(.4,device='cuda',requires_grad=True)
        rt.semantic_positions(len(p));result=rt.prepare_result(rt.call.snapshot,Status.ASSESSED,scalar=v)
        first=rt.call.result;rt.semantic_positions(len(result),True);rt.consumed(rt.call.snapshot)
        old_snapshot=rt.call.snapshot
        rt.close();rt.begin(Port.SCORE)
        rubric=rt.publish(Kind.RUBRIC,Semantic(slots(10),direction=2,midpoint=False))
        rt.writer_position();rt.writer_position();packet=rt.seal(new,t,(History(first),),rubric)
        self.assertEqual(rt.prepare_result(old_snapshot,Status.ASSESSED,scalar=v),None)
        self.assertTrue(any(p.role==Role.RETURN_END for p in packet.positions[:-1]))
        middle=[p for p in packet.positions if p.role==Role.MIDPOINT]
        self.assertTrue(all(torch.equal(p.payload,torch.zeros_like(p.payload)) for p in middle))
        encoded=self.codec.encode(packet)
        derivative=torch.autograd.grad(encoded.sum(),v)[0]
        self.assertGreater(derivative.abs().item(),1e-10)
        rt.semantic_positions(len(packet));out=rt.prepare_result(rt.call.snapshot,Status.INSUFFICIENT,reason=Reason(slots(4)))
        rt.semantic_positions(len(out),True);rt.consumed(rt.call.snapshot)
        self.assertEqual(rt.consumptions,2);self.assertEqual(rt.phase,1)
        self.assertEqual(rt.summary()['released_positions'],(50-14)+(61-29))
        self.assertEqual(rt.summary()['held_positions'],0)

    def test_maximum_choice_return_and_all_candidate_reservation(self):
        attrs=tuple(Attribute(slots(2),Value(ValueKind.INT64,True,i,Mode.CONSTRUCT)) for i in range(6))
        attrs+=(Attribute(slots(2),Value(ValueKind.UTF8,True,b'a'*128,Mode.CONSTRUCT)),
                Attribute(slots(2),Value(ValueKind.BOOLEAN,False)))
        self.assertEqual(sum(a.cost for a in attrs),46)
        rt=self.rt();p,*_=base(rt,Port.CHOICE,candidate_attributes=attrs)
        self.assertEqual(Renderer(rt.registry).return_max(rt.call.snapshot),64)
        rt.semantic_positions(len(p));ret=rt.prepare_result(rt.call.snapshot,Status.ASSESSED,selected_index=1)
        self.assertEqual(len(ret),64);self.assertEqual(ret.positions[12].payload.item,1)
        self.assertEqual(len([p for p in ret.positions if p.role==Role.ATTRIBUTE_DESC]),8)
        self.assertTrue(bool(self.codec.encode(ret).isfinite().all()))
        self.persist('choice_return_64',ret)

    def test_request_256_257_and_sixteen_candidates(self):
        for count,expected in ((10,256),(11,257)):
            rt=self.rt();reg=rt.registry;src=source(reg,b'x'*16)
            rt.begin(Port.CHOICE)
            attrs=tuple(Attribute(slots(2),Value(ValueKind.UTF8,True,b'x'*128,Mode.CONSTRUCT)) for _ in range(8))
            q=rt.publish(Kind.QUESTION,Semantic(slots(8)));t=rt.publish(Kind.TARGET,Semantic(slots(2),attrs))
            cs=tuple(rt.publish(Kind.ALTERNATIVE,Semantic(slots(4))) for _ in range(16))
            rt.writer_position();rt.writer_position();before=reg.refs()
            if expected==257:
                with self.assertRaises(Invalid):rt.seal(q,t,(Span(src,0,count),),candidates=cs)
                self.assertEqual(reg.refs(),before)
            else:
                p=rt.seal(q,t,(Span(src,0,count),),candidates=cs)
                self.assertEqual(len(p),256);self.assertEqual(len(p.candidate_ends),16)
                self.assertTrue(bool(self.codec.encode(p).isfinite().all()))
                self.persist('choice_request_256',p)

    def test_simultaneous_255_positions_256_handles_and_completion_records(self):
        rt=self.rt();p,attrs=reference_boundary(rt)
        self.assertEqual((len(p),len(p.handles),rt.call.position_hold,rt.call.handle_hold),(255,222,305,34))
        self.assertEqual(rt.call.allocated,10)
        rt.semantic_positions(len(p))
        result=rt.prepare_result(rt.call.snapshot,Status.NO_FEASIBLE_CHOICE,reason=Reason(slots(4),attrs))
        self.assertEqual((len(result),len(result.handles),rt.call.allocated),(50,256,12))
        self.assertEqual((rt.call.handle_hold,rt.call.record_hold),(0,0))
        self.assertTrue(bool(self.codec.encode(p).isfinite().all()))
        self.assertTrue(bool(self.codec.encode(result).isfinite().all()))
        self.persist('request_255_handles_222',p);self.persist('return_handles_256',result)
        rt.semantic_positions(len(result),True);rt.consumed(rt.call.snapshot)
        self.assertEqual(rt.summary()['held_positions'],0)

    def test_floor_branch_empty_memory_and_parallel_head_shapes(self):
        z=torch.zeros(256,device='cuda',requires_grad=True)
        g=torch.autograd.grad(norm(z).sum(),z)[0]
        self.assertTrue(bool(g.isfinite().all()));self.assertEqual(g[0].item(),1e6)
        self.assertEqual(sum(p.numel() for p in self.codec.reader.parameters()),788736)
        f=torch.randn(256,device='cuda',requires_grad=True)
        empty=self.codec.reader(f,f.new_empty(0,256))
        self.assertTrue(bool(empty.isfinite().all()))
        self.assertEqual(self.codec.byte_logits(f,128).shape,(128,256))
        for name,head in self.codec.heads.items():self.assertTrue(bool(head(f).isfinite().all()),name)
        self.assertEqual(len(self.codec.pointers),9)
        token=self.codec.pointer_key(torch.randn(1024,device='cuda'))
        self.assertEqual(self.codec.clipping_keys(token,[0,128]).shape,(2,256))
        score=torch.tensor(.7,device='cuda',requires_grad=True)
        wire=self.codec.encode(self.codec.startup(score))
        self.assertIsNone(torch.autograd.grad(wire.sum(),score,allow_unused=True)[0])

    def test_invalid_returns_cancellation_and_duplicate_session_delivery(self):
        rt=self.rt();p,*_=base(rt,Port.NOUL,evidence=False)
        rt.semantic_positions(len(p));before=rt.registry.refs();position=rt.position
        bad=[dict(status=Status.NO_FEASIBLE_CHOICE,reason=Reason(slots(4))),
             dict(status=Status.ASSESSED,scalar=torch.tensor(float('nan'),device='cuda')),
             dict(status=Status.INSUFFICIENT,scalar=torch.tensor(0.,device='cuda'),reason=Reason(slots(4))),
             dict(status=Status.UNRESOLVED,reason=Reason(slots(4),tuple(Attribute(slots(2),Value(ValueKind.UTF8,True,b'x'*128,Mode.CONSTRUCT)) for _ in range(2))))]
        for kwargs in bad:
            with self.assertRaises(Invalid):rt.prepare_result(rt.call.snapshot,**kwargs)
            self.assertEqual(rt.registry.refs(),before)
            self.assertEqual(rt.position,position)
        snapshot=rt.call.snapshot
        rt.terminate('Cancelled',emit_end=False)
        self.assertIsNone(rt.prepare_result(snapshot,Status.ASSESSED,scalar=torch.tensor(.5,device='cuda')))
        self.assertEqual(rt.position,position);self.assertEqual(rt.summary()['held_positions'],0)
        rt2=self.rt();p,*_=base(rt2,Port.NOUL,evidence=False)
        rt2.semantic_positions(len(p));out=rt2.prepare_result(rt2.call.snapshot,Status.ASSESSED,scalar=torch.tensor(.5,device='cuda'))
        rt2.semantic_positions(len(out),True);rt2.consumed(rt2.call.snapshot)
        session=Session.__new__(Session);session.runtime=rt2
        old=rt2.summary();self.assertFalse(session.deliver(out));self.assertEqual(old,rt2.summary())

    def test_reserved_record_slots_exact_fit_and_nonadmission(self):
        for port,cap in ((Port.NOUL,29),(Port.CHOICE,28),(Port.SCORE,29)):
            rt=self.rt();rt.begin(port)
            for _ in range(cap):rt.publish(Kind.QUESTION,Semantic(slots(8)))
            self.assertEqual(rt.call.allocated,cap)
            with self.assertRaises(Invalid):rt.publish(Kind.QUESTION,Semantic(slots(8)))
            rt.end_attempt('ObjectLimit',7)
            self.assertEqual(rt.call.released_records,2+(2 if port==Port.CHOICE else 1))
        for b,admitted in ((65,True),(64,False)):
            rt=self.rt(budget=b)
            if admitted:
                p,*_=base(rt,Port.NOUL,evidence=False);self.assertEqual(rt.call.position_hold,63)
            else:
                with self.assertRaises(NotAdmitted):base(rt,Port.NOUL,evidence=False)
                self.assertEqual((rt.dispatched,rt.call.allocated,rt.call.outcome),(0,3,3))
                self.assertIsNotNone(rt.call.snapshot)
                self.assertFalse(any(r.kind==Kind.RESULT for r in rt.registry.refs()))

    def test_reference_module_ownership_and_save_load_without_updates(self):
        import tempfile
        from safetensors.torch import save_file
        with torch.device('cuda'):pool=ReferencePooler()
        self.assertEqual(sum(p.numel() for p in pool.parameters()),2110720)
        pooled=pool(slots(8,True),0,8);self.assertEqual(pooled.shape,(8,1024))
        groups=parameter_groups(self.backbone,self.codec)
        identities=[id(p) for g in groups for _,p in g['named_parameters']]
        self.assertEqual(len(identities),len(set(identities)))
        rt=self.rt();p,*_=base(rt,Port.SCORE)
        baseline=self.codec.encode(p)
        with tempfile.TemporaryDirectory(prefix='qwenonetwo-codec-') as directory:
            path=str(Path(directory)/'codec.safetensors')
            save_file({k:v.contiguous() for k,v in self.codec.state_dict().items()},path)
            state=load_file(path,device='cuda')
            actual=torch.func.functional_call(self.codec,state,(p,))
            self.assertTrue(torch.equal(actual,baseline))
            hidden=torch.randn(1024,device='cuda',requires_grad=True)
            logits,indices=self.backbone.ordinary_logits(hidden,1)
            self.assertFalse(bool(((indices==248077)|(indices==248078)).any()))
            blocked=torch.autograd.grad(logits[0],(self.codec.budget_row,self.codec.score_row),allow_unused=True)
            self.assertTrue(all(g is None for g in blocked))
        self.assertTrue(all(p.grad is None for p in self.codec.parameters()))

    def test_temporal_rematerialization_matches_original_layer_and_gradients(self):
        index=next(i for i,x in enumerate(self.backbone.model.model.layers) if x.block_type=='linear_attention')
        a=self.backbone.model.model.layers[index].linear_attn
        state=self.backbone.initial().layers[index]
        x=(torch.randn(1,12,1024,device='cuda')*.1).requires_grad_()
        self.backbone.temporal_chunk=4
        old,old_state=Backbone._linear_attention(self.backbone,a,x,state)
        new,new_state=ExpandedBackbone._linear_attention(self.backbone,a,x,state)
        self.assertTrue(torch.equal(old,new))
        self.assertTrue(all(torch.equal(a,b) for a,b in zip(old_state,new_state)))
        target=torch.randn_like(old)
        gx=torch.autograd.grad((old*target).sum(),x,retain_graph=True)[0]
        gy=torch.autograd.grad((new*target).sum(),x)[0]
        self.assertTrue(torch.equal(gx,gy))

    def test_three_part_bf16_retains_third_input_residual(self):
        x=torch.tensor([[1.,1.+2**-10+2**-20]],device='cuda',requires_grad=True)
        w=torch.tensor([[1.,-1.]],device='cuda',dtype=torch.bfloat16)
        y=three_part_linear(x,w)
        reference=torch.nn.functional.linear(x,w.float())
        self.assertTrue(torch.equal(y,reference))
        self.assertTrue(torch.equal(torch.autograd.grad(y.sum(),x)[0],w.float()))

    def test_zero_budget_factory_and_direct_native_helper_preserve_one_cache(self):
        for score,expected in ((-1.,0),(0.,0),(.25,20480),(.5,65536),(.75,110592),(1.,131072),(2.,131072)):
            self.assertEqual(int(thinking_budget(torch.tensor(score,device='cuda'))),expected)
        for value in (float('nan'),float('inf')):
            with self.assertRaises(Invalid):thinking_budget(torch.tensor(value,device='cuda'))
        text=self.backbone.tokenizer.apply_chat_template([{'role':'user','content':'Hello.'}],tokenize=False,
                    add_generation_prompt=True,enable_thinking=True)
        ids=self.backbone.tokenizer.encode(text,add_special_tokens=False)
        token=self.backbone.tokenizer.encode('Hello',add_special_tokens=False)[0]
        with torch.no_grad():
            rt=Runtime('zero-cache',len(ids),0,2)
            s=Session(self.backbone,self.codec,rt);s.prefill(ids);s.startup();s.native([token])
            self.assertEqual((rt.b,rt.a,rt.h,rt.phase),(0,1,5,1))
            self.assertEqual(s.state.position,rt.position);self.assertEqual(s.prefills,1)
            before=rt.position;s.native_step(cancelled=True)
            self.assertEqual((rt.position,rt.terminal),(before,'Cancelled'))
            s2=Session.open(self.backbone,self.codec,ids,2,override=3,run='override')
            self.assertEqual((s2.predicted_budget,s2.runtime.budget,s2.override),(0,3,3))
            self.assertEqual(s2.prefills,1)

    def test_return_packet_views_cannot_rewrite_published_feedback(self):
        rt=self.rt();p,*_=base(rt,Port.NOUL,evidence=False)
        rt.semantic_positions(len(p))
        out=rt.prepare_result(rt.call.snapshot,Status.ASSESSED,scalar=torch.tensor(.25,device='cuda'))
        canonical=rt.registry.get(out.result).presentation
        expected=self.codec.encode(canonical)
        with torch.no_grad():out.positions[-2].payload.fill_(.9)
        self.assertEqual(rt.registry.get(out.result).scalar.item(),.25)
        delivered=[]
        s=Session.__new__(Session);s.runtime=rt;s.codec=self.codec
        s.step=lambda embeds,mode,label:delivered.append(embeds)
        self.assertTrue(s.deliver(out))
        self.assertTrue(torch.equal(delivered[0],expected))
        self.assertEqual(rt.consumptions,1)

    def test_unsafe_execution_failure_is_terminal_without_notice(self):
        from unittest.mock import patch
        ids=[1,2,3]
        for bad_position in (False,True):
            rt=Runtime('failed-cache',len(ids),0,2);s=Session(self.backbone,self.codec,rt)
            kwargs={'return_value':(torch.zeros(1,3,1024,device='cuda'),replace(s.state,position=4))} if bad_position else {'side_effect':RuntimeError('injected device failure')}
            with patch.object(self.backbone,'forward',**kwargs):
                with self.assertRaises(RuntimeError):s.prefill(ids)
            self.assertEqual((rt.terminal,rt.h,rt.summary()['held_positions']),('Failed',0,0))
            self.assertIsNone(s.native_step())


if __name__=='__main__':unittest.main()
