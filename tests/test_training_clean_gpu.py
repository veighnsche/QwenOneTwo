"""GPU engineering fixtures only; reference weights remain untrained."""
import os
import unittest
import torch
from torch.nn.attention import sdpa_kernel, SDPBackend
from qwen_one_two.g1_proof import configure
from qwen_one_two.codec.qualify import load
from qwen_one_two.training.reference import Reference
from qwen_one_two.training.clean import CleanEncoder, CleanView
from qwen_one_two.training.inputs import Source, Binding, Node, Provenance, Field, reference_input, digest


class CleanGPU(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        configure()
        cls.math = sdpa_kernel(SDPBackend.MATH); cls.math.__enter__()
        cls.b, cls.c, _ = load(os.environ['QOT_QUALIFY_SNAPSHOT'], 'data/g1-proof/gate-repair-v2-20261105/diagnostic.safetensors')
        cls.ref = Reference(cls.b, cls.c)
        # Hash actual untrained reference tensors; this does not grant semantic approval.
        cls.ref.freeze()

    @classmethod
    def tearDownClass(cls):
        del cls.ref, cls.c, cls.b
        cls.math.__exit__(None,None,None); torch.cuda.empty_cache()

    def make(self, hidden):
        text = b'Orientation completed.' + hidden
        source = Source('parent',1,text,digest(text))
        binding = Binding('parent',1,0,len(b'Orientation completed.'))
        question = 'Does the participant satisfy the requirements?'
        p = Provenance({'fact':Node('source_fact',digest(b'Orientation completed.'),binding=binding),
                        'q':Node('request_semantics',digest(question.encode()))}, {'parent':source})
        ref = reference_input('question',{'proposition':Field(question,'q')},p)
        return CleanView((ref,),(binding,),frozenset({'fact','q'})), p

    def test_clean_twins_reencode_and_have_local_registry(self):
        from pathlib import Path
        from qwen_one_two.training.manifest import DIAGNOSTIC_SHA
        encoder = CleanEncoder(self.b,self.ref,checkpoint_hash=DIAGNOSTIC_SHA,
                               profile_hash=digest(Path('docs/codec-feedback-precision-v3.md').read_bytes()))
        view,p = self.make(b' Hidden: active.')
        a = encoder.encode(view,p,run='twin-a')
        view,p = self.make(b' Hidden: expired and more text.')
        b = encoder.encode(view,p,run='twin-b')
        self.assertEqual(a['input_hash'],b['input_hash'])
        self.assertEqual([{k:v for k,v in x.items() if k != 'parent'} for x in a['audit']],
                         [{k:v for k,v in x.items() if k != 'parent'} for x in b['audit']])
        self.assertNotEqual(a['audit'][0]['parent']['sha256'],b['audit'][0]['parent']['sha256'])
        self.assertTrue(torch.equal(a['slots'][0],b['slots'][0]))
        sa=a['registry'].get(a['sources'][0]); sb=b['registry'].get(b['sources'][0])
        self.assertEqual(sa.text,b'Orientation completed.')
        self.assertTrue(torch.equal(sa.states,sb.states))
        self.assertEqual(len(a['registry'].refs()),1)
        with self.assertRaises(ValueError):a['registry'].get(b['sources'][0])
        view,p = self.make(b'')
        changed=Source('parent',1,b'Orientation incomplete.',digest(b'Orientation incomplete.'))
        binding=Binding('parent',1,0,len(changed.content))
        p=Provenance({'fact':Node('source_fact',changed.sha256,binding=binding)}, {'parent':changed})
        c=encoder.encode(CleanView((),(binding,),frozenset({'fact'})),p,run='visible-change')
        self.assertNotEqual(a['input_hash'],c['input_hash'])
        self.assertNotEqual(a['audit'][0]['tensor_hash'],c['audit'][0]['tensor_hash'])

    def test_frozen_reconstruction_passes_input_gradient_without_live_adapter_gradient(self):
        prefix=torch.randn(2,1024,device='cuda',requires_grad=True)
        before=[p.detach().clone() for p in self.b.adapter_parameters('R')]
        loss=self.ref.reconstruct(prefix,[self.b.tokenizer.encode('Done',add_special_tokens=False)[0],248046])
        params=list(self.b.adapter_parameters('R'))+list(self.b.adapter_parameters('D'))
        grads=torch.autograd.grad(loss,[prefix]+params,allow_unused=True)
        self.assertTrue(bool(grads[0].isfinite().all()));self.assertGreater(grads[0].norm().item(),0)
        self.assertTrue(all(g is None for g in grads[1:]))
        self.assertTrue(all(torch.equal(p,q) for p,q in zip(self.b.adapter_parameters('R'),before)))
        self.ref.assert_isolated(self.c)

    def test_full_bundle_export_preserves_rows_tying_and_forward(self):
        from pathlib import Path
        import tempfile
        from qwen_one_two.training.engine import Window, live_groups
        from qwen_one_two.training.checkpoint import bundle_tensors, save, restore_model, IDENTITIES
        window=Window(live_groups(self.b,self.c,self.ref))
        weights=bundle_tensors(self.b,self.c,self.ref)
        identities={k:digest(k.encode()) for k in IDENTITIES}
        identities['reference']=self.ref.artifact_id
        ids=self.b.tokenizer.encode('A short engineering input.',add_special_tokens=False)
        with torch.no_grad():
            hidden,_=self.b(self.b.initial(),'R',self.b.embedding(ids))
            scalar=self.b.table.scalar(248077,hidden[0,-1]).clone()
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'bundle'
            save(path,window,weights,identities,{'cursor':0},reference_frozen=True,reference_id=self.ref.artifact_id)
            with torch.no_grad():
                self.c.budget_row.add_(1)
                next(iter(self.b.adapter_parameters('R'))).add_(.01)
            restore_model(path,self.b,self.c,self.ref,identities)
            with torch.no_grad():
                restored,_=self.b(self.b.initial(),'R',self.b.embedding(ids))
                raw=self.b.table.scalar(248077,restored[0,-1])
            self.assertTrue(torch.equal(hidden,restored));self.assertTrue(torch.equal(scalar,raw))
            self.assertEqual(self.b.model.lm_head.weight.data_ptr(),self.b.model.model.embed_tokens.weight.data_ptr())
            self.ref.assert_isolated(self.c)
            self.assertEqual(window.optimizer_updates,0)

    def test_complete_session_accumulates_once_with_no_scalar_row_gradient(self):
        from qwen_one_two.codec.session import Session
        from qwen_one_two.training.engine import Window, live_groups
        from qwen_one_two.training.episodes import collect_session
        from qwen_one_two.training.inputs import Exposure
        ids=self.b.tokenizer.encode('Continue normally.',add_special_tokens=False)
        session=Session.open(self.b,self.c,ids,answer=4,override=512,run='training-guard-fixture')
        session.close()
        token=self.b.tokenizer.encode('Done',add_special_tokens=False)[0]
        losses,_=session.native([token],True)
        session.runtime.debit('H',1,'chat_end')
        session.step(self.b.embedding([248046])[0],'R','end')
        session.runtime.terminate('Completed',emit_end=False)
        window=Window(live_groups(self.b,self.c,self.ref))
        report=collect_session(window,session,{'native':losses[0]},Exposure('native_writer',native_teacher_forcing=True,budget_override=512))
        self.assertTrue(report['accepted']);self.assertEqual(window.examples,1)
        self.assertEqual(window.counts['native'],1)
        active={window.names[i] for i in window.buffers['native']}
        self.assertTrue(any(n.endswith('.R_B') for n in active))
        self.assertTrue(any(n.endswith('.D_B') for n in active))
        self.assertNotIn('codec.budget_row',active);self.assertNotIn('codec.score_row',active)
        self.assertEqual(window.optimizer_updates,0)
        window.abort()

    def test_reference_objective_has_independent_pooler_codec_and_T_gradients(self):
        from qwen_one_two.training.reference import reference_objective
        from qwen_one_two.training.engine import reference_groups, ownership_manifest
        reference=Reference(self.b,self.c)
        manifest=ownership_manifest(reference_groups(reference))
        queries=[entry for group in manifest if group['name']=='reference_other'
                 for entry in group['parameters'] if entry['name']=='pooler.queries']
        self.assertEqual(len(queries),1)
        text='Property meaning'
        provenance=Provenance({'key':Node('request_semantics',digest(text.encode()))},{})
        annotation=reference_input('key',{'property':Field(text,'key')},provenance)
        self.b.checkpointing=True
        try:
            losses,payload=reference_objective(reference,annotation,provenance)
            selected=[reference.pooler.queries,reference.codec.tags.weight,reference.T[1]]
            grads=torch.autograd.grad(losses['reconstruction'],selected)
        finally:
            self.b.checkpointing=False
        self.assertEqual(tuple(payload.shape),(2,1024))
        self.assertTrue(all(bool(g.isfinite().all()) and g.norm().item()>0 for g in grads))
        reference.assert_isolated(self.c)

    def test_reference_checkpoint_replay_retains_T_mapping(self):
        prefix=torch.randn(2,1024,device='cuda',requires_grad=True)
        target=[248046]
        self.b.checkpointing=False
        loss=self.ref.reconstruct(prefix,target)
        expected=torch.autograd.grad(loss,prefix)[0]
        self.b.checkpointing=True
        try:
            actual_loss=self.ref.reconstruct(prefix,target)
            # Replay after changing the outer flag must retain the scoped map.
            self.b.checkpointing=False
            actual=torch.autograd.grad(actual_loss,prefix)[0]
        finally:
            self.b.checkpointing=False
        self.assertTrue(torch.equal(loss,actual_loss))
        self.assertTrue(torch.equal(expected,actual))

    def test_complete_clean_packet_twins_and_supplied_returns(self):
        from pathlib import Path
        from qwen_one_two.training.clean_packets import CleanPackets
        from qwen_one_two.training.manifest import DIAGNOSTIC_SHA
        from qwen_one_two.codec.schema import Port, Status
        encoder=CleanEncoder(self.b,self.ref,checkpoint_hash=DIAGNOSTIC_SHA,
                             profile_hash=digest(Path('docs/codec-feedback-precision-v3.md').read_bytes()))
        packets=[]
        for index,hidden in enumerate((b' Hidden active.',b' Hidden expired.')):
            view,p=self.make(hidden)
            text='The bound participant'
            nodes=dict(p.nodes);nodes['target']=Node('request_semantics',digest(text.encode()))
            p=Provenance(nodes,p.sources)
            target=reference_input('target',{'referent':Field(text,'target')},p)
            view=CleanView(view.references+(target,),view.evidence,frozenset({'q','fact','target'}),purpose='reader',operation='Noul')
            encoded=encoder.encode(view,p,run=f'clean-packet-{index}')
            factory=CleanPackets(encoded,view,p)
            packet=factory.request(Port.NOUL,0,1,evidence=(factory.span(0,0,len(encoded['registry'].get(encoded['sources'][0]).token_ids)),))
            request,request_audit=factory.encode(self.c,packet)
            result=factory.supplied_return(Status.ASSESSED,scalar=torch.tensor(.25,device='cuda'))
            returned,return_audit=factory.encode(self.c,result)
            self.assertTrue(return_audit["supplied_component_return"])
            self.assertFalse(request_audit["supplied_component_return"])
            packets.append((request,returned,[(int(x.role),x.owner,x.slot) for x in packet.positions]))
            with self.assertRaises(ValueError):factory.supplied_return(Status.ASSESSED,scalar=torch.tensor(.5,device='cuda'))
        self.assertTrue(torch.equal(packets[0][0],packets[1][0]))
        self.assertTrue(torch.equal(packets[0][1],packets[1][1]))
        self.assertEqual(packets[0][2],packets[1][2])

    def test_clean_exact_attributes_are_local_grounded_and_immutable(self):
        from pathlib import Path
        from dataclasses import replace
        from qwen_one_two.training.clean_packets import CleanPackets
        from qwen_one_two.training.inputs import canonical
        from qwen_one_two.training.manifest import DIAGNOSTIC_SHA
        from qwen_one_two.codec.schema import Port,ValueKind,Role
        texts=[b'2',b'1.20',b'true',b'""',b'Jo']
        sources={str(i):Source(str(i),1,t,digest(t)) for i,t in enumerate(texts)}
        bindings=tuple(Binding(str(i),1,0,len(t)) for i,t in enumerate(texts))
        nodes={str(i):Node('source_fact',digest(t),binding=bindings[i]) for i,t in enumerate(texts)}
        for key,text in (('q','What are the supplied fields?'),('target','Item'),('key','Property')):
            nodes[key]=Node('request_semantics',digest(text.encode()))
        nodes['absent']=Node('request_semantics',digest(canonical({'kind':int(ValueKind.INT64),'present':False})))
        p=Provenance(nodes,sources)
        refs=(reference_input('question',{'proposition':Field('What are the supplied fields?','q')},p),
              reference_input('target',{'referent':Field('Item','target')},p),
              reference_input('key',{'property':Field('Property','key')},p),
              reference_input('target',{'referent':Field('Jo','4')},p))
        view=CleanView(refs,bindings,frozenset(nodes),operation='Noul')
        encoder=CleanEncoder(self.b,self.ref,checkpoint_hash=DIAGNOSTIC_SHA,
                             profile_hash=digest(Path('docs/codec-feedback-precision-v3.md').read_bytes()))
        encoded=encoder.encode(view,p,run='clean-exact-fixture')
        factory=CleanPackets(encoded,view,p)
        spans=[factory.span(i,0,len(encoded['registry'].get(ref).token_ids)) for i,ref in enumerate(encoded['sources'])]
        attrs=(factory.source_attribute(2,spans[0],ValueKind.INT64),
               factory.source_attribute(2,spans[1],ValueKind.DECIMAL),
               factory.source_attribute(2,spans[2],ValueKind.BOOLEAN),
               factory.quoted_utf8_attribute(2,spans[3]),
               factory.source_attribute(2,spans[4],ValueKind.SOURCE_STRING),
               factory.reference_attribute(2,3,spans[4]),
               factory.absent_attribute(2,ValueKind.INT64,p,'absent'))
        self.assertEqual(attrs[0].value.data,2);self.assertEqual(attrs[1].value.data,(120,-2))
        self.assertEqual(attrs[3].value.data,b'');self.assertTrue(attrs[3].value.present)
        self.assertFalse(attrs[6].value.present)
        packet=factory.request(Port.NOUL,0,1,target_attributes=attrs)
        self.assertTrue(bool(self.c.encode(packet).isfinite().all()))
        self.assertEqual(len([x for x in packet.positions if x.role==Role.ATTRIBUTE_DESC]),7)
        with self.assertRaises(ValueError):factory.source_attribute(2,replace(spans[0],source=attrs[5].value.data),ValueKind.INT64)
