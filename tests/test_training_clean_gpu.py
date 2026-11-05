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
        # Identifier marks synthetic engineering state, never semantic approval.
        cls.ref.freeze(digest(b'untrained engineering reference fixture'))

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
        encoder = CleanEncoder(self.b,self.ref,checkpoint_hash='1'*64,profile_hash='2'*64)
        view,p = self.make(b' Hidden: active.')
        a = encoder.encode(view,p,run='twin-a')
        view,p = self.make(b' Hidden: expired and more text.')
        b = encoder.encode(view,p,run='twin-b')
        self.assertEqual(a['input_hash'],b['input_hash'])
        self.assertEqual(a['audit'],b['audit'])
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
