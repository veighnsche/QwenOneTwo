"""Read-only fused predicates and transactional failure boundaries on CUDA."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import torch
from qwen_one_two.g1_proof import configure
from qwen_one_two.codec.finite import all_finite
from qwen_one_two.codec.session import Session
from qwen_one_two.codec.runtime import Runtime
from qwen_one_two.training.engine import Window


class FiniteGPU(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        configure()

    def test_real_extremes_views_dtypes_empty_and_readonly(self):
        xs=[]
        for dtype in (torch.bfloat16,torch.float32,torch.float64):
            f=torch.finfo(dtype)
            xs.append(torch.tensor([0.,-0.,f.max,-f.max,f.tiny,f.smallest_normal/2],device='cuda',dtype=dtype))
        x=torch.randn(8,9,device='cuda',requires_grad=True)
        xs += [x.t(),x[1::2,2::2],x[:0],None]
        versions=[v._version for v in xs if v is not None]
        self.assertTrue(all_finite(xs));self.assertTrue(all_finite([]))
        self.assertEqual(versions,[v._version for v in xs if v is not None])
        self.assertIsNone(x.grad)
        self.assertTrue(torch.equal(torch.autograd.grad(x.square().sum(),x)[0],2*x))
        # Complex absolute values may overflow despite finite components; the
        # fallback must check components, rather than taking their magnitude.
        z=torch.tensor([complex(1.7e308,1.7e308)],device='cuda',dtype=torch.complex128)
        self.assertTrue(all_finite([z,torch.ones(3,device='cuda',dtype=torch.int64)]))

    def test_nan_and_infinity_at_each_position(self):
        xs=[torch.ones(9,device='cuda') for _ in range(49)]
        for i in (0,24,48):
            for invalid in (float('nan'),float('inf'),-float('inf')):
                xs[i][-1]=invalid
                self.assertFalse(all_finite(xs));xs[i][-1]=1
        self.assertTrue(all_finite(xs))

    def test_cpu_rejected_without_tensor_compute(self):
        with patch('torch._foreach_norm',side_effect=AssertionError('CPU fallback')):
            self.assertFalse(all_finite([torch.empty(0)]))

    def window(self):
        p=torch.nn.Parameter(torch.tensor([.2],device='cuda'))
        q=torch.nn.Parameter(torch.tensor([.7],device='cuda'))
        return Window([{'name':'fixture','named_parameters':[('p',p),('q',q)],'lr':1e-4,'weight_decay':0.}]),p,q

    def fill(self,w):
        while w.examples<8:w.add_episode({},complete=True,positions=20)

    def test_finite_loss_nonfinite_derivative_aborts_before_accumulation(self):
        class Bad(torch.autograd.Function):
            @staticmethod
            def forward(ctx,x):return x.sum()
            @staticmethod
            def backward(ctx,g):return g.new_full((1,),float('nan'))
        w,p,q=self.window()
        w.add_episode({'native':p.sum()},complete=True,positions=20)
        with self.assertRaisesRegex(ValueError,'nonfinite component gradient'):
            w.add_episode({'native':Bad.apply(p)},complete=True,positions=20)
        self.assertEqual(w.examples,0);self.assertFalse(w.counts);self.assertFalse(w.buffers)
        self.assertTrue(all(x.grad is None for x in (p,q)));self.assertEqual(w.optimizer_updates,0)

    def test_invalid_loss_rejected_before_any_vjp(self):
        for invalid in (float('nan'),float('inf'),-float('inf')):
            w,p,q=self.window()
            with patch('torch.autograd.grad',side_effect=AssertionError('must reject before VJP')) as vjp:
                with self.assertRaisesRegex(ValueError,'invalid component loss'):
                    w.add_episode({'native':p.sum(),'structure':(q*invalid).sum()},complete=True,positions=20)
                vjp.assert_not_called()
            self.assertEqual(w.examples,0);self.assertFalse(w.buffers)

    def test_finite_gradient_sum_overflow_rejected(self):
        w,p,q=self.window()
        for _ in range(2):w.add_episode({'native':(p*3e38).sum()},complete=True,positions=20)
        self.fill(w)
        with patch.object(w.optimizer,'step',side_effect=AssertionError('must not step')):
            with self.assertRaisesRegex(ValueError,'nonfinite combined gradient'):w.step()
        self.assertEqual(w.examples,0);self.assertEqual(w.optimizer_updates,0)

    def test_prestate_rejection_and_poststep_rollback(self):
        for fault in ('pre_inf','pre_cpu','post_parameter','post_moment','post_cpu'):
            w,p,q=self.window()
            # Synthetic optimizer metadata; no real-model update or fitting.
            state={'step':torch.tensor(1.,device='cuda'),'exp_avg':torch.ones_like(p),'exp_avg_sq':torch.ones_like(p)}
            w.optimizer.state[p]=state
            w.add_episode({'native':p.sum()},complete=True,positions=20);self.fill(w)
            before=p.detach().clone();old={k:v.clone() for k,v in state.items()}
            if fault.startswith('pre_'):
                if fault=='pre_inf':state['exp_avg'].fill_(float('inf'))
                else:state['step']=torch.tensor(1.)  # invalid CPU metadata; never computed on
                with patch.object(w.optimizer,'step',side_effect=AssertionError('must not step')):
                    with self.assertRaisesRegex(ValueError,'invalid optimizer state'):w.step()
            else:
                def corrupt():
                    if fault=='post_parameter':
                        with torch.no_grad():p.fill_(float('nan'))
                    elif fault=='post_moment':state['exp_avg_sq'].fill_(float('inf'))
                    else:state['step']=torch.tensor(2.)
                with patch.object(w.optimizer,'step',side_effect=corrupt):
                    with self.assertRaisesRegex(ValueError,'nonfinite optimizer update'):w.step()
                self.assertTrue(all(torch.equal(v,w.optimizer.state[p][k]) for k,v in old.items()))
            self.assertTrue(torch.equal(p,before));self.assertIsNone(q.grad)
            self.assertEqual(w.optimizer_updates,0);self.assertEqual(w.scheduled_windows,0)
            self.assertEqual(w.examples,0)

    def test_session_metadata_guards_and_cancellation_preserved(self):
        for fault in ('mutation','position','lineage','cancel_before','cancel_after'):
            rt=Runtime('guard-fixture',1,100,1)
            prior=torch.ones(1,2,device='cuda')
            initial=SimpleNamespace(layers=((prior,prior.clone()),),position=0,lineage=1)
            final=SimpleNamespace(layers=initial.layers,position=2 if fault=='position' else 1,
                                  lineage=2 if fault=='lineage' else 1)
            calls=[]
            class Backbone:
                def initial(self):return initial
                def __call__(self,*args):
                    calls.append(True)
                    if fault=='mutation':prior.add_(1)
                    return torch.ones(1,1,2,device='cuda'),final
            cancelled=lambda:fault=='cancel_before' or (fault=='cancel_after' and bool(calls))
            s=Session(Backbone(),None,rt,cancelled)
            error=InterruptedError if fault.startswith('cancel') else RuntimeError
            with self.assertRaises(error):s.step(torch.ones(1,2,device='cuda'),'D','fixture',False)
            self.assertEqual(rt.terminal,'Cancelled' if fault.startswith('cancel') else 'Failed')
            self.assertEqual(rt.position,1)
            self.assertEqual(len(calls),0 if fault=='cancel_before' else 1)
            self.assertEqual(len(s.boundaries),1 if fault=='cancel_after' else 0)

    def test_session_hidden_and_each_cache_failure_same_boundary(self):
        for bad in (0,1,24,48):
            values=[torch.ones(1,1,2,device='cuda') for _ in range(49)]
            values[bad][...,0]=float('nan') if bad%2 else float('inf')
            rt=Runtime('finite-fixture',1,100,1)
            initial=SimpleNamespace(layers=(),position=0,lineage=1)
            final=SimpleNamespace(layers=tuple(zip(values[1::2],values[2::2])),position=1,lineage=1)
            class Backbone:
                def initial(self):return initial
                def __call__(self,*args):return values[0],final
            s=Session(Backbone(),None,rt)
            with self.assertRaisesRegex(RuntimeError,'nonfinite causal state'):
                s.step(torch.ones(1,2,device='cuda'),'D','fixture',False)
            self.assertEqual(rt.terminal,'Failed');self.assertFalse(s.boundaries)
            self.assertEqual(s.version_checks,0);self.assertEqual(rt.position,1)
