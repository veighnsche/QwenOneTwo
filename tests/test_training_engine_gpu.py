"""Tiny CUDA algebra/optimizer tests, no model fitting or capability training."""
import unittest
from unittest.mock import patch
import torch
from qwen_one_two.g1_proof import configure
from qwen_one_two.training.engine import Window, schedule, guided_events
from qwen_one_two.training.losses import (permitted_set, exact_value, regression, Eligibility,
    ExampleLosses, native_tokens, alignment, contrast)


class EngineGPU(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        configure()

    def window(self, phase='components'):
        p=torch.nn.Parameter(torch.tensor([.2],device='cuda'))
        q=torch.nn.Parameter(torch.tensor([.7],device='cuda'))
        groups=[{'name':'fixture','named_parameters':[('p',p),('q',q)],'lr':1e-4,'weight_decay':.01}]
        return Window(groups,phase),p,q

    def finish_empty(self,w):
        while w.examples<8:w.add_episode({},complete=True,positions=20)

    def test_eligible_example_denominators_and_component_weighting(self):
        w,p,q=self.window()
        w.add_episode({'native':(2*p).sum(),'structure':(4*p+q).sum()},complete=True,positions=20)
        w.add_episode({'native':(6*p).sum()},complete=True,positions=20)
        self.finish_empty(w);w.combine()
        # native (2+6)/2 + structure .5*4; q only structure .5*1.
        self.assertEqual(p.grad.item(),6.);self.assertEqual(q.grad.item(),.5)
        report=w.step();self.assertEqual(report['eligible_examples'],{'native':2,'structure':1})
        self.assertTrue(all(t.is_cuda for state in w.optimizer.state.values() for t in state.values()))

    def test_inactive_parameter_with_existing_momentum_stays_identical(self):
        w,p,q=self.window()
        w.add_episode({'native':(p+q).sum()},complete=True,positions=20)
        self.finish_empty(w);w.step()
        before=q.detach().clone();state={k:v.clone() for k,v in w.optimizer.state[q].items()}
        w.add_episode({'native':p.sum()},complete=True,positions=20)
        self.finish_empty(w);w.combine();self.assertIsNone(q.grad)
        w.step();self.assertTrue(torch.equal(q,before))
        self.assertTrue(all(torch.equal(v,w.optimizer.state[q][k]) for k,v in state.items()))

    def test_invalid_exposures_zero_eligibility_and_nonfinite_abort(self):
        w,p,q=self.window();before=p.detach().clone()
        w.add_episode({'native':p.sum()},complete=True,positions=513)
        w.add_episode({'native':p.sum()},complete=False,positions=10)
        self.finish_empty(w);report=w.step()
        self.assertFalse(report['stepped']);self.assertEqual(len(report['failed_exposures']),2)
        self.assertEqual(w.scheduled_windows,1);self.assertTrue(torch.equal(p,before))
        w.add_episode({'native':p.sum()},complete=True,positions=20)
        with self.assertRaises(ValueError):w.add_episode({'native':(p*float('nan')).sum()},complete=True,positions=20)
        self.assertEqual(w.examples,0);self.assertFalse(w.buffers);self.assertTrue(torch.equal(p,before))

    def test_partial_optimizer_failure_restores_parameters_and_moments(self):
        w,p,q=self.window()
        w.add_episode({'native':(p+q).sum()},complete=True,positions=20)
        self.finish_empty(w);w.step()
        before=[x.detach().clone() for x in (p,q)]
        old={x:{k:v.clone() for k,v in w.optimizer.state[x].items()} for x in (p,q)}
        w.add_episode({'native':(p+q).sum()},complete=True,positions=20);self.finish_empty(w)
        def fail():
            with torch.no_grad():p.add_(99)
            w.optimizer.state[p]['exp_avg'].add_(99)
            raise RuntimeError('injected device-step failure')
        with patch.object(w.optimizer,'step',side_effect=fail):
            with self.assertRaises(RuntimeError):w.step()
        for x,y in zip((p,q),before):self.assertTrue(torch.equal(x,y))
        for x in (p,q):
            for k,v in old[x].items():self.assertTrue(torch.equal(v,w.optimizer.state[x][k]))
        self.assertEqual(w.scheduled_windows,1);self.assertEqual(w.examples,0)

    def test_loss_subtypes_and_raw_regression(self):
        logits=torch.zeros(4,device='cuda',requires_grad=True)
        self.assertAlmostEqual(permitted_set(logits,[1,3]).item(),__import__('math').log(2),places=6)
        with self.assertRaises(ValueError):permitted_set(logits,[4])
        raw=torch.tensor(1.5,device='cuda',requires_grad=True)
        loss=regression(raw,interval=(.6,.8));self.assertGreater(torch.autograd.grad(loss,raw)[0].item(),0)
        with self.assertRaises(ValueError):regression(raw,point=.7,interval=(.6,.8))
        bytes_=torch.zeros(8,256,device='cuda',requires_grad=True)
        integer=exact_value('int64','construct',byte_logits=bytes_,value=-(1<<63))
        self.assertAlmostEqual(integer.item(),__import__('math').log(256),places=5)
        decimal=exact_value('decimal','construct',byte_logits=bytes_,exponent_logits=torch.zeros(37,device='cuda'),value=(120,-2))
        self.assertAlmostEqual(decimal.item(),(__import__('math').log(256)+__import__('math').log(37))/2,places=5)
        self.assertIsNone(exact_value('int64','copy'))
        self.assertIsNone(exact_value('reference','construct'))
        length=torch.zeros(129,device='cuda',requires_grad=True)
        empty=exact_value('utf8','construct',length_logits=length,value=b'')
        self.assertAlmostEqual(empty.item(),__import__('math').log(129),places=5)

    def test_masks_use_reference_validity_and_actual_reason_carrier(self):
        m=Eligibility(True,'Assessed','Insufficient','Score',True).mask()
        self.assertTrue(m['score']);self.assertFalse(m['reason'])
        m=Eligibility(True,'Insufficient','Assessed','Score',True).mask()
        self.assertTrue(m['status']);self.assertFalse(m['score']);self.assertFalse(m['reason'])
        self.assertTrue(Eligibility(True,'Insufficient','Insufficient','Noul',True).mask()['reason'])
        self.assertFalse(any(Eligibility(False,'Assessed','Assessed','Choice').mask().values()))
        self.assertFalse(any(Eligibility(True,'Assessed','Assessed','Choice',runtime_failure=True).mask().values()))
        self.assertFalse(any(Eligibility(True,None,'Assessed','Choice').mask().values()))

    def test_role_normalization_native_exclusions_and_detached_targets(self):
        value=torch.tensor(2.,device='cuda',requires_grad=True)
        book=ExampleLosses()
        book.add('alignment','q',value,role='question')
        for i in range(16):book.add('alignment',str(i),value*3,role='alternative')
        self.assertEqual(book.finish()['alignment'].item(),4.)
        logits=torch.randn(4,8,device='cuda',requires_grad=True)
        targets=torch.tensor([1,2,3,4],device='cuda')
        loss=native_tokens(logits,targets,['prefill','latent','native','runtime_control'],[True]*4)
        grads=torch.autograd.grad(loss,logits)[0]
        self.assertEqual(grads[:2].abs().sum().item()+grads[3].abs().sum().item(),0.)
        self.assertGreater(grads[2].abs().sum().item(),0.)
        a=torch.randn(2,3,device='cuda',requires_grad=True);r=torch.randn_like(a,requires_grad=True)
        grads=torch.autograd.grad(alignment(a,r),(a,r),allow_unused=True)
        self.assertIsNone(grads[1]);self.assertGreater(grads[0].norm().item(),0)
        candidates=torch.randn(3,2,3,device='cuda',requires_grad=True)
        loss=contrast(a,candidates,[0],[1,2]);grads=torch.autograd.grad(loss,(a,candidates),allow_unused=True)
        self.assertIsNone(grads[1]);self.assertGreater(grads[0].norm().item(),0)
        self.assertIsNone(contrast(a,candidates,[],[1]))

    def test_schedules_and_duplicate_ownership(self):
        self.assertEqual(schedule('components',100),1.)
        self.assertAlmostEqual(schedule('components',2000),.1)
        self.assertEqual(guided_events('transition',1,19),19)
        self.assertEqual(guided_events('transition',2000,19),0)
        self.assertEqual(guided_events('guided_joint',1,19,compatible=False),0)
        p=torch.nn.Parameter(torch.zeros(1,device='cuda'))
        with self.assertRaises(ValueError):Window([{'name':'bad','lr':1e-4,'weight_decay':0.,'named_parameters':[('p',p),('q',p)]}])
