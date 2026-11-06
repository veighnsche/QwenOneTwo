"""graph-v1 against the retained CUDA eager oracle; no optimizer steps."""
import json
import unittest
import torch
from torch.utils.checkpoint import checkpoint
from qwen_one_two.g1_proof import configure
from qwen_one_two.g1_checks import tensor_error,grad_error
from qwen_one_two.codec.backbone import recurrence
from qwen_one_two.codec.recurrence import graph_recurrence,clear_graph_cache


def inputs(length,seed=17,dk=128,dv=128,heads=16,dtype=torch.float32,scale=1.):
    torch.manual_seed(seed)
    def rand(*shape):return torch.randn(*shape,device='cuda',dtype=dtype)
    q=rand(1,length,heads,dk);q=q/q.norm(dim=-1,keepdim=True)/dk**.5
    k=rand(1,length,heads,dk);k=k/k.norm(dim=-1,keepdim=True)
    values=(q,k,rand(1,length,heads,dv)*.2,rand(1,length,heads).sigmoid(),
            -rand(1,length,heads).abs()*.1,rand(1,heads,dk,dv)*.01)
    return tuple((x*scale).detach().requires_grad_() for x in values)


def chunked(fn,*args):
    q,k,v,b,g,m=args;ys=[]
    for start in range(0,q.shape[1],32):
        y,m=fn(q[:,start:start+32],k[:,start:start+32],v[:,start:start+32],
               b[:,start:start+32],g[:,start:start+32],m)
        ys.append(y)
    return torch.cat(ys,1),m


class RecurrenceGPU(unittest.TestCase):
    @classmethod
    def setUpClass(cls):configure()

    def compare(self,length,seed=17,dk=128,dv=128,heads=16,scale=1.,mode='both'):
        args=inputs(length,seed,dk,dv,heads,scale=scale)
        versions=[x._version for x in args]
        actual=chunked(graph_recurrence,*args);expected=chunked(recurrence,*args)
        errors=[tensor_error(a,e,1e-6,1e-5,1e-5) for a,e in zip(actual,expected)]
        self.assertTrue(all(e['pass'] for e in errors),errors)
        weights=[torch.randn_like(x) for x in actual]
        def loss(outputs):return sum((y*w).sum() for i,(y,w) in enumerate(zip(outputs,weights)) if mode=='both' or i==int(mode))
        ga=torch.autograd.grad(loss(actual),args,allow_unused=True)
        ge=torch.autograd.grad(loss(expected),args,allow_unused=True)
        gradients=[grad_error([a],[e],1e-4,.99999) for a,e in zip(ga,ge)]
        self.assertTrue(all(e['pass'] for e in gradients),gradients)
        self.assertTrue(all(torch.equal(a,e) for a,e in zip(actual,expected)))
        self.assertTrue(all((a is None and e is None) or
            (a is not None and e is not None and torch.equal(a,e)) for a,e in zip(ga,ge)))
        self.assertEqual(versions,[x._version for x in args])
        print(json.dumps(dict(length=length,seed=seed,dk=dk,dv=dv,scale=scale,mode=mode,
                              forward=errors,gradients=gradients)),flush=True)

    def test_production_shapes_and_temporal_boundaries(self):
        for seed in (17,29,43):
            for length in (1,2,8,16,31,32,33,65):
                with self.subTest(seed=seed,length=length):self.compare(length,seed)

    def test_long_connected_cache(self):
        for length in (512,1024):self.compare(length)

    def test_near_zero_padding_and_single_output_derivatives(self):
        for scale in (0.,1e-8,1e-6,1.):self.compare(2,dk=7,dv=37,heads=2,scale=scale)
        for mode in ('0','1'):self.compare(8,dk=8,dv=7,heads=2,mode=mode)

    def test_checkpoint_and_three_repeats_are_exact(self):
        args=inputs(32)
        expected=graph_recurrence(*args)
        ge=torch.autograd.grad(sum(x.square().mean() for x in expected),args)
        for _ in range(3):
            clear_graph_cache() # Also exercise capture under checkpoint hooks.
            actual=checkpoint(graph_recurrence,*args,use_reentrant=False)
            ga=torch.autograd.grad(sum(x.square().mean() for x in actual),args)
            self.assertTrue(all(torch.equal(a,e) for a,e in zip(actual,expected)))
            self.assertTrue(all(torch.equal(a,e) for a,e in zip(ga,ge)))
            self.assertTrue(all(g.is_cuda and bool(g.isfinite().all()) and float(g.norm())>0 for g in ga))

    def test_gpu_fp64_directional_derivatives(self):
        for length in (1,2,8,16):
            args=inputs(length,dk=4,dv=5,heads=1,dtype=torch.float64)
            directions=[torch.randn_like(x)*.1 for x in args]
            def loss(values):
                y,m=graph_recurrence(*values)
                return y.square().sum()+m.square().sum()
            analytic=sum((g*d).sum() for g,d in zip(torch.autograd.grad(loss(args),args),directions))
            plus=loss([x.detach()+1e-6*d for x,d in zip(args,directions)])
            minus=loss([x.detach()-1e-6*d for x,d in zip(args,directions)])
            numerical=(plus-minus)/(2e-6)
            error=float((analytic-numerical).abs())
            self.assertLessEqual(error,1e-5+1e-3*float(numerical.abs()))
            print(json.dumps(dict(fp64_steps=length,absolute_error=error)),flush=True)

    def test_unsupported_precision_and_lengths_reject(self):
        args=inputs(1)
        with self.assertRaises(ValueError):graph_recurrence(*(x.bfloat16() for x in args))
        with self.assertRaises(ValueError):graph_recurrence(*inputs(33))

    def test_eviction_reuse_and_cuda_stream_isolation(self):
        import qwen_one_two.codec.recurrence as module
        clear_graph_cache();limit=module.CACHE_LIMIT;module.CACHE_LIMIT=2
        try:
            records=[inputs(n,seed=17+n) for n in (8,16,4)]
            actual=[graph_recurrence(*a) for a in records]
            expected=[recurrence(*a) for a in records]
            # Earlier calls have been evicted before their backward executes.
            flat=tuple(x for a in records for x in a)
            def loss(rows):return sum(x.square().sum() for row in rows for x in row)
            ga=torch.autograd.grad(loss(actual),flat)
            ge=torch.autograd.grad(loss(expected),flat)
            self.assertTrue(all(torch.equal(a,e) for a,e in zip(ga,ge)))
            stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):other=graph_recurrence(*records[0])
            torch.cuda.current_stream().wait_stream(stream)
            self.assertTrue(all(torch.equal(a,e) for a,e in zip(other,expected[0])))
            self.assertLessEqual(len(module._CACHE),2)
        finally:
            clear_graph_cache();module.CACHE_LIMIT=limit

    def test_nonfinite_state_remains_visible_to_existing_guard(self):
        from qwen_one_two.codec.neural import finite
        args=inputs(2)
        with torch.no_grad():args[-1][0,0,0,0]=float('nan')
        y,m=graph_recurrence(*args)
        with self.assertRaises(ValueError):finite(y)
        with self.assertRaises(ValueError):finite(m)


if __name__=='__main__':unittest.main()
