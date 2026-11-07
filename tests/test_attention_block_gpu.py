"""GPU-only exact operator/VJP checks for the opt-in larger attention block."""
import unittest
import torch
from torch.utils.checkpoint import checkpoint
from qwen_one_two.g1_proof import configure
from qwen_one_two.codec.attention_block import graph_block,clear_graph_cache
from qwen_one_two.codec.recurrence import graph_recurrence
from test_recurrence_gpu import inputs


def oracle(q,k,v,beta,g,memory,z,weight,epsilon):
    pieces=[]
    for i in range(0,q.shape[1],32):
        y,memory=graph_recurrence(q[:,i:i+32],k[:,i:i+32],v[:,i:i+32],
                                  beta[:,i:i+32],g[:,i:i+32],memory)
        pieces.append(y)
    y=torch.cat(pieces,1)
    y=y*torch.rsqrt(y.square().mean(-1,keepdim=True)+epsilon)
    return (y*weight)*torch.nn.functional.silu(z),memory


def values(n,seed=17,dk=8,dv=7,heads=2,dtype=torch.float32):
    args=inputs(n,seed,dk,dv,heads,dtype=dtype)
    z=torch.randn_like(args[2]).requires_grad_()
    w=torch.randn(dv,device='cuda',dtype=torch.bfloat16 if dtype==torch.float32 else dtype).requires_grad_()
    return (*args,z,w)


class AttentionBlockGPU(unittest.TestCase):
    @classmethod
    def setUpClass(cls):configure()

    def compare(self,args,mode='both',checkpointed=False):
        versions=[x._version for x in args]
        expected=oracle(*args,1e-6)
        actual=(checkpoint(lambda *xs:graph_block(*xs,1e-6),*args,use_reentrant=False)
                if checkpointed else graph_block(*args,1e-6))
        self.assertTrue(all(torch.equal(a,b) for a,b in zip(actual,expected)))
        weights=[torch.randn_like(x) for x in actual]
        def loss(out):return sum((v*w).sum() for i,(v,w) in enumerate(zip(out,weights)) if mode=='both' or i==int(mode))
        ga=torch.autograd.grad(loss(actual),args,allow_unused=True)
        gb=torch.autograd.grad(loss(expected),args,allow_unused=True)
        self.assertTrue(all((a is None and b is None) or (a is not None and b is not None and torch.equal(a,b))
                            for a,b in zip(ga,gb)))
        self.assertEqual(versions,[x._version for x in args])

    def test_lengths_seeds_and_production_dimensions(self):
        for seed in (17,29,43):
            for n in (1,2,8,31,32,33,65,128):
                with self.subTest(seed=seed,length=n):self.compare(values(n,seed));clear_graph_cache()
        for n in (1,32,128):
            with self.subTest(production=n):self.compare(values(n,dk=128,dv=128,heads=16));clear_graph_cache()

    def test_long_connected_and_single_output_derivatives(self):
        for n in (512,1024):self.compare(values(n));clear_graph_cache()
        for mode in ('0','1'):self.compare(values(33),mode=mode);clear_graph_cache()

    def test_checkpoint_reuse_eviction_stream_and_independent_outputs(self):
        from qwen_one_two.codec import attention_block as module
        clear_graph_cache();old=module.CACHE_LIMIT;module.CACHE_LIMIT=1
        try:
            for _ in range(3):self.compare(values(33),checkpointed=True)
            args=values(8);first=graph_block(*args,1e-6);saved=[x.detach().clone() for x in first]
            self.compare(values(7))
            reference=oracle(*args,1e-6)
            ga=torch.autograd.grad(sum(x.square().sum() for x in first),args)
            gb=torch.autograd.grad(sum(x.square().sum() for x in reference),args)
            self.assertTrue(all(torch.equal(a,b) for a,b in zip(ga,gb)))
            second=graph_block(*args,1e-6)
            self.assertTrue(all(torch.equal(a,b) for a,b in zip(first,saved)))
            self.assertTrue(all(a.data_ptr()!=b.data_ptr() for a,b in zip(first,second)))
            stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):self.compare(args)
            torch.cuda.current_stream().wait_stream(stream)
            self.assertEqual(len(module._CACHE),1)
        finally:clear_graph_cache();module.CACHE_LIMIT=old

    def test_fp64_directional_derivative(self):
        args=values(33,dk=4,dv=5,heads=1,dtype=torch.float64)
        directions=[torch.randn_like(x)*.1 for x in args]
        def loss(xs):return sum(x.square().sum() for x in graph_block(*xs,1e-6))
        derivative=sum((g*d).sum() for g,d in zip(torch.autograd.grad(loss(args),args),directions))
        numeric=(loss([x.detach()+1e-6*d for x,d in zip(args,directions)])-
                 loss([x.detach()-1e-6*d for x,d in zip(args,directions)]))/(2e-6)
        self.assertLessEqual(float((derivative-numeric).abs()),1e-5+1e-3*float(numeric.abs()))

    def test_zero_strided_extremes_and_nonfinite_visibility(self):
        from qwen_one_two.codec.neural import finite
        for scale in (0.,1e-8,1e-6,1.):
            args=list(values(8))
            args=[(x.detach()*scale).requires_grad_() for x in args]
            self.compare(tuple(args))
        args=list(values(2));args[0]=torch.randn(1,2,2,16,device='cuda')[...,::2].requires_grad_()
        self.compare(tuple(args))
        with torch.no_grad():args[5][0,0,0,0]=float('nan')
        out=graph_block(*args,1e-6)
        for value in out:
            with self.assertRaises(ValueError):finite(value)

    def test_invalid_shapes_dtype_and_epsilon_reject(self):
        args=values(1)
        with self.assertRaises(ValueError):graph_block(*(x.bfloat16() for x in args),1e-6)
        with self.assertRaises(ValueError):graph_block(*args,float('nan'))
        with self.assertRaises(ValueError):graph_block(*values(1025),1e-6)


if __name__=='__main__':unittest.main()
