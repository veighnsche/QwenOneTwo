"""GPU-only precision regressions. Normal CPU test discovery skips this class."""
import os
import unittest
from unittest.mock import patch


@unittest.skipUnless(os.environ.get('CUDA_VISIBLE_DEVICES')=='GPU-225509c3-d702-6c27-de03-76b5e99b33b1',
                     'Explicit RTX3090 UUID required; no CPU execution')
class G1PrecisionGPU(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from qwen_one_two.g1_proof import configure
        configure()
        if not torch.cuda.is_available() or 'RTX 3090' not in torch.cuda.get_device_name(0):
            raise RuntimeError('Requested GPU unavailable; no fallback')
        cls.torch=torch

    def test_cancellation_preserves_bf16_low_part(self):
        t=self.torch
        from qwen_one_two.g1_precision_math import compensated_linear
        x=t.tensor([[1.,1.+2**-10]],device='cuda',requires_grad=True)
        w=t.tensor([[1.,-1.]],device='cuda',dtype=t.bfloat16)
        ordinary=t.nn.functional.linear(x.to(t.bfloat16),w).float()
        reference=t.nn.functional.linear(x,w.float())
        actual=compensated_linear(x,w)
        self.assertEqual(ordinary.item(),0.)
        self.assertEqual(reference.item(),-2**-10)
        self.assertTrue(t.equal(actual,reference))
        self.assertTrue(t.equal(t.autograd.grad(actual.sum(),x)[0],w.float()))

    def test_residual_rounding_is_real_and_removed(self):
        t=self.torch
        large=t.tensor([1.],device='cuda',dtype=t.bfloat16)
        small=t.tensor([2**-10],device='cuda',dtype=t.bfloat16)
        self.assertEqual((large+small).item(),1.)
        self.assertEqual((large.float()+small.float()).item(),1.+2**-10)

    def test_forward_products_really_use_bf16(self):
        t=self.torch
        from qwen_one_two.g1_precision_math import compensated_linear
        x=t.randn(4,64,device='cuda');w=t.randn(16,64,device='cuda')
        calls=[];original=t.mm
        def observed(a,b,**kw):
            calls.append((a.dtype,b.dtype,kw.get('out_dtype'),a.device.type,b.device.type))
            return original(a,b,**kw)
        with patch('torch.mm',side_effect=observed): actual=compensated_linear(x,w)
        self.assertEqual(len(calls),4)
        self.assertTrue(all(z==(t.bfloat16,t.bfloat16,t.float32,'cuda','cuda') for z in calls))
        self.assertEqual(actual.dtype,t.float32)

    def test_forward_and_master_gradients_match_fp32(self):
        t=self.torch
        from qwen_one_two.g1_precision_math import compensated_linear
        from qwen_one_two.g1_checks import grad_error,tensor_error
        for scale in (1.,1e-3,100.):
            for frozen in (False,True):
                t.manual_seed(29)
                x=(t.randn(8,64,device='cuda')*scale).requires_grad_()
                w=t.randn(16,64,device='cuda',dtype=t.float32)
                if frozen: w=w.to(t.bfloat16)
                else: w.requires_grad_()
                actual=compensated_linear(x,w)
                reference=t.nn.functional.linear(x,w.float())
                self.assertTrue(tensor_error(actual,reference,.02,.02,.01)['pass'])
                inputs=(x,) if frozen else (x,w)
                seed=t.randn_like(reference)
                g=t.autograd.grad(actual,inputs,grad_outputs=seed,retain_graph=True)
                r=t.autograd.grad(reference,inputs,grad_outputs=seed)
                self.assertTrue(grad_error(g,r,1e-4,.99999)['pass'])


if __name__=='__main__': unittest.main()
