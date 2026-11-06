"""CUDA-only synthetic integrity checks; not learned semantic evidence."""
import unittest
import torch
from qwen_one_two.g1_proof import configure
from qwen_one_two.training.weight_repair import deflate_affine


class WeightRepairGPU(unittest.TestCase):
    @classmethod
    def setUpClass(cls):configure()

    def test_projection_preserves_complement_and_does_not_mutate_or_build_gradients(self):
        w=torch.randn(1024,256,device='cuda',requires_grad=True);b=torch.randn(1024,device='cuda',requires_grad=True)
        u=torch.randn(1024,device='cuda',dtype=torch.float64);u=u/u.norm()
        before_w=w.detach().clone();before_b=b.detach().clone();before_u=u.clone()
        rw,rb=deflate_affine(w,b,u)
        self.assertTrue(torch.equal(w,before_w));self.assertTrue(torch.equal(b,before_b));self.assertTrue(torch.equal(u,before_u))
        self.assertFalse(rw.requires_grad or rb.requires_grad)
        self.assertLess(float((u@rw.double()).norm()/w.double().norm()),1e-6)
        self.assertLess(float(abs(u@rb.double())/b.double().norm()),1e-6)
        v=torch.randn(1024,device='cuda',dtype=torch.float64);v=v-u*(u@v);v=v/v.norm()
        self.assertTrue(torch.allclose(v@rw.double(),v@w.double(),rtol=1e-5,atol=1e-6))
        self.assertTrue(torch.allclose(v@rb.double(),v@b.double(),rtol=1e-5,atol=1e-6))
        self.assertIsNone(w.grad);self.assertIsNone(b.grad)

    def test_invalid_inputs_reject_before_any_edit(self):
        w=torch.randn(1024,256,device='cuda');b=torch.randn(1024,device='cuda');u=torch.zeros(1024,device='cuda')
        with self.assertRaisesRegex(ValueError,'zero common'):deflate_affine(w,b,u)
        with self.assertRaisesRegex(ValueError,'FP32'):deflate_affine(w.bfloat16(),b,u+1)
        with self.assertRaisesRegex(ValueError,'shape'):deflate_affine(w,b,u[:3])
