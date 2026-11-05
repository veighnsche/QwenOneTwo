"""Feedback precision regression: execute all tensor arithmetic on RTX 3090."""
import os
import unittest
import torch
from torch.nn import functional as F
from qwen_one_two.g1_backbone import GPU_UUID
from qwen_one_two.g1_proof import configure
from qwen_one_two.g1_precision_math import compensated_linear,COUNTS as BF16_COUNTS
from qwen_one_two.codec.precision import feedback_linear,COUNTS


@unittest.skipUnless(os.environ.get('CUDA_VISIBLE_DEVICES')==GPU_UUID,'explicit RTX 3090 required')
class FeedbackPrecisionGPU(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        configure()

    def test_frozen_projection_matches_reference_and_saves_original_bf16_only(self):
        torch.manual_seed(17)
        for count in (1,32,512):
            x=torch.randn(1,count,1024,device='cuda',requires_grad=True)
            weight=torch.randn(128,1024,device='cuda',dtype=torch.bfloat16)
            bias=torch.randn(128,device='cuda',dtype=torch.bfloat16)
            saved=[]
            def pack(t):
                saved.append((t.shape,t.dtype,t.data_ptr()));return t
            with torch.autograd.graph.saved_tensors_hooks(pack,lambda t:t):
                actual=feedback_linear(x,weight,bias)
            reference=F.linear(x,weight.float(),bias.float())
            self.assertTrue(torch.equal(actual,reference))
            self.assertEqual(saved,[(weight.shape,torch.bfloat16,weight.data_ptr())])
            upstream=torch.randn_like(actual)
            dx=torch.autograd.grad(actual,x,upstream)[0]
            rx=torch.autograd.grad(reference,x,upstream)[0]
            self.assertTrue(torch.equal(dx,rx))
            self.assertIsNone(weight.grad)

    def test_checkpoint_replay_and_bf16_vocabulary_product_remain_connected(self):
        from torch.utils.checkpoint import checkpoint
        torch.manual_seed(29)
        x=torch.randn(1,8,1024,device='cuda',requires_grad=True)
        frozen=torch.randn(128,1024,device='cuda',dtype=torch.bfloat16)
        vocab=torch.randn(256,128,device='cuda',dtype=torch.bfloat16)
        a=torch.randn(8,1024,device='cuda',requires_grad=True)
        b=torch.randn(128,8,device='cuda',requires_grad=True)
        def block(x):return feedback_linear(x,frozen)+F.linear(F.linear(x,a),b)*2.
        before=BF16_COUNTS['bf16_forward_products']
        direct=compensated_linear(block(x),vocab)
        replay=compensated_linear(checkpoint(block,x,use_reentrant=False),vocab)
        self.assertEqual(BF16_COUNTS['bf16_forward_products']-before,4)
        self.assertTrue(torch.equal(direct,replay))
        gd=torch.autograd.grad(direct.square().mean(),(x,a,b))
        gr=torch.autograd.grad(replay.square().mean(),(x,a,b))
        for actual,reference in zip(gd,gr):
            self.assertTrue(torch.equal(actual,reference))
            self.assertTrue(bool(actual.isfinite().all()))
            self.assertGreater(actual.square().mean().item(),0.)
        self.assertGreater(COUNTS['fp32_frozen_backward'],0)

    def test_trainable_weight_cannot_silently_lose_its_gradient(self):
        x=torch.randn(1,8,device='cuda',requires_grad=True)
        w=torch.randn(4,8,device='cuda',requires_grad=True)
        with self.assertRaises(ValueError):feedback_linear(x,w)


if __name__=='__main__':unittest.main()
