"""GPU-only check of the functional_call race that private views must avoid."""
import threading
from types import SimpleNamespace
import unittest
import torch
from torch import nn
from torch.func import functional_call
from qwen_one_two.g1_device import require_cuda
from qwen_one_two.training.shared_frozen import clone_frozen,check_views


class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight=nn.Parameter(torch.tensor([[2.]],device='cuda'),requires_grad=False)
        self.register_buffer('fixed',torch.ones(1,device='cuda'))
        self.tokenizer={'mutable':[]};self.duals=[]
        self.model=SimpleNamespace(model=SimpleNamespace(rotary_emb=SimpleNamespace(rope_type='default')))
        self.table=SimpleNamespace(precision_owner=self)

    def forward(self,x):return x@self.weight


class Reference(nn.Module):
    def __init__(self,backbone):
        super().__init__();object.__setattr__(self,'backbone',backbone)
        self.weight=nn.Parameter(torch.tensor([[7.]],device='cuda'),requires_grad=False)

    def assert_isolated(self,codec):
        assert self.weight.data_ptr()!=self.backbone.weight.data_ptr()
        assert all(self.weight.data_ptr()!=p.data_ptr() for p in codec.parameters())


class SharedFrozenGPU(unittest.TestCase):
    @classmethod
    def setUpClass(cls):require_cuda(torch.cuda)

    def bundle(self):
        b=Backbone();c=nn.Module();c.table=b.table
        return b,c,Reference(b)

    def test_scoped_reference_swap_cannot_change_other_lane(self):
        original=self.bundle();copied,sharing=clone_frozen(original)
        b,c,ref=original;other,_,other_ref=copied
        self.assertIs(b.weight,other.weight)
        self.assertIs(ref.weight,other_ref.weight)
        swapped=threading.Event();release=threading.Event();errors=[];outputs=[]
        def pause(module,args):
            self.assertIs(module.weight,ref.weight)
            swapped.set();release.wait()
        hook=b.register_forward_pre_hook(pause)
        def run():
            try:
                with torch.no_grad():outputs.append(functional_call(b,{'weight':ref.weight},(torch.ones(1,1,device='cuda'),)))
            except BaseException as error:errors.append(error);swapped.set()
        thread=threading.Thread(target=run);thread.start();swapped.wait()
        try:
            if errors:raise errors[0]
            self.assertIsNot(b.weight,other.weight)
            self.assertEqual(float(other(torch.ones(1,1,device='cuda')).item()),2.)
            other.tokenizer['mutable'].append('private')
            self.assertEqual(b.tokenizer['mutable'],[])
        finally:release.set();thread.join();hook.remove()
        if errors:raise errors[0]
        self.assertEqual(float(outputs[0].item()),7.)
        self.assertIs(b.weight,other.weight)
        self.assertEqual(check_views(original,copied),sharing)

    def test_trainable_or_duplicated_tensors_rejected(self):
        bundle=self.bundle();bundle[0].weight.requires_grad_(True)
        with self.assertRaisesRegex(ValueError,'frozen tensors'):clone_frozen(bundle)
        bundle[0].weight.requires_grad_(False)
        copied,_=clone_frozen(bundle)
        copied[0].weight=nn.Parameter(copied[0].weight.detach().clone(),requires_grad=False)
        with self.assertRaisesRegex(ValueError,'frozen tensor duplicated'):check_views(bundle,copied)


if __name__=='__main__':unittest.main()
