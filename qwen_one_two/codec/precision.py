"""Explicit feedback precision and historical BF16 arithmetic ablations."""
import torch
from torch.nn import functional as F
from ..g1_precision_math import _CompensatedLinear,COUNTS as COMPENSATED_COUNTS


COUNTS={'fp32_frozen_forward':0,'fp32_frozen_backward':0,'fp32_adapter_forward':0}


class FrozenFeedbackLinear(torch.autograd.Function):
    """FP32 arithmetic, BF16 storage; retain no promoted weight in the tape.

    Save the original frozen tensor and promote it transiently in each forward
    and backward. This avoids storing a separate FP32 weight per latent call.
    The FP32 reference remains the ordinary PyTorch linear implementation.
    """
    @staticmethod
    def forward(ctx,x,weight,bias):
        if not x.is_cuda or not weight.is_cuda:
            raise RuntimeError('Feedback projection requires CUDA; no CPU fallback')
        if weight.requires_grad or (bias is not None and bias.requires_grad):
            raise ValueError('Frozen projection cannot own trainable weights')
        ctx.save_for_backward(weight)
        ctx.input_dtype=x.dtype
        COUNTS['fp32_frozen_forward']+=1
        return F.linear(x.float(),weight.float(),None if bias is None else bias.float())

    @staticmethod
    def backward(ctx,g):
        weight,=ctx.saved_tensors
        dx=None
        if ctx.needs_input_grad[0]:
            dx=(g.float()@weight.float()).to(ctx.input_dtype)
            COUNTS['fp32_frozen_backward']+=1
        return dx,None,None


def feedback_linear(x,weight,bias=None):
    return FrozenFeedbackLinear.apply(x,weight,bias)


class ThreePartLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,weight):
        if not x.is_cuda or not weight.is_cuda:raise RuntimeError('CUDA only')
        flat=x.reshape(-1,x.shape[-1]).float()
        def parts(value,count):
            rest=value.float();result=[]
            for _ in range(count):
                part=rest.to(torch.bfloat16);result.append(part);rest=rest-part.float()
            return result
        xs=parts(flat,3);ws=parts(weight,1 if weight.dtype==torch.bfloat16 else 3)
        out=torch.zeros((len(flat),len(weight)),device='cuda',dtype=torch.float64)
        for a in xs:
            for b in ws:out=out+torch.mm(a,b.T,out_dtype=torch.float32).double()
        COMPENSATED_COUNTS['bf16_forward_products']+=len(xs)*len(ws)
        ctx.save_for_backward(x,weight)
        return out.float().reshape(*x.shape[:-1],weight.shape[0])

    backward=staticmethod(_CompensatedLinear.backward)


def linear(x,weight,bias=None):
    out=ThreePartLinear.apply(x,weight)
    return out if bias is None else out+bias.float()
