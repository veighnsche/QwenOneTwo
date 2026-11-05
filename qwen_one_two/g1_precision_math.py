"""Compensated CUDA BF16 GEMMs with FP32 output and FP32 master gradients.

The four products expand (x_hi+x_lo)(w_hi+w_lo). For BF16 frozen weights
w_lo is exactly zero, requiring only two products. This is an explicit mixed-
precision implementation, not a switch of the forward model GEMMs to FP32.
"""
import torch


COUNTS={'bf16_forward_products':0,'fp32_backward_products':0}


class _CompensatedLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,weight):
        if x.device.type!='cuda' or weight.device.type!='cuda':
            raise RuntimeError('G1 precision math requires CUDA; no CPU fallback')
        flat=x.reshape(-1,x.shape[-1]).float()
        hi=flat.to(torch.bfloat16)
        lo=(flat-hi.float()).to(torch.bfloat16)
        wh=weight.to(torch.bfloat16)
        result=torch.mm(hi,wh.T,out_dtype=torch.float32)
        result=result+torch.mm(lo,wh.T,out_dtype=torch.float32)
        COUNTS['bf16_forward_products']+=2
        if weight.dtype!=torch.bfloat16:
            wl=(weight.float()-wh.float()).to(torch.bfloat16)
            result=result+torch.mm(hi,wl.T,out_dtype=torch.float32)
            result=result+torch.mm(lo,wl.T,out_dtype=torch.float32)
            COUNTS['bf16_forward_products']+=2
        ctx.save_for_backward(x,weight)
        return result.reshape(*x.shape[:-1],weight.shape[0])

    @staticmethod
    def backward(ctx,grad_output):
        x,weight=ctx.saved_tensors
        g=grad_output.reshape(-1,weight.shape[0]).float()
        dx=dw=None
        # Differentiate the underlying linear map in FP32, avoiding another
        # BF16 truncation at every input/master-gradient edge. Discrete model
        # gates remain hard and receive their own explicit supervision.
        if ctx.needs_input_grad[0]:
            dx=(g@weight.float()).reshape_as(x).to(x.dtype)
            COUNTS['fp32_backward_products']+=1
        if ctx.needs_input_grad[1]:
            dw=(g.T@x.reshape(-1,x.shape[-1]).float()).to(weight.dtype)
            COUNTS['fp32_backward_products']+=1
        return dx,dw


def compensated_linear(x,weight,bias=None):
    result=_CompensatedLinear.apply(x,weight)
    return result if bias is None else result+bias.float()
