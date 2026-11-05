"""Optional three-part BF16 ablation for expanded serial-return traces."""
import torch
from ..g1_precision_math import _CompensatedLinear


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
        ctx.save_for_backward(x,weight)
        return out.float().reshape(*x.shape[:-1],weight.shape[0])

    backward=staticmethod(_CompensatedLinear.backward)


def linear(x,weight,bias=None):
    out=ThreePartLinear.apply(x,weight)
    return out if bias is None else out+bias.float()
