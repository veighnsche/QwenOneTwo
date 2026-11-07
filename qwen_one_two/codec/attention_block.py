"""Opt-in exact-operation recurrence/normalization/gating CUDA replay.

Dense projections stay outside capture. Static inputs contain actual invocation
values, never a promoted frozen matrix or an implicit adapter/module lookup.
"""
from collections import OrderedDict
import math
import threading
import torch
from torch.nn import functional as F
from torch.autograd.function import once_differentiable
from ..g1_device import require_visibility
from ..training.host_metrics import current
from .recurrence import eager_recurrence

CACHE_LIMIT=8
_CACHE=OrderedDict()
_LOCK=threading.RLock()


def eager_block(q,k,v,beta,g,memory,z,weight,epsilon):
    pieces=[]
    for i in range(0,q.shape[1],32):
        out,memory=eager_recurrence(q[:,i:i+32],k[:,i:i+32],v[:,i:i+32],
                                    beta[:,i:i+32],g[:,i:i+32],memory)
        pieces.append(out)
    y=torch.cat(pieces,1)
    y=y*torch.rsqrt(y.square().mean(-1,keepdim=True)+epsilon)
    return (y*weight)*F.silu(z),memory


class _Graphs:
    def __init__(self,values,epsilon):
        self.inputs=tuple(x.detach().clone().requires_grad_() for x in values)
        self.epsilon=epsilon
        self.dy=torch.zeros_like(values[2]);self.dm=torch.zeros_like(values[5])
        self.forward=torch.cuda.CUDAGraph();self.backward=None
        stream=torch.cuda.Stream(device=values[0].device)
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream),torch.no_grad():eager_block(*self.inputs,epsilon)
        torch.cuda.current_stream().wait_stream(stream)
        with torch.cuda.graph(self.forward,stream=stream),torch.no_grad():
            self.outputs=eager_block(*self.inputs,epsilon)

    def prepare_backward(self):
        if self.backward is not None:return
        stream=torch.cuda.Stream(device=self.inputs[0].device)
        stream.wait_stream(torch.cuda.current_stream())
        with torch.autograd.graph.saved_tensors_hooks(lambda x:x,lambda x:x):
            with torch.cuda.stream(stream),torch.enable_grad():
                out=eager_block(*self.inputs,self.epsilon)
                torch.autograd.grad(out,self.inputs,(self.dy,self.dm))
            torch.cuda.current_stream().wait_stream(stream)
            self.backward=torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.backward,stream=stream),torch.enable_grad():
                out=eager_block(*self.inputs,self.epsilon)
                self.gradients=torch.autograd.grad(out,self.inputs,(self.dy,self.dm))
        if (meter:=current()) is not None:meter.counts['attention_block_backward_captures']+=1

    def copy(self,values):
        with torch.no_grad():
            for target,value in zip(self.inputs,values):target.copy_(value)


def _graphs(values,epsilon):
    key=(values[0].device.index,torch.cuda.current_stream(values[0].device).cuda_stream,
         tuple((tuple(x.shape),x.dtype) for x in values),epsilon)
    meter=current()
    if meter is not None:
        meter.counts['attention_block_lookups']+=1
        meter.counts['attention_block_hits' if key in _CACHE else 'attention_block_misses']+=1
    if key not in _CACHE:
        _CACHE[key]=_Graphs(values,epsilon)
        if meter is not None:meter.counts['attention_block_forward_captures']+=1
        while len(_CACHE)>CACHE_LIMIT:
            _CACHE.popitem(last=False)
            if meter is not None:meter.counts['attention_block_evictions']+=1
    _CACHE.move_to_end(key)
    return _CACHE[key]


def clear_graph_cache():
    with _LOCK:_CACHE.clear()


class _Block(torch.autograd.Function):
    @staticmethod
    def forward(ctx,epsilon,*values):
        ctx.set_materialize_grads(False);ctx.epsilon=epsilon;ctx.save_for_backward(*values)
        with _LOCK:
            graphs=_graphs(values,epsilon);graphs.copy(values);graphs.forward.replay()
            return tuple(x.clone() for x in graphs.outputs)

    @staticmethod
    @once_differentiable
    def backward(ctx,dy,dm):
        values=ctx.saved_tensors
        with _LOCK:
            graphs=_graphs(values,ctx.epsilon);graphs.prepare_backward();graphs.copy(values)
            graphs.dy.copy_(dy) if dy is not None else graphs.dy.zero_()
            graphs.dm.copy_(dm) if dm is not None else graphs.dm.zero_()
            graphs.backward.replay();gradients=[x.clone() for x in graphs.gradients]
        if dy is None:
            for i in (0,6,7):gradients[i]=None
        return None,*gradients


def graph_block(q,k,v,beta,g,memory,z,weight,epsilon):
    require_visibility()
    values=(q,k,v,beta,g,memory,z,weight)
    if any(x.device.type!='cuda' or x.device!=q.device for x in values) or q.device.index!=0:
        raise ValueError('attention block requires selected local CUDA tensors')
    if q.dtype not in (torch.float32,torch.float64) or any(x.dtype!=q.dtype for x in values[:-1]):
        raise ValueError('attention block requires FP32 state or FP64 derivative fixture')
    if weight.dtype not in (q.dtype,torch.bfloat16):raise ValueError('attention block norm dtype')
    if q.ndim!=4 or q.shape[0]!=1 or not 1<=q.shape[1]<=1024:raise ValueError('attention block length 1..1024')
    _,length,heads,dk=q.shape
    if v.ndim!=4 or not 1<=dk<=128 or not 1<=v.shape[-1]<=128:raise ValueError('attention block head dimensions')
    dv=v.shape[-1]
    if (k.shape!=q.shape or v.shape[:3]!=(1,length,heads) or beta.shape!=g.shape or
        beta.shape!=(1,length,heads) or memory.shape!=(1,heads,dk,dv) or z.shape!=v.shape or weight.shape!=(dv,)):
        raise ValueError('attention block shape mismatch')
    if type(epsilon) is not float or not math.isfinite(epsilon) or epsilon<=0:raise ValueError('attention block epsilon')
    if torch.cuda.is_current_stream_capturing():raise ValueError('nested attention block capture unsupported')
    return _Block.apply(epsilon,*(x.contiguous() for x in values))
