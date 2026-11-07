"""graph-v1: replay exact eager CUDA equations without per-op host dispatch.

Backward replays forward+VJP from this invocation's saved inputs. Graphs can be
reused across layers/cache versions without stale activations. No fitting.
"""
import os
import threading
from contextlib import contextmanager
from collections import OrderedDict
import torch
from torch.autograd.function import once_differentiable
from ..g1_backbone import GPU_UUID
from ..g1_device import require_visibility
from ..training.host_metrics import current


def eager_recurrence(q,k,v,beta,g,memory):
    outputs=[]
    for t in range(q.shape[1]):
        memory=memory*g[:,t].exp()[...,None,None]
        delta=(v[:,t]-(memory*k[:,t].unsqueeze(-1)).sum(-2))*beta[:,t].unsqueeze(-1)
        memory=memory+k[:,t].unsqueeze(-1)*delta.unsqueeze(-2)
        outputs.append((memory*q[:,t].unsqueeze(-1)).sum(-2))
    return torch.stack(outputs,1),memory


_CACHE=OrderedDict()
_LOCK=threading.RLock()
CACHE_LIMIT=8
_SEALED=False


@contextmanager
def replay_only():
    """Diagnostic barrier: warm all streams first, forbid new global captures.

    Covers every thread in this process. Does not change equations or cache
    keys; a missing shape/backward graph fails rather than capturing alongside
    another lane. Call only after warmup has joined all lanes.
    """
    global _SEALED
    with _LOCK:
        previous=_SEALED;_SEALED=True
    try:yield
    finally:
        with _LOCK:_SEALED=previous


class _Graphs:
    def __init__(self,values):
        if (meter:=current()) is not None:meter.counts['recurrence_forward_capture_attempts']+=1
        self.inputs=tuple(x.detach().clone().requires_grad_() for x in values)
        self.dy=torch.zeros_like(values[2]);self.dn=torch.zeros_like(values[5])
        self.forward=torch.cuda.CUDAGraph()
        self.backward=None
        stream=torch.cuda.Stream(device=values[0].device)
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream),torch.no_grad():
            eager_recurrence(*self.inputs)
        torch.cuda.current_stream().wait_stream(stream)
        with torch.cuda.graph(self.forward,stream=stream),torch.no_grad():
            self.outputs=eager_recurrence(*self.inputs)
        if meter is not None:meter.counts['recurrence_forward_captures']+=1

    def prepare_backward(self):
        if self.backward is not None:return
        if _SEALED:raise RuntimeError('unwarmed backward in replay-only diagnostic')
        if (meter:=current()) is not None:meter.counts['recurrence_backward_capture_attempts']+=1
        stream=torch.cuda.Stream(device=self.inputs[0].device)
        stream.wait_stream(torch.cuda.current_stream())
        # Internal capture tape belongs to this backend, even during an outer
        # checkpoint. The public operation saves actual inputs via caller hooks.
        with torch.autograd.graph.saved_tensors_hooks(lambda x:x,lambda x:x):
            with torch.cuda.stream(stream),torch.enable_grad():
                out=eager_recurrence(*self.inputs)
                torch.autograd.grad(out,self.inputs,(self.dy,self.dn))
            torch.cuda.current_stream().wait_stream(stream)
            self.backward=torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.backward,stream=stream),torch.enable_grad():
                out=eager_recurrence(*self.inputs)
                self.gradients=torch.autograd.grad(out,self.inputs,(self.dy,self.dn))
        if meter is not None:meter.counts['recurrence_backward_captures']+=1

    def copy(self,values):
        with torch.no_grad():
            for target,value in zip(self.inputs,values):target.copy_(value)


def _graphs(values):
    stream=torch.cuda.current_stream(values[0].device).cuda_stream
    key=(values[0].device.index,stream,tuple((tuple(x.shape),x.dtype) for x in values))
    meter=current()
    if meter is not None:
        meter.counts['recurrence_lookups']+=1
        meter.counts['recurrence_hits' if key in _CACHE else 'recurrence_misses']+=1
        meter.counts['recurrence_key_requests:'+repr(key)]+=1
    if key not in _CACHE:
        if _SEALED:raise RuntimeError('unwarmed recurrence in replay-only diagnostic')
        _CACHE[key]=_Graphs(values)
        while len(_CACHE)>CACHE_LIMIT:
            _CACHE.popitem(last=False)
            if meter is not None:meter.counts['recurrence_evictions']+=1
    _CACHE.move_to_end(key)
    return _CACHE[key]


def clear_graph_cache():
    """Release only this backend's graphs, never live model/cache state."""
    with _LOCK:_CACHE.clear()


class _Recurrence(torch.autograd.Function):
    @staticmethod
    def forward(ctx,*values):
        ctx.set_materialize_grads(False)
        ctx.save_for_backward(*values)
        with _LOCK:
            graphs=_graphs(values);graphs.copy(values);graphs.forward.replay()
            # Static outputs are overwritten on replay: return independent state.
            return tuple(x.clone() for x in graphs.outputs)

    @staticmethod
    @once_differentiable
    def backward(ctx,dy,dn):
        values=ctx.saved_tensors;uses_q=dy is not None
        with _LOCK:
            graphs=_graphs(values)
            graphs.prepare_backward();graphs.copy(values)
            graphs.dy.copy_(dy) if dy is not None else graphs.dy.zero_()
            graphs.dn.copy_(dn) if dn is not None else graphs.dn.zero_()
            graphs.backward.replay()
            gradients=[x.clone() for x in graphs.gradients]
        if not uses_q:gradients[0]=None
        return tuple(gradients)


def graph_recurrence(q,k,v,beta,g,memory):
    """Batch-one FP32 CUDA recurrence with connected first derivatives.

    FP64 is for small derivative checks. Unsupported inputs, nested graph capture
    and higher derivatives have no alternate execution path.
    """
    values=(q,k,v,beta,g,memory)
    try:require_visibility()
    except RuntimeError as exc:raise ValueError(str(exc)) from exc
    if q.device.index!=0:raise ValueError('selected local cuda:0 required')
    if any(x.device.type!='cuda' or x.device!=q.device or x.dtype!=q.dtype for x in values):
        raise ValueError('recurrence requires same-device CUDA tensors of one dtype')
    if q.dtype not in (torch.float32,torch.float64):raise ValueError('FP32 state or FP64 derivative check required')
    if q.ndim!=4 or q.shape[0]!=1 or not 1<=q.shape[1]<=32:
        raise ValueError('recurrence requires batch one and 1..32 positions')
    _,length,heads,dk=q.shape
    if v.ndim!=4 or not 1<=dk<=128 or not 1<=v.shape[-1]<=128:
        raise ValueError('recurrence head dimensions must be 1..128')
    dv=v.shape[-1]
    if k.shape!=q.shape or v.shape[:3]!=(1,length,heads) or beta.shape!=g.shape or beta.shape!=(1,length,heads) or memory.shape!=(1,heads,dk,dv):
        raise ValueError('recurrence input shape mismatch')
    if torch.cuda.is_current_stream_capturing():raise ValueError('nested graph capture unsupported')
    return _Recurrence.apply(*(x.contiguous() for x in values))
