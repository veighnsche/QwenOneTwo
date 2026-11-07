"""Expanded-path memory profile: exact recurrence, 32-token rematerialization.

No cache/ledger mutation is inside a checkpoint. No state is detached. Dense
projections remain whole-chunk operations, preserving the original GEMM shape.
The frozen G1 backend remains unchanged for historical reproduction.
"""
import math
import torch
from types import MethodType
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from ..g1_backbone import Backbone as CoreBackbone,DualLinear,EffectiveTable,MODE
from .precision import linear,feedback_linear,COUNTS
from .recurrence import eager_recurrence as recurrence,graph_recurrence


def expanded_dual(self,x):
    if (self.compensated and self.base.weight.dtype==torch.bfloat16
            and self.precision_owner.feedback_precision=='fp32'):
        mode=MODE.get();a,b=(self.R_A,self.R_B) if mode=='R' else (self.D_A,self.D_B)
        if self.effective_reference:a=a.to(torch.bfloat16).float();b=b.to(torch.bfloat16).float()
        update=F.linear(F.linear(x.float(),a),b)*2.
        base=feedback_linear(x,self.base.weight,self.base.bias)
        COUNTS['fp32_adapter_forward']+=2
        if self.capture and mode==self.capture_mode and update.requires_grad:self.evaluation_outputs.append(update)
        return base+update
    if not self.compensated or self.base.weight.dtype!=torch.bfloat16 or self.precision_owner.precision_parts==2:
        return DualLinear.forward(self,x)
    mode=MODE.get();a,b=(self.R_A,self.R_B) if mode=='R' else (self.D_A,self.D_B)
    update=linear(linear(x,a),b)*2.
    base=linear(x,self.base.weight,self.base.bias)
    if self.capture and mode==self.capture_mode and update.requires_grad:self.evaluation_outputs.append(update)
    return base+update


class Table(EffectiveTable):
    def project(self,hidden,indices):
        if self.compensated and self.base.dtype==torch.bfloat16 and self.precision_owner.precision_parts==3:
            if bool(((indices==248077)|(indices==248078)).any()):raise ValueError('Private rows')
            return linear(hidden,self.base[indices])
        return super().project(hidden,indices)


class Backbone(CoreBackbone):
    temporal_chunk=32
    recurrence_backend='graph-v1'
    attention_block_backend='eager'

    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.precision_parts=2
        # v3: serial latent feedback needs consistent FP32 dense arithmetic.
        # Frozen weights remain BF16; native vocabulary GEMMs remain compensated
        # BF16. The original v2 path stays selectable for diagnosis/reproduction.
        self.feedback_precision='fp32'
        for dual in self.duals:
            # Non-module owner reference: do not register the backbone as its
            # own descendant or create a second parameter owner.
            object.__setattr__(dual,'precision_owner',self)
            dual.forward=MethodType(expanded_dual,dual)

    def bind_rows(self,codec):
        provider=Table(self.model.model.embed_tokens.weight,codec.budget_row,codec.score_row)
        provider.precision_owner=self;provider.compensated=self.compensated
        self.table=codec.table=provider
        return provider

    def _linear_attention(self,a,x,old):
        conv,memory=old
        raw=a.in_proj_qkv(x).transpose(1,2)
        joined=torch.cat((conv,raw),-1)
        new_conv=joined[:,:,-a.conv_kernel_size:]
        mixed=F.silu(F.conv1d(joined[:,:,-(raw.shape[-1]+a.conv_kernel_size-1):],
                             a.conv1d.weight.to(raw.dtype),a.conv1d.bias,groups=a.conv_dim)).transpose(1,2)
        q,k,v=torch.split(mixed,[a.key_dim,a.key_dim,a.value_dim],-1)
        q=q.reshape(1,-1,a.num_k_heads,a.head_k_dim).float()
        k=k.reshape_as(q).float();v=v.reshape(1,-1,a.num_v_heads,a.head_v_dim).float()
        q=q*torch.rsqrt(q.square().sum(-1,keepdim=True)+1e-6)
        k=k*torch.rsqrt(k.square().sum(-1,keepdim=True)+1e-6)
        q=q/math.sqrt(a.head_k_dim)
        if a.num_v_heads!=a.num_k_heads:
            q=q.repeat_interleave(a.num_v_heads//a.num_k_heads,2)
            k=k.repeat_interleave(a.num_v_heads//a.num_k_heads,2)
        beta=a.in_proj_b(x).sigmoid().float()
        g=-a.A_log.float().exp()*F.softplus(a.in_proj_a(x).float()+a.dt_bias.float())
        pieces=[]
        if self.recurrence_backend not in {'eager','graph-v1'}:
            raise ValueError('unknown recurrence backend')
        if self.attention_block_backend not in {'eager','graph-v1'}:
            raise ValueError('unknown attention block backend')
        if self.attention_block_backend=='graph-v1':
            if self.feedback_precision!='fp32' or x.dtype!=torch.float32 or self.recurrence_backend!='graph-v1':
                raise ValueError('attention block requires v3 FP32/recurrence graph profile')
            from .attention_block import graph_block
            # Independent dense gate projection is an explicit block input.
            # Its CUDA/frozen-weight admission remains outside capture; no
            # finite, Session or publication boundary is crossed.
            z=a.in_proj_z(x).reshape_as(v)
            y,memory=graph_block(q,k,v,beta,g,memory,z,a.norm.weight,float(a.norm.variance_epsilon))
            return a.out_proj(y.reshape(1,x.shape[1],-1)),(new_conv,memory)
        for i in range(0,x.shape[1],self.temporal_chunk):
            args=(q[:,i:i+self.temporal_chunk],k[:,i:i+self.temporal_chunk],v[:,i:i+self.temporal_chunk],
                  beta[:,i:i+self.temporal_chunk],g[:,i:i+self.temporal_chunk],memory)
            if self.recurrence_backend=='graph-v1':
                out,memory=graph_recurrence(*args)
            elif torch.is_grad_enabled():
                out,memory=checkpoint(recurrence,*args,use_reentrant=False,determinism_check='default')
            else:out,memory=recurrence(*args)
            pieces.append(out)
        y=torch.cat(pieces,1).to(x.dtype)
        z=a.in_proj_z(x).reshape_as(y)
        y=(y.float()*torch.rsqrt(y.float().square().mean(-1,keepdim=True)+a.norm.variance_epsilon)).to(x.dtype)
        y=((y*a.norm.weight)*F.silu(z.float())).to(x.dtype)
        return a.out_proj(y.reshape(1,x.shape[1],-1)),(new_conv,memory)
