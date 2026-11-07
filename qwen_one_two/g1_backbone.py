"""Isolated CUDA G1 backend. No service imports, PEFT switches or hub kernels.

Uses the installed Qwen layer modules/weights and its token-ordered gated-delta
equations, with explicit immutable state transitions. Checkpoint functions are
pure; registry/ledger updates live outside them. This is a proof implementation,
not an alternative deployment or a completed semantic codec.
"""
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from safetensors.torch import load_file
from transformers import AutoConfig, AutoTokenizer
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5ForCausalLM, Qwen3_5TextRotaryEmbedding, apply_rotary_pos_emb,
)
from .g1_precision_math import compensated_linear
from .g1_device import PRIMARY as GPU_UUID, require_cuda

MODE = ContextVar('g1_adapter', default='R')
TARGETS = {'q_proj','k_proj','v_proj','o_proj','in_proj_qkv','in_proj_z',
           'in_proj_b','in_proj_a','out_proj','gate_proj','up_proj','down_proj'}


class EffectiveTable:
    """One functional tied-table provider; leaves stay registered only in Codec.

    Never materialize a concatenated trainable full table: it would turn unused
    scalar leaves into zero-grad rather than None-grad optimizer participants.
    """
    def __init__(self,base,budget_row,score_row):
        self.base=base;self.replacements={248077:budget_row,248078:score_row}
        self.compensated=False

    def row(self,index):
        value=self.replacements.get(index)
        return self.base[index] if value is None else value.to(torch.bfloat16)

    def scalar(self,index,hidden):
        return (hidden.float()*self.row(index).float()).sum(-1)

    def embed(self,ids):
        if bool(((ids==248077)|(ids==248078)).any()):
            raise ValueError('Private scalar indices are input-only typed ports, not text tokens')
        return F.embedding(ids,self.base)

    def project(self,hidden,indices):
        assert not bool(((indices==248077)|(indices==248078)).any())
        if self.compensated and self.base.dtype==torch.bfloat16:
            return compensated_linear(hidden,self.base[indices])
        return F.linear(hidden.to(self.base.dtype),self.base[indices]).float()


def norm(x):
    x = x.float()
    return x / x.square().mean(-1, keepdim=True).clamp_min(1e-12).sqrt()


class DualLinear(nn.Module):
    def __init__(self, base, a, b):
        super().__init__()
        self.base = base
        self.R_A = nn.Parameter(torch.empty(8, base.in_features, device='cuda'))
        self.R_B = nn.Parameter(torch.zeros(base.out_features, 8, device='cuda'))
        nn.init.uniform_(self.R_A, -1/math.sqrt(base.in_features), 1/math.sqrt(base.in_features))
        self.D_A = nn.Parameter(a.float())
        self.D_B = nn.Parameter(b.float())
        self.capture = False
        self.capture_mode = 'D'
        self.evaluation_outputs = []
        self.effective_reference = False
        self.fp32_residual = False
        self.compensated = False

    def forward(self, x):
        mode = MODE.get()
        a, b = (self.R_A, self.R_B) if mode == 'R' else (self.D_A, self.D_B)
        if self.effective_reference:
            a=a.to(torch.bfloat16).float()
            b=b.to(torch.bfloat16).float()
        if self.compensated and self.base.weight.dtype==torch.bfloat16:
            update=compensated_linear(compensated_linear(x,a),b)*2.0
            base=compensated_linear(x,self.base.weight,self.base.bias)
        else:
            work=x.to(self.base.weight.dtype)
            update = F.linear(F.linear(work, a.to(work.dtype)), b.to(work.dtype)) * 2.0
            base=self.base(work)
        if self.capture and mode == self.capture_mode and update.requires_grad:
            self.evaluation_outputs.append(update)
        return base.float()+update.float() if self.fp32_residual else base+update


@dataclass(frozen=True)
class State:
    layers: tuple
    position: int = 0
    lineage: int = 1


class Backbone(nn.Module):
    def __init__(self, snapshot, qev):
        super().__init__()
        require_cuda(torch.cuda)
        snapshot, qev = Path(snapshot), Path(qev)
        self.tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
        config = AutoConfig.from_pretrained(snapshot, local_files_only=True).text_config
        assert config.hidden_size == 1024 and config.num_hidden_layers == 24
        config._attn_implementation = 'sdpa'
        with torch.device('meta'):
            model = Qwen3_5ForCausalLM(config)
        weights = load_file(str(snapshot/'model.safetensors-00001-of-00001.safetensors'), device='cuda')
        mapped = {k.replace('model.language_model.', 'model.'):v for k,v in weights.items()
                  if k.startswith('model.language_model.')}
        mapped['lm_head.weight'] = mapped['model.embed_tokens.weight']
        model.load_state_dict(mapped, strict=True, assign=True)
        model.tie_weights()
        # Nonpersistent RoPE buffers are not in the safetensors checkpoint.
        with torch.device('cuda'):
            model.model.rotary_emb = Qwen3_5TextRotaryEmbedding(config)
        model.requires_grad_(False).eval()
        self.model = model
        del weights, mapped
        adapters = load_file(str(qev/'adapter/adapter_model.safetensors'), device='cuda')
        self.duals = []
        for name, module in list(model.model.named_modules()):
            if name.rsplit('.',1)[-1] in TARGETS and isinstance(module, nn.Linear):
                parent_name, leaf = name.rsplit('.',1)
                prefix = 'base_model.model.' + name
                wrapped = DualLinear(module, adapters[prefix+'.lora_A.weight'], adapters[prefix+'.lora_B.weight'])
                setattr(model.model.get_submodule(parent_name), leaf, wrapped)
                self.duals.append(wrapped)
        assert len(self.duals)*2 == len(adapters), (len(self.duals),len(adapters))
        self.width = 1024
        # Frozen embedding RMS, on GPU; valid tokenizer rows only.
        self.register_buffer('sigma', model.model.embed_tokens.weight[:len(self.tokenizer)].float().square().mean().sqrt())
        self.checkpointing = False
        self.precision = torch.bfloat16
        self.base_dtypes = {id(p):p.dtype for p in model.parameters() if not p.requires_grad}
        self.physical_layer_calls = 0
        self.round_reference_inputs = False
        self.fp32_residual = False
        self.compensated = False

    def set_stability(self,enabled):
        self.fp32_residual=enabled
        for module in self.duals: module.fp32_residual=enabled

    def set_compensated(self,enabled):
        self.compensated=enabled
        self.set_stability(enabled)
        for module in self.duals: module.compensated=enabled
        if hasattr(self,'table'): self.table.compensated=enabled

    def set_precision(self, dtype):
        # Only frozen base storage changes outside any active graph. LoRA masters
        # remain FP32 and each forward casts through autograd to effective dtype.
        for p in self.model.parameters():
            if not p.requires_grad:
                p.data = p.data.to(self.base_dtypes[id(p)] if dtype==torch.bfloat16 else dtype)
        self.precision = dtype

    def embedding(self, ids):
        ids = torch.as_tensor(ids, device='cuda', dtype=torch.long).reshape(1,-1)
        if bool(((ids == 248077) | (ids == 248078)).any()):
            raise ValueError('Scalar rows cannot appear in native text')
        return self.table.embed(ids) if hasattr(self,'table') else self.model.model.embed_tokens(ids)

    def bind_rows(self,codec):
        provider=EffectiveTable(self.model.model.embed_tokens.weight,codec.budget_row,codec.score_row)
        self.table=provider
        codec.table=provider
        provider.compensated=self.compensated
        return provider

    def initial(self):
        cache_dtype=torch.float32 if self.fp32_residual else self.precision
        result = []
        for layer in self.model.model.layers:
            if layer.block_type == 'linear_attention':
                a = layer.linear_attn
                result.append((torch.zeros(1,a.conv_dim,a.conv_kernel_size,device='cuda',dtype=cache_dtype),
                               torch.zeros(1,a.num_v_heads,a.head_k_dim,a.head_v_dim,device='cuda',dtype=torch.float32)))
            else:
                a = layer.self_attn
                shape = (1,a.config.num_key_value_heads,0,a.head_dim)
                result.append((torch.empty(shape,device='cuda',dtype=cache_dtype),
                               torch.empty(shape,device='cuda',dtype=cache_dtype)))
        return State(tuple(result))

    def _linear_attention(self, a, x, old):
        conv, memory = old
        raw = a.in_proj_qkv(x).transpose(1,2)
        joined = torch.cat((conv,raw),-1)
        new_conv = joined[:,:,-a.conv_kernel_size:]
        mixed = F.silu(F.conv1d(joined[:,:,-(raw.shape[-1]+a.conv_kernel_size-1):],
                               a.conv1d.weight.to(raw.dtype), a.conv1d.bias, groups=a.conv_dim)).transpose(1,2)
        q,k,v = torch.split(mixed,[a.key_dim,a.key_dim,a.value_dim],-1)
        q = q.reshape(1,-1,a.num_k_heads,a.head_k_dim).float()
        k = k.reshape_as(q).float()
        v = v.reshape(1,-1,a.num_v_heads,a.head_v_dim).float()
        q = q * torch.rsqrt(q.square().sum(-1,keepdim=True)+1e-6)
        k = k * torch.rsqrt(k.square().sum(-1,keepdim=True)+1e-6)
        q = q / math.sqrt(a.head_k_dim)
        if a.num_v_heads != a.num_k_heads:
            q = q.repeat_interleave(a.num_v_heads//a.num_k_heads,2)
            k = k.repeat_interleave(a.num_v_heads//a.num_k_heads,2)
        beta = a.in_proj_b(x).sigmoid().float()
        g = -a.A_log.float().exp() * F.softplus(a.in_proj_a(x).float()+a.dt_bias.float())
        outputs = []
        for t in range(x.shape[1]):
            memory = memory * g[:,t].exp()[...,None,None]
            delta = (v[:,t]-(memory*k[:,t].unsqueeze(-1)).sum(-2))*beta[:,t].unsqueeze(-1)
            memory = memory + k[:,t].unsqueeze(-1)*delta.unsqueeze(-2)
            outputs.append((memory*q[:,t].unsqueeze(-1)).sum(-2))
        y = torch.stack(outputs,1).to(x.dtype)
        z = a.in_proj_z(x).reshape_as(y)
        y = (y.float()*torch.rsqrt(y.float().square().mean(-1,keepdim=True)+a.norm.variance_epsilon)).to(x.dtype)
        y = ((y*a.norm.weight)*F.silu(z.float())).to(x.dtype)
        return a.out_proj(y.reshape(1,x.shape[1],-1)), (new_conv,memory)

    def _attention(self, a, x, old, rope):
        shape = (1,x.shape[1],-1,a.head_dim)
        q,gate = a.q_proj(x).view(1,x.shape[1],-1,a.head_dim*2).chunk(2,-1)
        gate = gate.reshape(1,x.shape[1],-1)
        q = a.q_norm(q.reshape(shape)).transpose(1,2)
        k = a.k_norm(a.k_proj(x).reshape(shape)).transpose(1,2)
        v = a.v_proj(x).reshape(shape).transpose(1,2)
        q,k = apply_rotary_pos_emb(q,k,*rope)
        k = torch.cat((old[0],k),-2)
        v = torch.cat((old[1],v),-2)
        groups = a.num_key_value_groups
        kk,vv = k.repeat_interleave(groups,1),v.repeat_interleave(groups,1)
        positions = torch.arange(x.shape[1],device='cuda')+old[0].shape[-2]
        mask = torch.arange(k.shape[-2],device='cuda')[None,:] <= positions[:,None]
        y = F.scaled_dot_product_attention(q,kk,vv,attn_mask=mask,dropout_p=0,scale=a.scaling)
        y = y.transpose(1,2).reshape(1,x.shape[1],-1)*gate.sigmoid()
        return a.o_proj(y),(k,v)

    def _layer(self, index, mode, h, s0, s1, cos, sin):
        self.physical_layer_calls += 1
        token = MODE.set(mode)
        try:
            layer = self.model.model.layers[index]
            x = layer.input_layernorm(h)
            if layer.block_type == 'linear_attention':
                x,new = self._linear_attention(layer.linear_attn,x,(s0,s1))
            else:
                x,new = self._attention(layer.self_attn,x,(s0,s1),(cos,sin))
            h = h+x
            h = h+layer.mlp(layer.post_attention_layernorm(h))
            return h,*new
        finally:
            MODE.reset(token)

    def forward(self, state, mode, embeds):
        assert mode in ('R','D') and embeds.device.type=='cuda'
        x = embeds.reshape(1,-1,1024).to(torch.float32 if self.fp32_residual else self.precision)
        if self.precision==torch.float32 and self.round_reference_inputs:
            x=x.to(torch.bfloat16).float()
        positions = torch.arange(state.position,state.position+x.shape[1],device='cuda').view(1,1,-1).expand(3,1,-1)
        rope = self.model.model.rotary_emb(x,positions)
        new = []
        for i,old in enumerate(state.layers):
            if self.checkpointing and torch.is_grad_enabled():
                def fn(h,s0,s1,cos,sin,idx=i,adapter=mode):
                    return self._layer(idx,adapter,h,s0,s1,cos,sin)
                x,s0,s1 = checkpoint(fn,x,*old,*rope,use_reentrant=False,determinism_check='default')
            else:
                x,s0,s1 = self._layer(i,mode,x,*old,*rope)
            new.append((s0,s1))
        return self.model.model.norm(x), State(tuple(new),state.position+x.shape[1],state.lineage)

    def ordinary_logits(self, h, phase):
        # Slice private rows OUT of the graph; masked zero gradients would still
        # allow Adam momentum to move an unused row.
        w = self.model.model.embed_tokens.weight
        indices = torch.cat((torch.arange(248044,device='cuda'),
                             torch.tensor([248069 if phase==0 else 248046],device='cuda')))
        logits=self.table.project(h,indices) if hasattr(self,'table') else F.linear(h.to(w.dtype),w[indices]).float()
        return logits,indices

    def capture_evaluator(self, enabled, mode='D'):
        for m in self.duals:
            m.capture = enabled
            m.capture_mode = mode
            if enabled:
                m.evaluation_outputs.clear()

    def adapter_parameters(self, mode):
        for m in self.duals:
            yield getattr(m,mode+'_A')
            yield getattr(m,mode+'_B')

    def snapshot_state(self,state):
        return tuple(t.detach().clone() for pair in state.layers for t in pair)
