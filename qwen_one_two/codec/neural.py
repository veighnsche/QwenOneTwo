"""CUDA-only full codec, parallel writer heads and Qev readouts (no trainer)."""
import os
import torch
from torch import nn
from torch.nn import functional as F
from .schema import Role, Metadata, Atom, Port, Status, bits, require
from ..g1_backbone import GPU_UUID
from ..g1_device import require_cuda
from .layout import LEGACY,DIRECT,DIRECT_ROLES,valid_profile


def cuda_only(x=None):
    try:require_cuda(torch.cuda,x)
    except RuntimeError as exc:raise ValueError(str(exc)) from exc


def finite(x,label='codec value'):
    cuda_only(x)
    require(bool(x.isfinite().all()),'nonfinite '+label)
    return x


def finite_linear_output(module,inputs,output):
    # Validation only: preserve the identical output tensor/autograd graph.
    finite(output,'codec linear output')


def norm(x):
    cuda_only(x)
    x=x.float();m=x.square().mean(-1,keepdim=True)
    finite(m,'codec normalization magnitude')
    return x/torch.sqrt(torch.where(m>1e-12,m,torch.full_like(m,1e-12)))


def thinking_budget(raw):
    cuda_only(raw)
    require(raw.dtype==torch.float32 and raw.numel()==1 and bool(raw.isfinite().all()),'finite FP32 effort score')
    s=raw.detach().double().clamp(0,1)
    u=s*s;v=3.*u;w=2.*u;z=w*s;curve=v-z;scaled=curve*131072.
    return (scaled+.5).floor().to(torch.int64)


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.q=nn.Linear(256,256);self.k=nn.Linear(256,256)
        self.v=nn.Linear(256,256);self.o=nn.Linear(256,256)
        self.ff1=nn.Linear(256,1024);self.ff2=nn.Linear(1024,256)

    def forward(self,q,m):
        cuda_only(q)
        one=q.ndim==1;q=q.reshape(-1,256)
        if m.numel():
            qq=self.q(norm(q)).reshape(-1,8,32).transpose(0,1)[None]
            kk=self.k(norm(m)).reshape(-1,8,32).transpose(0,1)[None]
            vv=self.v(norm(m)).reshape(-1,8,32).transpose(0,1)[None]
            a=F.scaled_dot_product_attention(qq,kk,vv,dropout_p=0).squeeze(0).transpose(0,1).reshape(-1,256)
            u=q+self.o(a)
        else:u=q  # bypass output bias for empty memory
        out=u+self.ff2(F.silu(self.ff1(norm(u))))
        return out[0] if one else out


class ReferencePooler(nn.Module):
    """Untrained offline module; reference preparation/training is out of scope."""
    def __init__(self):
        cuda_only()
        require(torch.get_default_device().type=='cuda','construct reference pooler on CUDA')
        super().__init__()
        self.memory=nn.Linear(1024,256)
        self.blocks=nn.ModuleList((Block(),Block()))
        self.queries=nn.Parameter(torch.empty(30,256))
        self.lift=nn.Linear(256,1024)
        self.apply(Codec._init)
        nn.init.normal_(self.queries,std=.02)

    def forward(self,states,start,count):
        cuda_only(states)
        require(states.ndim==2 and states.shape[1]==1024 and 1<=len(states)<=256,'reference annotation bound')
        require(type(start) is int and type(count) is int and 0<=start<start+count<=30,'reference slot query')
        q=self.queries[start:start+count];m=self.memory(states.float())
        for block in self.blocks:q=block(q,m)
        return norm(self.lift(norm(q)))


class DirectRoleProjection(nn.Module):
    """One shared FP32 residual owner; zero delta is exact identity initialization."""
    def __init__(self):
        super().__init__();cuda_only()
        self.weight=nn.Parameter(torch.zeros(1024,1024,device='cuda',dtype=torch.float32))
        self.bias=nn.Parameter(torch.zeros(1024,device='cuda',dtype=torch.float32))

    def forward(self,x):
        finite(x,'direct role state')
        return finite(x.float()+F.linear(x.float(),self.weight,self.bias),'direct role projection')


class DirectRoleLengths(nn.Module):
    def __init__(self):
        super().__init__();cuda_only()
        self.weight=nn.Parameter(torch.zeros(33,256,device='cuda',dtype=torch.float32))
        self.bias=nn.Parameter(torch.zeros(33,device='cuda',dtype=torch.float32))

    def forward(self,x):return finite(F.linear(x.float(),self.weight,self.bias),'direct length logits')


class Codec(nn.Module):
    def __init__(self,sigma,embedding,pointer):
        cuda_only(embedding)
        require(torch.get_default_device().type=='cuda','construct full codec inside CUDA device context')
        super().__init__()
        self.register_buffer('sigma',sigma.detach().clone().float())
        self.role=nn.Embedding(64,256,padding_idx=0);self.slot=nn.Embedding(128,256)
        self.owner=nn.Embedding(256,256);self.attribute=nn.Embedding(8,256);self.previous=nn.Embedding(7,256)
        self.tags=nn.Linear(256,1024)
        self.meta=nn.Sequential(nn.Linear(128,256),nn.SiLU(),nn.Linear(256,1024))
        self.source=nn.Sequential(nn.Linear(1024,256),nn.SiLU(),nn.Linear(256,1024))
        self.atom=nn.Sequential(nn.Linear(67,256),nn.SiLU(),nn.Linear(256,1024))
        self.value=nn.Sequential(nn.Linear(3,128),nn.SiLU(),nn.Linear(128,1024))
        self.budget_value=nn.Sequential(nn.Linear(3,128),nn.SiLU(),nn.Linear(128,1024))
        self.control=nn.Linear(1032,256);self.gate=nn.Linear(256,4)
        self.draft_in=nn.Linear(1024,256);self.reader=Block();self.writer_action=nn.Linear(256,7)
        self.feedback=nn.Linear(256,1024)
        self.emit1=nn.Linear(256,512);self.emit2=nn.Linear(512,256);self.emitout=nn.Linear(256,1024)
        sizes={'record_mode':2,'evidence_kind':2,'child_count':9,'attribute_type':6,
               'presence':1,'source_mode':3,'utf8_length':129,'span_length':33,
               'decimal_exponent':37,'direction':3,'attachment_kind':3,'boolean':1}
        self.heads=nn.ModuleDict({n:nn.Linear(256,k) for n,k in sizes.items()})
        self.bytes=nn.Sequential(nn.Linear(256,256),nn.SiLU(),nn.Linear(256,256))
        self.pointers=nn.ModuleDict({n:nn.Linear(256,256,bias=False) for n in
            ('reuse','source','history','copy','unit','reference','span_start','clip_start','clip_end')})
        self.pointer_key=nn.Linear(1024,256,bias=False)
        self.boundary_key=nn.Sequential(nn.Linear(272,256),nn.SiLU(),nn.Linear(256,256))
        self.qev_q=nn.Linear(1024,256,bias=False);self.qev_k=nn.Linear(1024,256,bias=False)
        self.eval_trunk=nn.Linear(1024,256);self.status=nn.Linear(256,5)
        self.feasibility=nn.Sequential(nn.Linear(1024,256),nn.SiLU(),nn.Linear(256,4))
        self.budget_row=nn.Parameter(torch.zeros(1024));self.score_row=nn.Parameter(torch.zeros(1024))
        self.prototypes=nn.Parameter(embedding[[3721,1802]].detach().float().clone())
        self.apply(self._init)
        with torch.no_grad():
            self.role.weight[0].zero_()
            self.qev_q.weight.copy_(pointer['q.weight']);self.qev_k.weight.copy_(pointer['k.weight'])

    @staticmethod
    def _init(module):
        if isinstance(module,nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:nn.init.zeros_(module.bias)
            module.register_forward_hook(finite_linear_output)
        elif isinstance(module,nn.Embedding):nn.init.normal_(module.weight,std=.02)

    def enable_direct_roles(self):
        require(not hasattr(self,'direct_role_projection'),'direct role owners already constructed')
        self.direct_role_projection=DirectRoleProjection()
        self.direct_role_length=DirectRoleLengths()
        return self

    def require_profile(self,profile):
        valid_profile(profile)
        require(profile!=DIRECT or all(hasattr(self,name) for name in ('direct_role_projection','direct_role_length')),
                'direct role modules require explicit construction')

    def semantic_length_logits(self,f,role):
        self.require_profile(DIRECT)
        require(role in DIRECT_ROLES,'direct emission role')
        logits=self.direct_role_length(self.condition(f,role)).clone();logits[0]=-torch.inf
        return logits

    def semantic_count(self,f,role):
        return int(self.semantic_length_logits(f,role).argmax())

    def atom_features(self,a):
        if a.role in (Role.INT64,Role.COEFFICIENT,Role.EXPONENT):
            require(a.coefficient is not None,'missing canonical numeric helpers')
            if a.role==Role.EXPONENT:require(a.data==bytes(7)+bytes([a.exponent&255]),'exponent atom mismatch')
            else:require(int.from_bytes(a.data,'big',signed=True)==a.coefficient,'numeric atom mismatch')
        raw=torch.tensor(list(a.data),device='cuda',dtype=torch.int64)
        shifts=torch.arange(7,-1,-1,device='cuda',dtype=torch.int64)
        binary=((raw[:,None]>>shifts)&1).flatten().float()
        helpers=torch.zeros(3,device='cuda',dtype=torch.float32)
        if a.coefficient is not None:
            n=torch.tensor(a.coefficient,device='cuda',dtype=torch.int64)
            mag=torch.where(n<0,-n,n)  # INT64_MIN keeps its unsigned bit pattern
            shifts64=torch.arange(64,device='cuda',dtype=torch.int64)
            k=(((mag>>shifts64)&1)*shifts64).max()
            remainder=mag-(torch.ones((),device='cuda',dtype=torch.int64)<<k)
            denom=torch.ldexp(torch.ones((),device='cuda',dtype=torch.float64),k.to(torch.int32))
            fraction=remainder.double()/denom
            magnitude=(k.double()+1)+fraction
            magnitude=magnitude/64
            magnitude=torch.where(n==0,torch.zeros_like(magnitude),magnitude)
            exponent=torch.tensor(a.exponent,device='cuda',dtype=torch.float64)/18
            helpers=torch.stack((n.sign().float(),magnitude.float(),exponent.float()))
        return torch.cat((binary,helpers))

    def wire(self,payload,role,slot=None,owner=None,attribute=None,previous=None):
        if int(role)==0:return self.sigma.new_zeros(1024)
        Role(role)
        require(role not in (Role.CONSUME,Role.CLOSE,Role.TERMINATE),'control role not emitted in v1')
        tag=self.role.weight[int(role)]
        for table,index in ((self.slot,slot),(self.owner,owner),(self.attribute,attribute),(self.previous,previous)):
            if index is not None:
                require(type(index) is int and 0<=index<table.num_embeddings,'tag index')
                tag=tag+table.weight[index]
        return self.sigma*norm(norm(payload)+norm(self.tags(tag)))

    def position(self,p,profile=LEGACY):
        if isinstance(p.payload,Metadata):
            payload=self.meta(torch.tensor(p.payload.encode(p.role),device='cuda',dtype=torch.float32))
        elif isinstance(p.payload,Atom):payload=self.atom(self.atom_features(p.payload))
        elif p.role in (Role.SOURCE,Role.ATTRIBUTE_SOURCE):payload=self.source(p.payload.float())
        elif p.role in (Role.NOUL,Role.SCORE,Role.BUDGET):
            v=p.payload.float()
            if p.role==Role.BUDGET:v=v.detach().clamp(0,1)
            payload=(self.budget_value if p.role==Role.BUDGET else self.value)(torch.stack((torch.ones_like(v),v,v*v)))
        else:payload=p.payload
        if profile==DIRECT and p.role in DIRECT_ROLES:payload=self.direct_role_projection(payload)
        cuda_only(payload)
        return self.wire(payload,p.role,p.slot,p.owner,p.attribute,p.previous)

    def encode(self,packet):
        self.require_profile(packet.profile)
        require(all(not isinstance(p.payload,Metadata) or p.payload.profile==packet.profile for p in packet.positions),
                'packet metadata layout mismatch')
        require(len(packet)>0,'empty packet')
        require(all(p.role!=Role.PAD for p in packet.positions),'padding is outside real packet positions')
        return torch.stack([self.position(p,packet.profile) for p in packet.positions])

    def forward(self,packet):
        return self.encode(packet)

    def encode_attribute_view(self,registry,owner,index):
        from .render import Renderer,Packet
        view=Renderer(registry).object(owner)
        positions=view.positions
        start=next((i for i,p in enumerate(positions) if p.role==Role.ATTRIBUTE_DESC and p.attribute==index),None)
        require(start is not None,'attribute view path')
        count=1+positions[start].payload.following
        return self.encode(Packet(positions[start:start+count],view.handles,None,profile=view.profile))

    def feature(self,h,counters,memory=None):
        f=F.silu(self.control(torch.cat((norm(h),counters))))
        return f if memory is None else self.reader(f,self.draft_in(memory.float()))

    def condition(self,f,role,slot=None,attribute=None):
        u=f+self.role.weight[int(role)]
        if slot is not None:u=u+self.slot.weight[slot]
        if attribute is not None:u=u+self.attribute.weight[attribute]
        return u

    def emit(self,f,role,count,attribute=None):
        out=[]
        for i in range(count):
            u=self.condition(f,role,i,attribute)
            v=u+self.emit2(F.silu(self.emit1(norm(u))))
            out.append(norm(self.emitout(norm(v))))
        return torch.stack(out)

    def byte_logits(self,f,count,attribute=None):
        require(0<=count<=128,'byte cap')
        return self.bytes(torch.stack([self.condition(f,Role.UTF8,i,attribute) for i in range(count)])) if count else f.new_empty(0,256)

    def pointer(self,name,f,keys,optional=False,mask=None):
        k=self.pointer_key(keys)
        if optional:k=torch.cat((k.new_zeros(1,256),k))
        require(len(k)>0,'empty required pointer')
        logits=(k*self.pointers[name](f)).sum(-1)/16
        finite(logits,'pointer logits')
        if mask is not None:
            require(mask.shape==logits.shape and bool(mask.any()),'pointer mask')
            logits=logits.masked_fill(~mask,-torch.inf)
        return logits

    def clipping_keys(self,token_key,offsets):
        require(all(type(x) is int and 0<=x<=65535 for x in offsets),'clipping offsets')
        binary=torch.tensor([bits([(0,16,x)],16) for x in offsets],device='cuda',dtype=torch.float32)
        return self.boundary_key(torch.cat((token_key.expand(len(offsets),256),binary),-1))

    def evaluate(self,h,port,candidate_states=None):
        f=F.silu(self.eval_trunk(norm(h)))
        status=self.status(f)
        if port!=Port.CHOICE:status=status.clone();status[4]=-torch.inf
        result={'status_logits':status,'reason_slots':self.emit(f,Role.REASON,4),'feature':f}
        q=self.qev_q(h.float())
        if port==Port.NOUL:
            logits=(self.qev_k(self.prototypes)*q).sum(-1)/16
            finite(logits,'Noul logits')
            raw=logits[1]-logits[0]
            finite(raw,'Noul truth logit')
            result.update(value=raw.sigmoid(),raw=raw)
        elif port==Port.SCORE:
            raw=self.table.scalar(248078,h)
            finite(raw,'raw task Score')
            result.update(value=raw.clamp(0,1),raw=raw)
        else:
            require(candidate_states is not None and 1<=len(candidate_states)<=16,'candidate boundary states')
            k=self.qev_k(candidate_states.float())
            result['pointer_logits']=(k*q).sum(-1)/16
            finite(result['pointer_logits'],'Choice logits')
            result['feasibility_logits']=self.feasibility(torch.cat((q.expand_as(k),k,q*k,q-k),-1))
        return result

    def startup(self,score):
        from .render import Packet,Position
        md=Metadata(Port.BUDGET,value_kind=7,present=True,sealed=False,following=2,complete=3)
        end=Metadata(Port.BUDGET,sealed=False)
        return Packet((Position(Role.BUDGET_HEADER,md),Position(Role.BUDGET,score,None,0),
                       Position(Role.BUDGET_END,end)),(),None)
