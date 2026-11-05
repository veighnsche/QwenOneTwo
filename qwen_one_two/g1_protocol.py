"""Exercised B1/F6 Noul/Score subset and independent bounded proof ledger."""
from dataclasses import dataclass, field
import math
import torch
from torch import nn
from torch.nn import functional as F
from .g1_backbone import norm


from .g1_accounting import Ledger, AdmissionError, bit_fields


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.q=nn.Linear(256,256);self.k=nn.Linear(256,256)
        self.v=nn.Linear(256,256);self.o=nn.Linear(256,256)
        self.ff1=nn.Linear(256,1024);self.ff2=nn.Linear(1024,256)

    def forward(self,q,m):
        if m.numel():
            qq=self.q(norm(q)).reshape(1,8,1,32)
            kk=self.k(norm(m)).reshape(-1,8,32).transpose(0,1).unsqueeze(0)
            vv=self.v(norm(m)).reshape(-1,8,32).transpose(0,1).unsqueeze(0)
            a=F.scaled_dot_product_attention(qq,kk,vv,dropout_p=0).reshape(256)
            u=q+self.o(a)
        else: u=q
        return u+self.ff2(F.silu(self.ff1(norm(u))))


class Codec(nn.Module):
    def __init__(self,sigma,embedding,pointer_weights):
        super().__init__()
        self.register_buffer('sigma',sigma.detach().clone())
        self.role=nn.Embedding(64,256);self.slot=nn.Embedding(128,256)
        self.owner=nn.Embedding(256,256);self.attribute=nn.Embedding(8,256)
        self.previous=nn.Embedding(7,256)
        self.tags=nn.Linear(256,1024)
        self.meta=nn.Sequential(nn.Linear(128,256),nn.SiLU(),nn.Linear(256,1024))
        self.source=nn.Sequential(nn.Linear(1024,256),nn.SiLU(),nn.Linear(256,1024))
        self.value=nn.Sequential(nn.Linear(3,128),nn.SiLU(),nn.Linear(128,1024))
        self.budget_value=nn.Sequential(nn.Linear(3,128),nn.SiLU(),nn.Linear(128,1024))
        self.control=nn.Linear(1032,256);self.gate=nn.Linear(256,4)
        self.draft_in=nn.Linear(1024,256);self.reader=Block()
        self.writer_action=nn.Linear(256,7);self.feedback=nn.Linear(256,1024)
        self.emit1=nn.Linear(256,512);self.emit2=nn.Linear(512,256)
        self.emitout=nn.Linear(256,1024)
        self.qev_q=nn.Linear(1024,256,bias=False);self.qev_k=nn.Linear(1024,256,bias=False)
        self.eval_trunk=nn.Linear(1024,256);self.status=nn.Linear(256,5)
        # Virtual tied-table provider: actual private row indices, no scalar MLP.
        # Native LM logits exclude these leaves entirely, preserving grad=None.
        self.budget_row=nn.Parameter(torch.zeros(1024))
        self.score_row=nn.Parameter(torch.zeros(1024))
        self.prototypes=nn.Parameter(embedding[[3721,1802]].detach().float().clone())
        self.apply(self._init)
        with torch.no_grad():
            self.role.weight[0].zero_()
            self.qev_q.weight.copy_(pointer_weights['q.weight'])
            self.qev_k.weight.copy_(pointer_weights['k.weight'])

    @staticmethod
    def _init(module):
        if isinstance(module,nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None: nn.init.zeros_(module.bias)
        if isinstance(module,nn.Embedding): nn.init.normal_(module.weight,std=.02)

    def wire(self,payload,role,slot=None,owner=None,previous=None):
        tag=self.role.weight[role]
        if slot is not None: tag=tag+self.slot.weight[slot]
        if owner is not None: tag=tag+self.owner.weight[owner]
        if previous is not None: tag=tag+self.previous.weight[previous]
        return self.sigma*norm(norm(payload)+norm(self.tags(tag)))

    def metadata(self,role,port,owner=None,length=0,kind=0,present=False,
                 item=None,header=False,complete=0,status=0,question=None,target=None,
                 rubric=None,evidence=0,attempt=1,midpoint=False,parameter_a=0,parameter_b=0,sealed=True):
        fields=[(0,4,1),(4,3,port),(7,3,status),(10,3,7 if present and header else 0),
                (15,4,kind),(19,1,present),(20,1,owner is not None),(21,1,item is not None),
                (32,8,owner or 0),(40,8,item or 0),(48,9,length)]
        if header:
            fields += [(25,1,question is not None),(26,1,target is not None),(28,1,rubric is not None),
                       (29,1,midpoint),(30,1,sealed),(64,8,question or 0),(72,8,target or 0),
                       (88,8,rubric or 0),(96,4,evidence),(109,5,attempt),(114,9,complete),
                       (123,2,1 if rubric is not None else 0)]
        else:
            fields += [(29,1,midpoint),(30,1,sealed),(92,2,1 if role==8 else 0),
                       (96,16,parameter_a),(112,16,parameter_b)]
        payload=self.meta(torch.tensor(bit_fields(fields),device='cuda',dtype=torch.float32))
        return self.wire(payload,role,owner=owner)

    def control_input(self,role,port,ledger,draft,previous=None,outcome=0):
        fields=[(0,4,1),(4,3,port),(7,1,ledger.phase),(8,18,ledger.budget-ledger.b),
                (26,19,ledger.answer-ledger.a),(45,7,64-ledger.h),(52,19,262144-ledger.position),
                (71,6,32-ledger.writer),(77,5,16-ledger.attempts),(82,6,32-ledger.records),
                (88,3,0 if previous is None else previous+1),(91,9,draft.get('length',0)),
                (100,6,ledger.records),(111,4,1 if 'evidence' in draft else 0),(121,4,outcome)]
        payload=self.meta(torch.tensor(bit_fields(fields),device='cuda',dtype=torch.float32))
        return payload

    def feature(self,h,counters,draft=None):
        f=F.silu(self.control(torch.cat((norm(h),counters))))
        if draft is None: return f
        m=self.draft_in(draft) if draft.numel() else draft.reshape(0,256)
        return self.reader(f,m)

    def emit(self,feature,role,count):
        outputs=[]
        for i in range(count):
            u=feature+self.role.weight[role]+self.slot.weight[i]
            v=u+self.emit2(F.silu(self.emit1(norm(u))))
            outputs.append(norm(self.emitout(norm(v))))
        return torch.stack(outputs)

    def packet(self,draft,port,result=None,detach_payload=False,clipping_end=0,clipping_start=0,attempt=1):
        # First encounter header X,Q,T,U (Score); then source S.
        q,t=1,2;u=3 if port==3 else None;s=4 if port==3 else 3
        result_owner=5 if port==3 else 4
        m=draft['evidence'].shape[0]
        length=(25+m if port==3 else 14+m) if result is None else (25 if port==3 else 14)
        owner=0 if result is None else result_owner
        out=[self.metadata(1 if result is None else 26,port,owner,length-1,
                           9 if result is None else 7,result is not None,header=True,complete=length,
                           status=0 if result is None else 1,question=q,target=t,rubric=u,
                           evidence=1 if result is None else 0,midpoint=port==3,attempt=attempt)]
        qs=draft['question'].detach() if detach_payload else draft['question']
        ts=draft['target'].detach() if detach_payload else draft['target']
        out += [self.wire(v,2,i,q) for i,v in enumerate(qs)]
        out += [self.metadata(3,port,t,2,2,True)]
        out += [self.wire(v,4,i,t) for i,v in enumerate(ts)]
        if result is None:
            out += [self.metadata(5,port,s,m,1,True,item=0,parameter_a=clipping_start,parameter_b=clipping_end)]
            out += [self.wire(self.source(v.float()),6,i,s) for i,v in enumerate(draft['evidence'])]
        if port==3:
            out += [self.metadata(8,port,u,10,6,True,midpoint=True)]
            for role in range(9,14):
                values=draft['rubric'][role-9]
                if detach_payload: values=values.detach()
                out += [self.wire(v,role,i,u) for i,v in enumerate(values)]
        if result is not None:
            vv=torch.stack((torch.ones_like(result),result,result.square()))
            out += [self.wire(self.value(vv),28 if port==3 else 27,0,result_owner)]
        out += [self.metadata(25 if result is None else 31,port,owner,kind=9 if result is None else 7)]
        assert len(out)==length
        return torch.stack(out)

    def draft_memory(self,draft,port):
        if 'question' not in draft:
            return torch.empty(0,1024,device='cuda')
        out=[self.wire(v,2,i,0) for i,v in enumerate(draft['question'])]
        out += [self.metadata(3,port,1,2,2,True,sealed=False)]
        out += [self.wire(v,4,i,1) for i,v in enumerate(draft['target'])]
        if 'evidence' in draft:
            m=len(draft['evidence'])
            out += [self.metadata(5,port,2,m,1,True,item=0,sealed=False)]
            out += [self.wire(self.source(v.float()),6,i,2) for i,v in enumerate(draft['evidence'])]
        if 'rubric' in draft:
            out += [self.metadata(8,port,3,10,6,True,midpoint=True,sealed=False)]
            for role in range(9,14):
                out += [self.wire(v,role,i,3) for i,v in enumerate(draft['rubric'][role-9])]
        return torch.stack(out)

    def scalar(self,h,port):
        if port==1:
            logits=(self.qev_k(self.prototypes)*self.qev_q(h.float())).sum(-1)/16
            return (logits[1]-logits[0]).sigmoid(),logits[1]-logits[0]
        raw=self.table.scalar(248078,h)
        return raw.clamp(0,1),raw

    def startup(self,raw):
        v=raw.clamp(0,1).detach()
        vv=torch.stack((torch.ones_like(v),v,v.square()))
        return torch.stack((self.metadata(32,4,length=2,present=True,header=True,complete=3,attempt=0,sealed=False),
                            self.wire(self.budget_value(vv),33,0),self.metadata(34,4,sealed=False)))
