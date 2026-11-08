"""Exact parameter owners, usable for memory qualification without training."""
from torch import nn
from .schema import require


def parameter_groups(backbone,codec):
    groups=[];seen=set()
    def add(name,items,lr,decay):
        entries=list(items)
        for n,p in entries:
            require(p.is_cuda and p.requires_grad and id(p) not in seen,'parameter owner/device')
            seen.add(id(p))
        groups.append({'name':name,'named_parameters':entries,'lr':lr,'weight_decay':decay})
    bp=dict(backbone.named_parameters())
    for mode,lr in (('R',1e-4),('D',2e-5)):
        ids={id(p) for p in backbone.adapter_parameters(mode)}
        add(mode,((n,p) for n,p in bp.items() if id(p) in ids),lr,0.)
    add('Qev_pointer',[(f'codec.qev_{side}.weight',getattr(codec,f'qev_{side}').weight) for side in ('q','k')],2e-5,0.)
    add('budget',[('codec.budget_row',codec.budget_row)],1e-4,0.)
    add('score',[('codec.score_row',codec.score_row)],1e-4,0.)
    for owner in ('direct_role_projection','direct_role_length'):
        if not hasattr(codec,owner):continue
        for suffix,matrix in (('matrix',True),('bias',False)):
            add(owner+'_'+suffix,[(f'codec.{owner}.{n}',p) for n,p in getattr(codec,owner).named_parameters()
                if (p.ndim==2)==matrix],2e-4,.01 if matrix else 0.)
    embeddings={id(p) for m in codec.modules() if isinstance(m,nn.Embedding) for p in m.parameters()}
    matrix=[];other=[]
    for n,p in codec.named_parameters():
        if id(p) in seen:continue
        (matrix if p.ndim==2 and id(p) not in embeddings and n!='prototypes' else other).append(('codec.'+n,p))
    add('new_matrix',matrix,2e-4,.01);add('new_other',other,2e-4,0.)
    expected={id(p) for p in list(backbone.parameters())+list(codec.parameters()) if p.requires_grad}
    require(seen==expected,'missing parameter owner')
    return groups
