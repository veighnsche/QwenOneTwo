"""Read-only diagnostic views: private module topology, shared frozen tensors.

Not a training, export or live-serving API. Importing this file loads no model.
"""
from copy import deepcopy
from .inputs import check


def clone_frozen(bundle):
    """Copy controls/tokenizer while preserving every registered tensor object.

    In particular functional_call must swap entries in PRIVATE parameter dicts;
    sharing a backbone module itself would race the live and reference adapters.
    Call before any forward creates activations or cache state.
    """
    tensors=[t for m in bundle for t in (*m.parameters(),*m.buffers())]
    check(all(not t.requires_grad and t.grad is None for t in tensors),'frozen tensors required')
    clone=deepcopy(bundle,{id(t):t for t in tensors})
    sharing=check_views(bundle,clone)
    return clone,sharing


def check_views(original,clone):
    names=0;unique={}
    for left,right in zip(original,clone,strict=True):
        lm=dict(left.named_modules());rm=dict(right.named_modules())
        check(lm.keys()==rm.keys(),'module topology changed')
        for key,a in lm.items():
            b=rm[key]
            check(a is not b and a._parameters is not b._parameters and a._buffers is not b._buffers,
                  'mutable module containers shared')
        for kind in ('named_parameters','named_buffers'):
            a=dict(getattr(left,kind)());b=dict(getattr(right,kind)())
            check(a.keys()==b.keys(),'tensor names changed')
            for key,t in a.items():
                check(t is b[key] and t.data_ptr()==b[key].data_ptr(),'frozen tensor duplicated')
                unique[id(t)]=t.numel()*t.element_size();names+=1
    b,c,ref=clone
    check(ref.backbone is b and c.table is b.table,'private backbone/table wiring')
    check(b.tokenizer is not original[0].tokenizer,'tokenizer state shared')
    check(b.table.precision_owner is b,'table precision owner escaped private view')
    for left,right in zip(original[0].duals,b.duals,strict=True):
        check(right.precision_owner is b and right.forward.__self__ is right,'dual bound to another view')
        check(left.evaluation_outputs is not right.evaluation_outputs,'mutable evaluation tape shared')
    check(b.model.model.rotary_emb.rope_type=='default','mutable dynamic RoPE unsupported')
    ref.assert_isolated(c)
    return {'shared_named_tensors':names,'unique_frozen_tensors':len(unique),
            'unique_frozen_bytes':sum(unique.values()),'private_module_containers':True,
            'private_tokenizer':True,'reference_live_isolated':True}
