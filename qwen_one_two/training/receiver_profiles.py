"""Named input contracts and the existing receiver ownership predicate."""
from .inputs import check

LEGACY = 'd-pooled-role-slots-v1'
PROFILES = {
    LEGACY: {'source_policy': 'current-d-source-v1', 'roles': 'frozen-reference-pooled-v1',
             'implementation': 'legacy', 'required_slices': []},
    'frozen-w-pooled-role-slots-v1': {'source_policy': 'frozen-w-source-v1',
             'roles': 'frozen-reference-pooled-v1', 'implementation': 'pending',
             'required_slices': ['RD-03', 'RD-05', 'RD-07']},
    'direct-role-states-v1': {'source_policy': 'frozen-w-source-v1',
             'roles': 'direct-role-states-v1', 'implementation': 'pending',
             'required_slices': ['RD-03', 'RD-04', 'RD-05', 'RD-07']},
}
PROFILES.update({
    'frozen-w-pooled-compact-v2': {'source_policy':'frozen-w-source-v1','roles':'frozen-reference-pooled-v1',
        'wire':'pooled-role-slots-v2','implementation':'compact-v2','required_slices':['RD-07-v2']},
    'direct-role-states-v2': {'source_policy':'frozen-w-source-v1','roles':'direct-role-states-v2',
        'wire':'direct-role-states-v2','implementation':'compact-v2','required_slices':['RD-07-v2']},
})

PREFIXES = ('role.', 'slot.', 'owner.', 'attribute.', 'tags.', 'meta.', 'source.', 'atom.', 'value.',
            'qev_q.', 'qev_k.', 'eval_trunk.', 'status.', 'feasibility.', 'emit1.', 'emit2.', 'emitout.',
            'heads.child_count.')
PREFIXES = PREFIXES+('direct_role_projection.',)


def direct_projection_owners():
    return [{'name':'direct_role_projection_'+suffix,'lr':2e-4,'weight_decay':decay,
        'parameters':[{'name':'codec.direct_role_projection.'+name,'shape':shape,'dtype':'torch.float32'}]}
        for suffix,name,shape,decay in [('matrix','weight',[1024,1024],.01),('bias','bias',[1024],0.)]]


def extend_receiver_groups(codec,groups,input_profile):
    """After restoring the exact legacy parent, add fresh opt-in owners only.

    Existing masters/moments are not relabeled as projection state. Future fits
    still need qualified initialization/resume and their separate authorization.
    """
    profile(input_profile,'component')
    if input_profile not in ('direct-role-states-v1','direct-role-states-v2'):return groups
    codec.enable_direct_roles()
    codec.direct_role_length.requires_grad_(False)
    parameters=dict(codec.named_parameters());extended=list(groups)
    for spec in direct_projection_owners():
        entries=[]
        for item in spec['parameters']:
            p=parameters[item['name'][6:]]
            check(p.is_cuda and p.requires_grad and list(p.shape)==item['shape'] and str(p.dtype)==item['dtype'],
                  'direct projection owner/device/shape')
            entries.append((item['name'],p))
        extended.append({k:v for k,v in spec.items() if k!='parameters'} | {'named_parameters':entries})
    names=[n for group in extended for n,_ in group['named_parameters']]
    ids=[id(p) for group in extended for _,p in group['named_parameters']]
    check(len(set(names))==len(names) and len(set(ids))==len(ids),'duplicate receiver owner')
    return extended


def profile(name, regime):
    check(name in PROFILES, 'unknown input profile')
    check(regime == 'component', 'input profile requires component regime; native wiring belongs to RD-06')
    return dict(PROFILES[name])


def receiver_parameter(group, name):
    return group == 'D' or (name.startswith('codec.') and
        (name[6:] in {'score_row', 'prototypes'} or name[6:].startswith(PREFIXES)))
