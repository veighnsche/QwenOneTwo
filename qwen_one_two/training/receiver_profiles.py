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
PREFIXES = ('role.', 'slot.', 'owner.', 'attribute.', 'tags.', 'meta.', 'source.', 'atom.', 'value.',
            'qev_q.', 'qev_k.', 'eval_trunk.', 'status.', 'feasibility.', 'emit1.', 'emit2.', 'emitout.',
            'heads.child_count.')


def profile(name, regime):
    check(name in PROFILES, 'unknown input profile')
    check(regime == 'component', 'input profile requires component regime; native wiring belongs to RD-06')
    return dict(PROFILES[name])


def receiver_parameter(group, name):
    return group == 'D' or (name.startswith('codec.') and
        (name[6:] in {'score_row', 'prototypes'} or name[6:].startswith(PREFIXES)))
