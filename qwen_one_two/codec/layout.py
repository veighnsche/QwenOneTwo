"""Versioned, model-free role layout and unchanged admission envelopes."""
from .schema import Kind, Port, Role, require, integer

LEGACY = 'b1-full-v1-compensated'
DIRECT = 'direct-role-states-v1'
REQUEST_LIMIT = 256
RETURN_LIMIT = 64
FIELD_LIMIT = 32
REASON_LENGTH = 4
REASON_EXTRA = 32
RUBRIC_ROLES = (Role.DIMENSION, Role.CONDITIONS, Role.ZERO, Role.ONE, Role.MIDPOINT)
DIRECT_ROLES = (Role.QUESTION, Role.TARGET, Role.CANDIDATE, Role.KEY)+RUBRIC_ROLES
GPU_CHECKS = {
    'RD04-GPU-IDENTITY-PROJECTION': 'test_identity_projection_and_legacy_wire',
    'RD04-GPU-DIRECT-ROUNDTRIP': 'test_direct_packets_returns_and_immutable_bindings',
    'RD04-GPU-DIRECT-GRADIENTS': 'test_live_projection_and_consumer_gradients',
}


def valid_profile(profile):
    require(profile in (LEGACY,DIRECT), 'unknown latent layout profile')
    return profile


def semantic_lengths(kind, profile, lengths=(), *, midpoint=False):
    valid_profile(profile)
    old={Kind.QUESTION:(8,),Kind.TARGET:(2,),Kind.ALTERNATIVE:(4,),Kind.RUBRIC:(2,2,2,2,2)}
    require(kind in old, 'semantic kind')
    if profile==LEGACY:
        require(lengths==(), 'legacy layout has fixed lengths; no implicit conversion')
        return old[kind]
    require(type(lengths) is tuple and len(lengths)==(5 if kind==Kind.RUBRIC else 1), 'explicit direct field lengths')
    for index,length in enumerate(lengths):
        integer(length,0 if kind==Kind.RUBRIC and index==4 and not midpoint else 1,FIELD_LIMIT,'direct field length')
    if kind==Kind.RUBRIC:require((lengths[4]>0)==midpoint, 'midpoint presence/length mismatch')
    return lengths


def attribute_cost(profile, key_length, body_length):
    valid_profile(profile);integer(body_length,0,32,'attribute body length')
    integer(key_length,2 if profile==LEGACY else 1,2 if profile==LEGACY else FIELD_LIMIT,'attribute key length')
    return 1+key_length+body_length


def packet_costs(port, question, target, *, rubric=(), candidates=(), evidence=(), target_attributes=0):
    """Direct-profile counts, including largest candidate and all abstentions.

    candidates are (role-state count, exact attribute position cost); evidence
    entries are complete descriptor+body presentation costs. No cropping/padding.
    """
    require(port in (Port.NOUL,Port.CHOICE,Port.SCORE), 'operation')
    integer(question,1,FIELD_LIMIT,'question length');integer(target,1,FIELD_LIMIT,'target length')
    require(bool(rubric)==(port==Port.SCORE), 'rubric applicability')
    require(bool(candidates)==(port==Port.CHOICE) and len(candidates)<=16, 'candidate applicability/count')
    require(len(evidence)<=8, 'evidence count')
    integer(target_attributes,0,4096,'target attribute positions')
    for length,cost in candidates:
        integer(length,1,FIELD_LIMIT,'candidate length');integer(cost,0,4096,'candidate attribute positions')
    for cost in evidence:integer(cost,2,65,'evidence presentation cost')
    if rubric:semantic_lengths(Kind.RUBRIC,DIRECT,rubric,midpoint=rubric[4]>0)
    rubric_cost=1+sum(rubric) if rubric else 0
    echo=1+question+1+target+rubric_cost
    request=echo+target_attributes+sum(evidence)+sum(1+n+a for n,a in candidates)+1
    assessed=echo+(max(1+n+a for n,a in candidates) if candidates else 1)+1
    nonassessed=echo+1+REASON_LENGTH+REASON_EXTRA+1
    return {'request':request,'assessed_return':assessed,'nonassessed_return':nonassessed,
            'return_max':max(assessed,nonassessed),'echo':echo,
            'candidate_ends':tuple(question+target+2+target_attributes+sum(evidence)+rubric_cost+
                sum(1+n+a for n,a in candidates[:i+1])-1 for i in range(len(candidates)))}


def admit(costs):
    require(costs['request']<=REQUEST_LIMIT, 'direct request exceeds 256 positions; never crop')
    require(costs['return_max']<=RETURN_LIMIT, 'direct worst-case return exceeds 64 positions')
    return costs
