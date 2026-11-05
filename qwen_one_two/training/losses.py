"""A4 loss primitives and explicit within-example normalization on CUDA."""
from collections import defaultdict
from dataclasses import dataclass
import torch
from torch.nn import functional as F
from ..codec.neural import cuda_only, norm
from .inputs import check

COEFFICIENTS = {'native':1., 'gate':1., 'writer':1., 'structure':.5, 'pointer':1.,
                'exact':1., 'alignment':1., 'reconstruction':.5, 'contrast':.1,
                'status':1., 'feasibility':.5, 'noul':1., 'choice':1., 'score':1., 'effort':.5}
REFERENCE_COEFFICIENTS = {'reconstruction':1., 'contrast':.1}


def finite(x):
    cuda_only(x)
    check(bool(x.isfinite().all()), 'nonfinite loss input')
    return x.float()


def ce(logits, target):
    cuda_only(logits)
    # Structural masks may contain -inf; at least one class must remain finite.
    check(not bool(torch.isnan(logits).any() | torch.isposinf(logits).any()), 'invalid classification logits')
    check(bool(torch.isfinite(logits).any(-1).all()), 'fully masked classification')
    target = torch.as_tensor(target, device=logits.device, dtype=torch.long)
    value = F.cross_entropy(logits.reshape(-1,logits.shape[-1]).float(),target.reshape(-1))
    return finite(value)


def permitted_set(logits, approved):
    cuda_only(logits)
    check(logits.ndim == 1 and bool(approved) and len(set(approved)) == len(approved), 'approved set')
    check(all(type(i) is int and 0 <= i < len(logits) for i in approved), 'absent approved alternative')
    check(not bool(torch.isnan(logits).any() | torch.isposinf(logits).any()), 'invalid set logits')
    chosen = logits[list(approved)].float()
    check(bool(chosen.isfinite().all()), 'approved choice structurally masked')
    return finite(torch.logsumexp(logits.float(),0)-torch.logsumexp(chosen,0))


def binary(logits, target):
    logits = finite(logits)
    target = torch.as_tensor(target,device=logits.device,dtype=torch.float32)
    check(bool(((target >= 0)&(target <= 1)&target.isfinite()).all()), 'binary target')
    return F.binary_cross_entropy_with_logits(logits,target.expand_as(logits))


def regression(raw, *, point=None, interval=None):
    raw = finite(raw)
    check((point is None) != (interval is None), 'choose reviewed point or interval')
    if point is not None:
        check(0 <= point <= 1, 'point target range')
        return (raw-point).square().mean()
    a,b = interval
    check(0 <= a <= b <= 1 and b-a <= .20+1e-12, 'reviewed interval range/width')
    return ((a-raw).clamp_min(0).square()+(raw-b).clamp_min(0).square()).mean()


def exact_value(kind, mode, *, byte_logits=None, value=None, exponent_logits=None,
                length_logits=None, boolean_logits=None):
    """Copy/parse and references have pointer/mode losses, never duplicate bytes."""
    check(mode in {'construct','copy','parse'}, 'attribute mode')
    if mode != 'construct' or kind in {'reference','source_string'}:
        return None
    if kind == 'boolean':
        check(type(value) is bool, 'Boolean target')
        return binary(boolean_logits, float(value))
    if kind == 'int64':
        check(type(value) is int and -(1<<63) <= value < (1<<63), 'int64 target')
        check(byte_logits.shape == (8,256), 'int64 byte head')
        return ce(byte_logits,list(value.to_bytes(8,'big',signed=True)))
    if kind == 'decimal':
        coefficient, exponent = value
        check(type(exponent) is int and -18 <= exponent <= 18 and exponent_logits.shape == (37,), 'decimal exponent')
        coefficient_loss = exact_value('int64','construct',byte_logits=byte_logits,value=coefficient)
        return (coefficient_loss+ce(exponent_logits,exponent+18))/2
    check(kind == 'utf8' and isinstance(value,bytes) and len(value) <= 128, 'UTF8 target')
    value.decode('utf-8')
    check(length_logits.shape == (129,), 'UTF8 length head')
    length_loss = ce(length_logits,len(value))
    if not value:
        return length_loss
    check(byte_logits.shape == (len(value),256), 'active UTF8 bytes only')
    return (length_loss+ce(byte_logits,list(value)))/2


def alignment(predicted, reference):
    check(predicted.shape == reference.shape, 'semantic slot shape')
    return (norm(predicted)-norm(reference.detach())).square().mean()


def contrast(anchor, candidates, positives, negatives):
    """Only reviewed same-role/shape relations, no incidental batch negatives."""
    cuda_only(anchor); cuda_only(candidates)
    if not positives or not negatives:
        return None
    check(candidates.shape[1:] == anchor.shape, 'contrast shape/role group')
    check(not set(positives)&set(negatives), 'conflicting contrast relation')
    selected = list(positives)+list(negatives)
    check(len(set(selected)) == len(selected) and all(0 <= i < len(candidates) for i in selected), 'contrast relation indices')
    a = F.normalize(norm(anchor).flatten(),dim=0)
    c = F.normalize(norm(candidates[selected].detach()).flatten(1),dim=1)
    logits = (c*a).sum(-1)/.1
    return torch.logsumexp(logits,0)-torch.logsumexp(logits[:len(positives)],0)


def native_tokens(logits, targets, kinds, approved):
    check(len(logits) == len(targets) == len(kinds) == len(approved), 'native target lengths')
    selected = [i for i,(kind,valid) in enumerate(zip(kinds,approved))
                if valid and kind in {'native','predicted_termination'}]
    if not selected:
        return None
    return ce(logits[selected],targets[selected])


@dataclass(frozen=True)
class Eligibility:
    supported_packet: bool
    reference_status: str | None
    actual_status: str | None
    operation: str
    reason_supported: bool = False
    runtime_failure: bool = False

    def __post_init__(self):
        statuses = {'Assessed','Insufficient','Conflicting','Unresolved','NoFeasibleChoice',None}
        check(self.operation in {'Noul','Choice','Score'} and self.reference_status in statuses
              and self.actual_status in statuses, 'eligibility operation/status')
        check(self.operation == 'Choice' or 'NoFeasibleChoice' not in (self.reference_status,self.actual_status), 'Choice-only eligibility')

    def mask(self):
        semantic = self.supported_packet and not self.runtime_failure and self.reference_status is not None
        assessed = semantic and self.reference_status == 'Assessed'
        return {'status':semantic, 'noul':assessed and self.operation == 'Noul',
                'choice':assessed and self.operation == 'Choice',
                'score':assessed and self.operation == 'Score',
                'feasibility':semantic and self.operation == 'Choice',
                'reason':semantic and self.reason_supported and self.reference_status == self.actual_status
                         and self.actual_status in {'Insufficient','Conflicting','Unresolved','NoFeasibleChoice'}}


class ExampleLosses:
    """Average subcomponents per field, then fields per role, then roles.

    Semantic components must carry their role; this prevents 16 alternatives
    from outweighing one question. Eligibility counts belong to the window.
    """
    def __init__(self):
        self.fields = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    def add(self, component, field, value, *, role=None):
        check(component in COEFFICIENTS, 'unknown loss component')
        if value is None:
            return
        check(value.ndim == 0 and bool(value.isfinite()), 'finite scalar component')
        cuda_only(value)
        if component in {'alignment','reconstruction','contrast'}:
            check(role is not None, 'semantic role normalization required')
        self.fields[component][role][field].append(value)

    def finish(self):
        return {component:torch.stack([
            torch.stack([torch.stack(values).mean() for values in fields.values()]).mean()
            for fields in roles.values()]).mean() for component,roles in self.fields.items()}
