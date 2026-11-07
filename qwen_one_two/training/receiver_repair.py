"""Opt-in readout repair; no production default changes or labels in forward."""
from contextlib import contextmanager
from .inputs import check, digest

PROTOCOL = 'receiver-residual-repair-v1'
UPDATES = 2000
WIDTH = 32


def training_ids(release):
    check(release.purpose == 'training', 'repair selection requires training release')
    ids = [e['id'] for e in release.manifest['records'] if e['selected']
           and release.record(e['id'])['training_category'] == 'receiver'
           and release.record(e['id'])['operation'] in ('Noul', 'Score')]
    check(len(ids) == len(set(ids)) == 1003, 'complete existing Noul/Score receiver corpus')
    return sorted(ids, key=lambda k: digest((PROTOCOL + ':17:' + k).encode()))


def visit_ids(ids, update):
    check(1 <= update <= UPDATES and len(ids) == 1003, 'declared repair schedule')
    return [ids[((update-1)*8+i) % len(ids)] for i in range(8)]


def make_repair():
    import torch
    from torch import nn
    from ..codec.neural import cuda_only, norm
    cuda_only()

    class Residual(nn.Module):
        def __init__(self):
            super().__init__()
            self.down = nn.Linear(1024, WIDTH, device='cuda', dtype=torch.float32)
            self.up = nn.Linear(WIDTH, 1024, device='cuda', dtype=torch.float32)
            nn.init.zeros_(self.up.weight)
            nn.init.zeros_(self.up.bias)

        def forward(self, h):
            cuda_only(h)
            return h.float() + self.up(torch.nn.functional.silu(self.down(norm(h))))

    return nn.ModuleDict({op: Residual() for op in ('Noul', 'Score')})


def repair_groups(repair):
    return [{'name': 'repair_' + kind, 'named_parameters': [
        ('repair.' + n, p) for n, p in repair.named_parameters()
        if (p.ndim == 2) == matrix], 'lr': 2e-4, 'weight_decay': .01 if matrix else 0.}
        for kind, matrix in (('matrix', True), ('bias', False))]


def readout(codec, repair, h, operation):
    """Original FP32 head equations on an operation-specific residual state."""
    import torch
    from torch.nn import functional as F
    from ..codec.neural import norm
    check(operation in ('Noul', 'Score'), 'Choice never enters a repair branch')
    x = repair[operation](h)
    status = codec.status(F.silu(codec.eval_trunk(norm(x)))).clone()
    status[4] = -torch.inf
    if operation == 'Noul':
        logits = (codec.qev_k(codec.prototypes)*codec.qev_q(x.float())).sum(-1)/16
        raw = logits[1]-logits[0]
        value = raw.sigmoid()
    else:
        # Preserve actual private LM-head row and its BF16-effective storage ABI.
        raw = codec.table.scalar(248078, x)
        value = raw.clamp(0, 1)
    return {'status_logits': status, 'raw': raw, 'value': value}


def losses(outputs, row):
    from .behavior_qualification import expected
    from .episodes import STATUS_NAMES
    from .losses import ce, binary, regression
    gold = expected(row)
    target = next(int(k)-1 for k, v in STATUS_NAMES.items() if v == gold['status'])
    result = {'status': ce(outputs['status_logits'], target)}
    if gold['status'] == 'Assessed':
        if row['operation'] == 'Noul':
            result['noul'] = binary(outputs['raw'], gold['payload']['p_true'])
        else:
            result['score'] = regression(outputs['raw'], interval=tuple(gold['payload']['score_interval']))
    return result


@contextmanager
def installed(codec, repair):
    """Adapt status/scalars; retain Choice and original Reason/structure.

    Requires the versioned parent plus repair, never a production default.
    """
    from ..codec.schema import Port
    prior_override = codec.__dict__.get('evaluate')
    original = codec.evaluate

    def evaluate(h, port, candidate_states=None):
        result = original(h, port, candidate_states)
        if port != Port.CHOICE:
            check(port in (Port.NOUL, Port.SCORE), 'supported receiver port')
            result.update(readout(codec, repair, h, 'Noul' if port == Port.NOUL else 'Score'))
        return result

    codec.evaluate = evaluate
    try:
        yield
    finally:
        if prior_override is None:
            del codec.evaluate
        else:
            codec.evaluate = prior_override
