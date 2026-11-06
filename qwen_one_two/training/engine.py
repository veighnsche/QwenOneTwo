"""F6 per-component VJPs and transactional A4 optimizer windows on CUDA.

No dataset launcher or capability-training loop. Callers must supply complete
supported episodes and reviewed targets; this module never invents labels.
"""
from collections import defaultdict
import torch
from torch import nn
from ..codec.neural import cuda_only
from ..codec.ownership import parameter_groups
from .inputs import check
from .losses import COEFFICIENTS, REFERENCE_COEFFICIENTS

from .profile import PHASES, schedule, scheduled_rate, guided_events


def reference_groups(reference):
    check(not reference.frozen, 'reference is frozen')
    t_ids = {id(p) for p in reference.T}
    embeddings = {id(p) for m in reference.modules() if isinstance(m,nn.Embedding) for p in m.parameters()}
    groups = {name:[] for name in ('T','reference_matrix','reference_other')}
    for name,p in reference.named_parameters():
        if not p.requires_grad:
            continue
        group = 'T' if id(p) in t_ids else ('reference_matrix' if p.ndim == 2 and id(p) not in embeddings
                 and not name.endswith(('queries','prototypes')) else 'reference_other')
        groups[group].append((name,p))
    return [{'name':name,'named_parameters':items,'lr':1e-4 if name == 'T' else 2e-4,
             'weight_decay':.01 if name == 'reference_matrix' else 0.} for name,items in groups.items()]


def live_groups(backbone, codec, reference=None):
    if reference is not None:
        check(reference.frozen, 'freeze/version reference before live optimization')
        reference.assert_isolated(codec)
    return parameter_groups(backbone, codec)


def ownership_manifest(groups):
    seen, names, manifest = set(), set(), []
    for group in groups:
        check(group['lr'] > 0 and group['weight_decay'] >= 0, 'optimizer hyperparameters')
        entries = []
        for name,p in group['named_parameters']:
            cuda_only(p)
            check(p.requires_grad and p.is_leaf and p.dtype == torch.float32, 'trainable FP32 CUDA masters required')
            check(id(p) not in seen and name not in names, 'duplicate optimizer ownership')
            seen.add(id(p)); names.add(name)
            entries.append({'name':name,'shape':list(p.shape),'dtype':str(p.dtype)})
        manifest.append({'name':group['name'],'lr':group['lr'],'weight_decay':group['weight_decay'],'parameters':entries})
    check(bool(seen), 'empty optimizer ownership')
    return manifest


def clone_state(value):
    if torch.is_tensor(value):
        return value.detach().clone()
    if isinstance(value,dict):
        return {k:clone_state(v) for k,v in value.items()}
    if isinstance(value,list):
        return [clone_state(v) for v in value]
    if isinstance(value,tuple):
        return tuple(clone_state(v) for v in value)
    return value


def tensors(value):
    if torch.is_tensor(value):
        yield value
    elif isinstance(value,dict):
        for item in value.values():
            yield from tensors(item)
    elif isinstance(value,(list,tuple)):
        for item in value:
            yield from tensors(item)


class Window:
    def __init__(self, groups, phase='components', *, schedule_profile='a4-v1'):
        cuda_only()
        check(phase in PHASES, 'phase')
        scheduled_rate(phase, 1, schedule_profile)
        self.schedule_profile = schedule_profile
        self.ownership = ownership_manifest(groups)
        self.parameters = [p for group in groups for _,p in group['named_parameters']]
        self.names = [name for group in groups for name,_ in group['named_parameters']]
        self.optimizer = torch.optim.AdamW([
            {'params':[p for _,p in g['named_parameters']], 'lr':g['lr'], 'peak_lr':g['lr'],
             'weight_decay':g['weight_decay'], 'name':g['name']} for g in groups],
            betas=(.9,.95),eps=1e-8,foreach=False,fused=False,capturable=True)
        self.phase, self.scheduled_windows, self.optimizer_updates = phase, 0, 0
        self.buffers, self.counts = {}, defaultdict(int)
        self.examples, self.failures = 0, []
        self.optimizer.zero_grad(set_to_none=True)

    @property
    def coefficients(self):
        return REFERENCE_COEFFICIENTS if self.phase == 'reference' else COEFFICIENTS

    def abort(self):
        self.buffers.clear(); self.counts.clear(); self.examples = 0; self.failures.clear()
        self.optimizer.zero_grad(set_to_none=True)

    def add_episode(self, losses, *, complete, positions, failure=None):
        """One completed episode graph at a time; no rollout replay/retention."""
        check(self.examples < 8, 'window already full')
        limit = 1024 if self.phase in {'guided_joint','transition','free_joint'} else 512
        check(type(positions) is int and positions >= 0, 'episode positions')
        if not complete or positions > limit or failure is not None:
            self.examples += 1
            self.failures.append(failure or ('overlength' if positions > limit else 'incomplete'))
            return
        try:
            check(set(losses) <= set(self.coefficients), 'component not eligible in this phase')
            for component, loss in losses.items():
                cuda_only(loss)
                check(loss.ndim == 0 and loss.requires_grad and bool(loss.isfinite()), 'invalid component loss')
            for index,(component,loss) in enumerate(losses.items()):
                grads = torch.autograd.grad(loss,self.parameters,allow_unused=True,
                                            retain_graph=index < len(losses)-1)
                check(all(g is None or bool(g.isfinite().all()) for g in grads), 'nonfinite component gradient')
                buffers = self.buffers.setdefault(component,{})
                for pi,g in enumerate(grads):
                    if g is not None:
                        if pi not in buffers:
                            buffers[pi] = g.detach().float().clone()
                        else:
                            buffers[pi].add_(g.detach().float())
                self.counts[component] += 1
            self.examples += 1
        except BaseException:
            self.abort()
            raise

    def combine(self):
        check(self.examples == 8, 'eight scheduled examples required')
        self.optimizer.zero_grad(set_to_none=True)
        for component,buffers in self.buffers.items():
            weight = self.coefficients[component]/self.counts[component]
            for pi,g in buffers.items():
                p = self.parameters[pi]
                if p.grad is None:
                    p.grad = g.mul(weight)
                else:
                    p.grad.add_(g,alpha=weight)
        check(all(p.grad is None or bool(p.grad.isfinite().all()) for p in self.parameters), 'nonfinite combined gradient')

    def step(self):
        multiplier = scheduled_rate(self.phase, self.scheduled_windows+1, self.schedule_profile)
        backup = None
        try:
            self.combine()
            report = {'phase':self.phase,'window':self.scheduled_windows+1,'eligible_examples':dict(self.counts),
                      'failed_exposures':list(self.failures),'active_parameters':[n for n,p in zip(self.names,self.parameters) if p.grad is not None]}
            active = [p for p in self.parameters if p.grad is not None]
            if active:
                for group in self.optimizer.param_groups:
                    group['lr'] = group['peak_lr']*multiplier
                check(all(t.is_cuda and bool(t.isfinite().all()) for state in self.optimizer.state.values()
                          for t in tensors(state)), 'invalid optimizer state')
                report['unclipped_norm'] = float(torch.nn.utils.clip_grad_norm_(active,1.,error_if_nonfinite=True))
                # Roll back a partial AdamW step on device/numeric failure. Save
                # only active masters/moments; unused rows remain grad=None.
                backup = [(p,p.detach().clone(),clone_state(self.optimizer.state.get(p))) for p in active]
                self.optimizer.step()
                check(all(bool(p.isfinite().all()) for p in active) and
                      all(t.is_cuda and bool(t.isfinite().all()) for p in active for t in tensors(self.optimizer.state[p])),
                      'nonfinite optimizer update')
                self.optimizer_updates += 1
            report['stepped'] = bool(active)
            self.scheduled_windows += 1
            self.abort()
            return report
        except BaseException:
            if backup is not None:
                with torch.no_grad():
                    for p,old,state in backup:
                        p.copy_(old)
                        if state is None:
                            self.optimizer.state.pop(p,None)
                        else:
                            self.optimizer.state[p] = state
            self.abort()
            raise

    def next_phase(self, phase, promotion):
        check(self.examples == 0 and not self.buffers, 'active episode/window')
        phases = list(PHASES)
        check(phases.index(phase) == phases.index(self.phase)+1, 'phase order')
        check(self.phase != 'reference', 'create fresh live optimizer after reference freeze')
        total = PHASES[self.phase]
        check(self.scheduled_windows == total, 'phase windows incomplete')
        check(promotion.get('phase') == self.phase and promotion.get('checkpoints') == [total-100,total]
              and promotion.get('passed') is True and bool(promotion.get('evidence_hash')), 'development promotion evidence required')
        self.phase, self.scheduled_windows = phase, 0
