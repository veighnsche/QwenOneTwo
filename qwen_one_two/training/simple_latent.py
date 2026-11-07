"""One read-only, fixed-capacity alternative to the learned reference pooler."""
from .inputs import check, SLOTS, digest, canonical
from .behavior_qualification import grade

PROFILE = 'ordered-full-width-means-v1'


def bins(length, count):
    """Disjoint ordered bins cover every original token, including template tokens."""
    check(type(length) is int and type(count) is int and 1 <= count <= length <= 256,
          'nonempty bounded token bins required')
    return tuple((i*length//count, (i+1)*length//count) for i in range(count))


def ordered_means(states, count):
    import torch
    from ..codec.neural import cuda_only, norm
    cuda_only(states)
    check(not torch.is_grad_enabled() and not states.requires_grad, 'read-only pooling only')
    check(states.ndim == 2 and states.shape[1] == 1024 and states.dtype == torch.float32
          and bool(states.isfinite().all()), 'finite FP32 source states required')
    return norm(torch.stack([states[a:b].mean(0) for a,b in bins(len(states), count)]))


def assert_input_match(case, control):
    """Targets/packet lengths may change delivery, never the supplied source view."""
    for name in ('id', 'record_sha256', 'view_sha256'):
        check(case.get(name) == control.get(name), 'comparison input drift: '+name)
    current, previous = case['runtime'], control['runtime']
    check(current['prefills'] == current['lineage'] == 1 and current['held_positions'] == 0,
          'single lineage and released holds required')
    check(current['total'] == sum(current[k] for k in ('P','B','A','H')), 'comparison accounting')
    for name in ('predicted_budget','granted_budget','override','P'):
        check(current[name] == previous[name], 'startup/budget drift')
    def prefix(summary):
        events = summary['events']; end = next(i for i,e in enumerate(events) if e['label']=='request')
        return events[:end+1]
    check(canonical(prefix(current)) == canonical(prefix(previous)), 'request position/framing drift')
    # D-dependent evidence states and exact original byte/offset lineage stay identical.
    before = [x for x in control['construction_audit'] if 'local_source' in x]
    after = [x for x in case['construction_audit'] if 'local_source' in x]
    check(canonical(before) == canonical(after), 'source evidence changed')


def paired_outcomes(release, controls, candidates):
    check([x['id'] for x in controls] == [x['id'] for x in candidates] and
          len({x['id'] for x in controls}) == len(controls), 'paired coverage/order')
    result = {}
    for op in ('Noul','Choice','Score'):
        pairs = [(a,b) for a,b in zip(controls,candidates) if release.record(a['id'])['operation']==op]
        counts = {'both_pass':0, 'control_only':0, 'candidate_only':0, 'both_fail':0}
        for a,b in pairs:
            x,y = (grade(release.record(v['id']),v)['passed'] for v in (a,b))
            counts['both_pass' if x and y else 'control_only' if x else 'candidate_only' if y else 'both_fail'] += 1
        result[op] = dict(cases=len(pairs), **counts)
    return result


def cached_encoder(backbone, reference, cache, roles, *, receiver, profile_hash, cache_sha, candidate):
    """Same permitted inputs, new explicitly named basis; never reference targets."""
    import torch
    from .clean import CleanEncoder

    class CachedEncoder(CleanEncoder):
        def __init__(self):
            check(not torch.is_grad_enabled() and all(not p.requires_grad for p in backbone.parameters())
                  and all(not p.requires_grad for p in reference.parameters()), 'frozen diagnostic encoder only')
            super().__init__(backbone, reference, checkpoint_hash=receiver, profile_hash=profile_hash)
            self.payloads = {}; self.offline_source_tokens = 0; self.used = set()

        def _states(self, ids, limit):
            self.offline_source_tokens += len(ids)
            return super()._states(ids, limit)

        def reference_payload(self, ref):
            check(not torch.is_grad_enabled(), 'read-only candidate cannot supply training targets')
            check(roles.get(ref.sha256) == ref.role, 'unknown/mismatched permitted reference')
            self.used.add(ref.sha256)
            if ref.sha256 not in self.payloads:
                self.payloads[ref.sha256] = (ordered_means(cache['reference.'+ref.sha256+'.pre'], SLOTS[ref.role][1])
                    if candidate else cache['reference.'+ref.sha256+'.pool'].clone())
            return self.payloads[ref.sha256].clone()

        def encode(self, view, provenance, *, run):
            encoded = super().encode(view, provenance, run=run)
            for ref, audit in zip(view.references, encoded['audit'][len(encoded['sources']):]):
                audit['transformations'] = (['role_whitelist','cached_frozen_W_states','ordered_bin_mean_fp32','canonical_norm']
                    if candidate else ['role_whitelist','cached_frozen_reference_pool'])
                audit['basis_profile'] = PROFILE if candidate else 'unchanged-frozen-reference'
                audit['reference_target_compatible'] = not candidate
                audit['capture_file_sha256'] = cache_sha
                audit['capture_tensor'] = 'reference.'+ref.sha256+('.pre' if candidate else '.pool')
                if candidate:
                    audit['token_bins'] = bins(len(cache[audit['capture_tensor']]), SLOTS[ref.role][1])
            return encoded

    return CachedEncoder()
