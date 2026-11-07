"""Read-only representation statistics. Tensor operations require the selected GPU.

No classifier, threshold, regression or other probe is fitted. Distances, spectra
and label-group summaries describe these samples; they do not certify semantics.
"""
from .inputs import check, canonical


def validate_replay(case, saved):
    for key in ('id', 'record_sha256', 'view_sha256', 'actual', 'predicted_status',
                'predicted_value', 'score', 'failure', 'transport', 'runtime'):
        # Live ledgers contain tuples; JSON receipts contain arrays. Compare the
        # complete canonical serialization, without dropping any ledger fields.
        check(canonical(case.get(key)) == canonical(saved.get(key)), 'instrumented replay drift: ' + key)
    check(case['failure'] is None and case['score']['transport_valid'], 'incomplete diagnostic case')


def difference(a, b):
    import torch
    from ..codec.neural import cuda_only
    cuda_only(a); cuda_only(b)
    shapes = [list(a.shape), list(b.shape)]
    if a.shape != b.shape:
        return {'shapes': shapes, 'same_shape': False, 'exact_equal': False,
                'relative_l2': None, 'cosine': None}
    x, y = a.double().flatten(), b.double().flatten()
    check(bool(torch.isfinite(x).all() & torch.isfinite(y).all()), 'nonfinite feature')
    nx, ny = x.norm(), y.norm()
    scale = torch.maximum(nx, ny).clamp_min(torch.finfo(torch.float64).tiny)
    return {'shapes': shapes, 'same_shape': True, 'exact_equal': torch.equal(a, b),
            'relative_l2': float((x-y).norm()/scale),
            'cosine': float((x@y)/(nx*ny)) if bool((nx>0)&(ny>0)) else None}


def spectrum(matrix, *, center=True):
    import torch
    from ..codec.neural import cuda_only
    cuda_only(matrix)
    check(matrix.ndim == 2 and min(matrix.shape) > 0 and bool(matrix.isfinite().all()), 'finite feature matrix')
    x = matrix.double()
    if center:
        x = x-x.mean(0, keepdim=True)
    singular = torch.linalg.svdvals(x)
    top = singular[0]
    cutoff = top * max(x.shape) * torch.finfo(torch.float32).eps
    rank = int((singular > cutoff).sum())
    energy = singular.square().sum()
    return {'shape': list(matrix.shape), 'centered': center,
            'singular_values': singular.tolist(), 'fp32_scale_rank': rank,
            'rank_cutoff': float(cutoff),
            'condition_retained': float(top/singular[rank-1]) if rank else None,
            'stable_rank': float(energy/top.square()) if bool(top>0) else 0.,
            'diagnostic_arithmetic': 'CUDA FP64 SVD of captured FP32 features; FP32-scale rank cutoff',
            'interpretation': 'sample geometry only; centered rank <= min(samples-1, features); no separability certificate'}


def readout(codec, h, operation, *, prototypes=None, score_row=None):
    """Existing readout equations with explicitly supplied frozen saved heads."""
    import torch
    check(not torch.is_grad_enabled(), 'read-only diagnostic requires no_grad')
    if operation == 'Noul':
        keys = codec.qev_k(codec.prototypes if prototypes is None else prototypes)
        logits = (keys*codec.qev_q(h.float())).sum(-1)/16
        raw = logits[1]-logits[0]
        return raw, raw.sigmoid()
    check(operation == 'Score', 'scalar diagnostic operation')
    row = codec.score_row if score_row is None else score_row
    raw = (h.float()*row.to(torch.bfloat16).float()).sum(-1)
    return raw, raw.clamp(0, 1)


def label_geometry(features, labels, raw):
    """Descriptive binary class statistics at the unchanged zero-logit threshold."""
    import torch
    from ..codec.neural import cuda_only
    cuda_only(features); cuda_only(labels); cuda_only(raw)
    check(features.ndim == 2 and labels.shape == raw.shape == (len(features),), 'binary summary shape')
    yes, no = labels.bool(), ~labels.bool()
    check(bool(yes.any() & no.any()), 'both reviewed classes required')
    x = features.double(); positive, negative = raw[yes], raw[no]
    center_yes, center_no = x[yes].mean(0), x[no].mean(0)
    within = torch.cat(((x[yes]-center_yes).norm(dim=1), (x[no]-center_no).norm(dim=1))).square().mean().sqrt()
    wins = positive[:, None] > negative[None, :]
    ties = positive[:, None] == negative[None, :]
    return {'positive': int(yes.sum()), 'negative': int(no.sum()),
            'threshold_correct': int((((raw>0)==yes) & (raw!=0)).sum()),
            'positive_logit_range': [float(positive.min()), float(positive.max())],
            'negative_logit_range': [float(negative.min()), float(negative.max())],
            'positive_above_all_negative': bool(positive.min()>negative.max()),
            'auc': float((wins.double()+.5*ties.double()).mean()),
            'centroid_distance': float((center_yes-center_no).norm()), 'within_class_rms': float(within),
            'scope': 'descriptive correlated samples; no fitted threshold, classifier or significance claim'}


def capture_encoder(backbone, reference, *, checkpoint_hash, profile_hash):
    """Instrument existing CleanEncoder calls without altering their inputs/outputs."""
    from .clean import CleanEncoder

    class CapturingEncoder(CleanEncoder):
        def __init__(self):
            super().__init__(backbone, reference, checkpoint_hash=checkpoint_hash,
                             profile_hash=profile_hash, cache_references=True)
            self.references = {}; self.current_refs = []; self.offline_source_tokens = 0

        def _states(self, ids, limit):
            self.offline_source_tokens += len(ids)
            return super()._states(ids, limit)

        def reference_payload(self, ref):
            import torch
            check(not torch.is_grad_enabled(), 'capture must be read-only')
            self.current_refs.append({'sha256': ref.sha256, 'role': ref.role})
            if ref.sha256 in self.references:
                return super().reference_payload(ref)
            observed = []
            hook = self.reference.pooler.register_forward_hook(
                lambda module, args, output: observed.append((args[0].detach().clone(), output.detach().clone())))
            try:
                result = super().reference_payload(ref)
            finally:
                hook.remove()
            check(len(observed) == 1 and torch.equal(result, observed[0][1]), 'pooler capture mismatch')
            self.references[ref.sha256] = {'role': ref.role, 'pre': observed[0][0], 'pool': observed[0][1]}
            return result

    return CapturingEncoder()


class CaptureRequest:
    """Scoped observation of the real encode/evaluate calls; never substitutes results."""
    def __init__(self, codec):
        self.codec = codec; self.data = {}; self.calls = 0

    def __enter__(self):
        from ..codec.schema import Role
        self.encode = self.codec.encode; self.evaluate = self.codec.evaluate

        def encode(packet):
            result = self.encode(packet)
            if packet.positions[0].role == Role.REQUEST:
                check('wire' not in self.data, 'multiple request encodings')
                self.data['wire'] = result.detach().clone()
                self.data['roles'] = [int(p.role) for p in packet.positions]
            return result

        def evaluate(h, port, candidate_states=None):
            result = self.evaluate(h, port, candidate_states)
            self.calls += 1
            self.data['h'] = h.detach().clone()
            self.data['q'] = self.codec.qev_q(h.float()).detach().clone()
            self.data['status_logits'] = result['status_logits'].detach().clone()
            if 'raw' in result:
                self.data['raw'] = result['raw'].detach().clone()
            if candidate_states is not None:
                self.data['candidates'] = candidate_states.detach().clone()
            return result

        self.codec.encode = encode; self.codec.evaluate = evaluate
        return self

    def __exit__(self, *exc):
        # Remove instance overrides, restoring normal class method lookup.
        del self.codec.encode; del self.codec.evaluate


def pair_report(group, captures, references, rows):
    from .behavior_qualification import expected
    left, right = (captures[k] for k in group['ids'])
    comparisons = []
    for index, (a, b) in enumerate(zip(left['refs'], right['refs'])):
        if a['role'] != b['role']:
            comparisons.append({'index': index, 'comparable': False, 'roles': [a['role'], b['role']]})
            continue
        x, y = references[a['sha256']], references[b['sha256']]
        comparisons.append({'index': index, 'role': a['role'], 'comparable': True,
            'input_equal': a['sha256'] == b['sha256'], 'pre': difference(x['pre'], y['pre']),
            'pre_last': difference(x['pre'][-1], y['pre'][-1]),
            'pre_mean': difference(x['pre'].mean(0), y['pre'].mean(0)), 'pool': difference(x['pool'], y['pool'])})
    report = dict(group, references=comparisons, reference_counts=[len(left['refs']), len(right['refs'])],
                  wire=difference(left['wire'], right['wire']), h=difference(left['h'], right['h']),
                  q=difference(left['q'], right['q']), permitted_inputs_equal=left['view']==right['view'])
    # Task labels enter reporting only after both original request forwards.
    gold = [expected(rows[k]) for k in group['ids']]
    report['reviewed'] = gold
    if all('raw' in c for c in (left, right)):
        report['raw'] = [float(left['raw']), float(right['raw'])]
        report['raw_change'] = report['raw'][1]-report['raw'][0]
    if report['permitted_inputs_equal']:
        check(report['wire']['exact_equal'] and report['h']['exact_equal'] and report['q']['exact_equal'],
              'identical permitted inputs changed features')
    return report
