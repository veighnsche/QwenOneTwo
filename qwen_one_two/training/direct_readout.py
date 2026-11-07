"""Standalone affine probes of saved latent states; never install in the codec."""
from .inputs import check, digest
from .behavior_qualification import expected, fp32

PROTOCOL = 'direct-latent-readout-v1'
LAMBDAS = (1e-9, 1e-6, 1e-3, 1.)
FOLDS = 3


def fold(family):
    return int(digest((PROTOCOL+':family:'+family).encode()),16) % FOLDS


def selected(release, ids, operation):
    check(release.purpose in ('training','development'), 'no locked probe data')
    return [k for k in ids if release.record(k)['operation'] == operation
            and expected(release.record(k))['status'] == 'Assessed']


def targets(release, ids):
    check(release.purpose in ('training','development'), 'no locked probe targets')
    result = []
    for key in ids:
        row = release.record(key); gold = expected(row)
        check(gold['status'] == 'Assessed', 'scalar targets require reviewed Assessed')
        if row['operation'] == 'Noul':
            result.append(gold['payload']['p_true'])
        else:
            lo, hi = gold['payload']['score_interval']
            check(lo == hi, 'point regression cannot replace an interval annotation')
            result.append(fp32(lo))
    return result


def fit_ridge(x, y, strength):
    """Minimize mean raw squared error + lambda ||w||² in centered/scaled h.

    CUDA FP64 dual solve, unpenalized intercept. Mean/scale are training-only.
    """
    import torch
    from ..g1_device import require_cuda, require_training_context
    require_training_context(); require_cuda(torch.cuda,x); require_cuda(torch.cuda,y)
    check(x.ndim == 2 and y.shape == (len(x),) and len(x) > 1 and strength > 0, 'ridge shapes/penalty')
    check(bool(x.isfinite().all() & y.isfinite().all()), 'finite ridge inputs')
    x, y = x.double(), y.double()
    center = x.mean(0); centered = x-center
    scale = centered.square().sum(-1).mean().sqrt()
    check(float(scale) > 1e-12, 'constant representation')
    z = centered/scale; intercept = y.mean(); target = y-intercept
    gram = z@z.T
    system = gram + len(x)*strength*torch.eye(len(x),device=x.device,dtype=x.dtype)
    dual = torch.linalg.solve(system,target)
    residual = (system@dual-target).norm()/target.norm().clamp_min(1e-12)
    check(float(residual) <= 1e-8, 'FP64 ridge solve residual')
    weight = z.T@dual
    check(bool(weight.isfinite().all()), 'finite direct readout')
    return {'center':center,'scale':scale,'weight':weight,'intercept':intercept}, float(residual)


def predict(model, x):
    import torch
    from ..g1_device import require_cuda
    require_cuda(torch.cuda,x)
    return ((x.double()-model['center'])/model['scale'])@model['weight']+model['intercept']


def fit_operation(release, ids, x):
    """No development argument: family-grouped training CV, then one full fit."""
    check(release.purpose == 'training', 'never fit/select on development or locked data')
    import torch
    from ..g1_device import require_cuda
    check(len(ids)==len(set(ids))==len(x), 'ordered complete training features')
    require_cuda(torch.cuda,x)
    y = torch.tensor(targets(release,ids),device=x.device,dtype=torch.float64)
    groups = [fold(release.record(k)['family']) for k in ids]
    check(set(groups)==set(range(FOLDS)), 'all training folds required')
    candidates = []
    for strength in LAMBDAS:
        oof = torch.empty_like(y); checks = []
        for held in range(FOLDS):
            mask = torch.tensor([f==held for f in groups],device=x.device,dtype=torch.bool)
            check(int(mask.sum())>0 and int((~mask).sum())>1, 'nonempty grouped split')
            model, residual = fit_ridge(x[~mask],y[~mask],strength)
            oof[mask] = predict(model,x[mask]); checks.append(residual)
        risk = float((oof-y).square().mean())
        candidates.append({'lambda':strength, 'training_oof_raw_mse':risk, 'solve_relative_residuals':checks,
                           'oof_raw':oof.tolist()})
    # Fixed tie rule prefers stronger regularization; never consult development.
    chosen = min(candidates,key=lambda row:(row['training_oof_raw_mse'],-row['lambda']))
    model, residual = fit_ridge(x,y,chosen['lambda'])
    return model, {'selected_lambda':chosen['lambda'], 'candidates':candidates,
                   'folds':groups, 'final_solve_relative_residual':residual}


def metrics(release, ids, raw):
    """Recorded scalar predictions only. These are not delivered-result grades."""
    check(len(ids)==len(raw) and bool(ids), 'complete scalar metric coverage')
    y = targets(release,ids); op = release.record(ids[0])['operation']
    check(all(release.record(k)['operation']==op for k in ids), 'single operation summary')
    import math
    check(all(math.isfinite(v) for v in raw), 'finite diagnostic predictions')
    # Same FP32 scalar ABI and exact intervals; no extra rounding/tolerance.
    raw32 = [fp32(v) for v in raw]; value = [min(1.,max(0.,v)) for v in raw32]
    result = {'scheduled':len(ids), 'families':len({release.record(k)['family'] for k in ids}),
        'raw_mse':sum((v-t)**2 for v,t in zip(raw32,y))/len(y),
        'clipped_mse':sum((v-t)**2 for v,t in zip(value,y))/len(y),
        'raw_range':[min(raw32),max(raw32)], 'value_range':[min(value),max(value)],
        'clipped_predictions':sum(a!=b for a,b in zip(raw32,value)),
        'scope':'assessed scalar head diagnostic; no status/transport/delivery qualification'}
    if op == 'Noul':
        positive = [v for v,t in zip(value,y) if t>.5]; negative = [v for v,t in zip(value,y) if t<.5]
        check(positive and negative and len(positive)+len(negative)==len(y), 'both non-tied truth classes')
        tp = sum(v>.5 for v in positive); tn = sum(v<.5 for v in negative)
        result.update(correct=tp+tn, positive=len(positive), negative=len(negative), true_positive=tp, true_negative=tn,
            ties=sum(v==.5 for v in value), balanced_accuracy=.5*(tp/len(positive)+tn/len(negative)),
            auc=sum((p>n)+.5*(p==n) for p in positive for n in negative)/(len(positive)*len(negative)))
    else:
        result.update(exact_hits=sum(v==t for v,t in zip(value,y)),
                      mean_interval_distance=sum(abs(v-t) for v,t in zip(value,y))/len(y))
    return result


def paired_values(release, pairs, raw_by_id):
    """Only pairs with two assessed scalar targets; never imply delivery."""
    check(release.purpose=='development','development intervention reporting only')
    rows=[]
    for pair in pairs:
        if not all(k in raw_by_id for k in pair['ids']):
            continue
        y=targets(release,pair['ids']);v=[min(1.,max(0.,fp32(raw_by_id[k]))) for k in pair['ids']]
        correct=[value!=.5 and (value>.5)==(target>.5) if pair['operation']=='Noul' else value==target
                 for value,target in zip(v,y)]
        delta=y[1]-y[0];actual=v[1]-v[0]
        rows.append(dict(pair,both_scalar_correct=all(correct),target_delta=delta,value_delta=actual,
                         direction_correct=(actual*delta>0) if delta else None))
    return {'scheduled_pairs':len(pairs),'two_assessed_scalar_pairs':len(rows),
            'both_scalar_correct':sum(r['both_scalar_correct'] for r in rows),
            'changed_target_pairs':sum(r['direction_correct'] is not None for r in rows),
            'direction_correct':sum(r['direction_correct'] is True for r in rows),'pairs':rows,
            'scope':'two-sided scalar correctness only; excludes Choice/non-assessed targets; not delivered behavior'}
