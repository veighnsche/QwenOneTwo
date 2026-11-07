"""One finite, GPU-only Noul/Score residual repair; never auto-promote."""
import argparse
from collections import Counter
import json
from pathlib import Path
import random
import shutil
import time
from qwen_one_two.training.inputs import canonical, check, digest
from qwen_one_two.training.manifest import file_hash, BASE_SHA, DIAGNOSTIC_SHA
from qwen_one_two.training.release import load_release, decode
from qwen_one_two.training.two_gpu import ROOT, CHECKPOINT, CHECKPOINT_SHA, RECEIVER, REFERENCE, source_hashes
from qwen_one_two.training.receiver_repair import PROTOCOL, UPDATES, training_ids, visit_ids
from qwen_one_two.training.behavior_qualification import expected, suite
from scripts.fit_reference import TRAIN_SHA, DEV_SHA
from qwen_one_two.g1_device import PRIMARY, require_visibility


def rows(path):
    return [decode(line) for line in path.read_bytes().splitlines()]


def numeric_checks(codec, repair, features, train, ids):
    """New residual independently checked in CUDA FP64, then restored to identity."""
    import torch
    from torch.nn import functional as F
    from qwen_one_two.codec.schema import Port
    from qwen_one_two.training.receiver_repair import readout, losses
    initial = {n: p.detach().clone() for n, p in repair.state_dict().items()}
    result = {}
    def compare(a, b):
        error = float((a.double()-b.double()).square().mean().sqrt())
        scale = float(b.double().square().mean().sqrt())
        check(error <= (1e-5*scale if scale > 1e-6 else 1e-11), 'new residual FP64 tolerance')
        return {'absolute_rms': error, 'relative_rms': error/max(scale, 1e-30)}
    try:
        for op in ('Noul', 'Score'):
            key = next(k for k in ids if train.record(k)['operation'] == op and expected(train.record(k))['status'] == 'Assessed')
            h = features[key].detach().clone().requires_grad_(True)
            branch = repair[op]
            with torch.no_grad():
                branch.up.weight.normal_(std=.02)
                branch.up.bias.normal_(std=.02)
            inputs = [h, *branch.parameters()]
            doubles = [v.detach().double().requires_grad_(True) for v in inputs]
            hd, dw, db, uw, ub = doubles
            normalized = hd / hd.square().mean(-1, keepdim=True).clamp_min(1e-12).sqrt()
            exact = hd + F.linear(F.silu(F.linear(normalized, dw, db)), uw, ub)
            actual = branch(h); cot = torch.randn_like(h)
            ag = torch.autograd.grad((actual*cot).sum(), inputs)
            eg = torch.autograd.grad((exact*cot.double()).sum(), doubles)
            result[op] = {'forward': compare(actual, exact), 'vjp': [compare(a, b) for a, b in zip(ag, eg)]}
            check(all(bool(g.isfinite().all()) and bool(g.abs().sum() > 0) for g in ag), 'disconnected residual gradient')
            port = Port.NOUL if op == 'Noul' else Port.SCORE
            full = codec.evaluate(branch(h), port)
            fast = readout(codec, repair, h, op)
            check(all(torch.equal(full[n], fast[n]) for n in fast), 'minimal/full head forward')
            pars = tuple(repair.parameters())
            fg = torch.autograd.grad(sum(losses(full, train.record(key)).values()), pars, allow_unused=True)
            mg = torch.autograd.grad(sum(losses(fast, train.record(key)).values()), pars, allow_unused=True)
            check(all((a is None and b is None) or (a is not None and b is not None and torch.equal(a, b))
                      for a, b in zip(fg, mg)), 'minimal/full head gradients')
            result[op]['full_head_forward_and_vjp_exact'] = True
    finally:
        repair.load_state_dict(initial)
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--snapshot', required=True); ap.add_argument('--output', required=True)
    ap.add_argument('--execute-gpu', action='store_true'); args = ap.parse_args()
    out = Path(args.output).resolve()
    check(out.is_relative_to(ROOT/'data') and not out.exists(), 'fresh local data/ output required')
    out.mkdir(parents=True)
    train = load_release(ROOT/'data/training-collections-20261105/releases/train', expected_hash=TRAIN_SHA)
    dev = load_release(ROOT/'data/training-collections-20261105/releases/development', expected_hash=DEV_SHA)
    ids = training_ids(train)
    pairs, omitted = suite(dev)
    eids = [e['id'] for e in dev.manifest['records'] if e['selected'] and dev.record(e['id']).get('evaluation_slice') == 'E']
    xids = [k for pair in pairs for k in pair['ids']]
    check(not {train.record(k)['family'] for k in ids} & {dev.record(k)['family'] for k in eids+xids}, 'family overlap')
    old_root = ROOT/'data/receiver-head-only-20261107/run'
    evidence = decode((ROOT/'docs/evidence/receiver-head-only-v1.json').read_bytes())
    reused = [r for r in rows(old_root/'features.jsonl') if r['record'] in ids]
    check(len(reused) == 24, '24 existing compatible features')
    for name in ('features.jsonl', 'features.safetensors'):
        path = old_root/name
        sha = evidence['feature_file_sha256'] if name.endswith('safetensors') else evidence['raw_file_sha256'][str(path.relative_to(ROOT))]
        check(file_hash(path) == sha, 'feature origin drift')
    for name, sha in evidence['sources'].items():
        check(file_hash(ROOT/name) == sha, 'frozen feature implementation drift: '+name)
    baseline = {}
    for name, sha in evidence['development_baseline']['source_hashes'].items():
        check(file_hash(ROOT/name) == sha, 'parent development drift')
        baseline.update({r['id']: r for r in rows(ROOT/name)})
    sources = source_hashes()
    sources[str(Path(__file__).relative_to(ROOT))] = file_hash(Path(__file__))
    sources['docs/receiver-residual-repair-v1.md'] = file_hash(ROOT/'docs/receiver-residual-repair-v1.md')
    plan = {'protocol': PROTOCOL, 'sources': sources, 'gpu_uuid': PRIMARY, 'parent': str(CHECKPOINT),
        'parent_manifest_sha256': CHECKPOINT_SHA, 'receiver': RECEIVER, 'reference': REFERENCE,
        'dataset': TRAIN_SHA, 'development': DEV_SHA, 'ids': ids, 'updates': UPDATES, 'visits': UPDATES*8,
        'reused_feature_hashes': {n: file_hash(old_root/n) for n in ('features.jsonl', 'features.safetensors')},
        'development_baseline': evidence['development_baseline'], 'omitted_reader_ids': omitted,
        'snapshot': str(Path(args.snapshot).resolve()), 'seed': 17, 'new_data': 0,
        'trainable_parameters': 133184, 'scope': 'operation-specific readout repair; no promotion or native training'}
    (out/'plan.json').write_bytes(canonical(plan))
    for name in sources:
        dst = out/'source'/name; dst.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(ROOT/name, dst)
    if not args.execute_gpu:
        print('PREPARED_GPU_NOT_RUN'); return
    require_visibility()
    check(file_hash(ROOT/CHECKPOINT/'manifest.json') == CHECKPOINT_SHA, 'parent manifest')
    check(file_hash(Path(args.snapshot)/'model.safetensors-00001-of-00001.safetensors') == BASE_SHA, 'W file')
    diagnostic = ROOT/'data/g1-proof/gate-repair-v2-20261105/diagnostic.safetensors'
    check(file_hash(diagnostic) == DIAGNOSTIC_SHA, 'diagnostic file')
    import torch
    from safetensors.torch import load_file, save_file
    from torch.nn.attention import sdpa_kernel, SDPBackend
    from qwen_one_two.g1_proof import configure
    from qwen_one_two.g1_evidence import frozen_check
    from qwen_one_two.codec.qualify import load
    from qwen_one_two.training.reference import Reference, state_fingerprint
    from qwen_one_two.training.receiver_learning import groups, ReceiverProducer
    from qwen_one_two.training.receiver_repair import make_repair, repair_groups, readout, losses, installed
    from qwen_one_two.training.engine import Window
    from qwen_one_two.training.checkpoint import restore_model, save, resume
    from qwen_one_two.training.behavior_qualification import receiver_hash, evaluate_case, summarize, PROFILE
    from qwen_one_two.training.component_producer import component_view
    from qwen_one_two.training.information_loss import validate_replay
    from qwen_one_two.training.clean import CleanEncoder
    from qwen_one_two.training.learning_report import summarize_learning
    from qwen_one_two.training.episodes import STATUS_NAMES
    configure(); started = time.monotonic()
    report = {'protocol': PROTOCOL, 'status': 'RUNNING', 'gpu_uuid': PRIMARY,
        'plan_sha256': file_hash(out/'plan.json'), 'optimizer_updates': 0, 'measurements': [], 'checkpoints': []}
    def persist(): (out/'report.json').write_bytes(canonical(report))
    def emit(**entry): print(json.dumps(entry, allow_nan=False), flush=True)
    with sdpa_kernel(SDPBackend.MATH):
        try:
            b, c, _ = load(args.snapshot, diagnostic)
            ref = Reference(b, c, pooler_kind='target-v2'); ref.freeze(); groups(b, c, ref)
            parent = decode((ROOT/CHECKPOINT/'manifest.json').read_bytes())
            restore_model(ROOT/CHECKPOINT, b, c, ref, parent['identities'])
            b.requires_grad_(False); c.requires_grad_(False); ref.requires_grad_(False); b.checkpointing = False
            check(receiver_hash(b, c) == RECEIVER and ref.fingerprint() == REFERENCE, 'restored identities')
            report['initial_base'] = frozen_check(b, args.snapshot)
            check(report['initial_base']['pass'] and report['initial_base']['base_tied_storage'], 'initial frozen W')
            torch.manual_seed(17); torch.cuda.manual_seed_all(17); random.seed(17)
            repair = make_repair(); w = Window(repair_groups(repair), 'components')
            check(sum(p.numel() for p in repair.parameters()) == 133184, 'repair ownership size')
            report['ownership'] = w.ownership
            class CountingEncoder(CleanEncoder):
                offline_source_tokens = 0
                def _states(self, ids, limit):
                    self.offline_source_tokens += len(ids)
                    return super()._states(ids, limit)
            def new_encoder():
                return CountingEncoder(b, ref, checkpoint_hash=RECEIVER, profile_hash=digest(PROFILE.encode()), cache_references=True)
            reference_tokens = []
            pool_hook = ref.pooler.register_forward_pre_hook(lambda module, args: reference_tokens.append(len(args[0])))
            producer = ReceiverProducer(b, c, ref, train)
            producer.encoder = new_encoder()
            def frozen():
                check(receiver_hash(b, c) == RECEIVER and ref.fingerprint() == REFERENCE, 'frozen parent changed')
                check(all(not p.requires_grad and p.grad is None for m in (b, c, ref) for p in m.parameters()), 'parent gradient ownership')
                check(all(p.grad is None for p in repair.parameters()), 'outstanding repair gradients')
            def extract(key):
                captured = []
                hook = c.qev_q.register_forward_pre_hook(lambda module, args: captured.append(args[0].detach().clone()))
                try:
                    with torch.no_grad(): s, _, _ = producer(train.record(key))
                finally: hook.remove()
                case = s.episode_report
                check(not case.get('failure') and len(captured) >= 1 and torch.equal(captured[0], s.hidden[0, -1]), 'complete actual feature')
                check(case['summary']['prefills'] == case['summary']['lineage'] == 1 and case['summary']['held_positions'] == 0, 'feature lineage')
                return captured[0], case
            cache_old = load_file(str(old_root/'features.safetensors'), device='cuda')
            features = {r['record']: cache_old[r['record']+'.h'] for r in reused}
            controls = {r['record']: r for r in reused}
            report['numeric_checks'] = numeric_checks(c, repair, features, train, list(features))
            persist()
            fresh_cases = []
            with (out/'features.jsonl').open('x') as stream, torch.no_grad():
                for i, key in enumerate(ids):
                    if key not in features:
                        features[key], controls[key] = extract(key); fresh_cases.append(controls[key])
                    case = controls[key]; row = train.record(key)
                    view, provenance, _ = component_view(train, row)
                    check(case['record_sha256'] == digest(canonical(row)) and
                          case['input_sha256'] == digest(view.materialize(provenance)[1]), 'feature record/view binding')
                    got = readout(c, repair, features[key], row['operation'])
                    head = {'status': STATUS_NAMES[int(got['status_logits'].argmax())+1], 'raw': float(got['raw']), 'value': float(got['value'])}
                    check(head == case['head'], 'initial identity differs from actual parent')
                    stream.write(json.dumps(dict(case, reused=key in {r['record'] for r in reused}), allow_nan=False)+'\n'); stream.flush()
                    if (i+1) % 25 == 0: emit(event='extraction', completed=i+1, total=len(ids))
            save_file(features, str(out/'features.safetensors')); feature_hash = state_fingerprint(features)
            report.update(feature_sha256=file_hash(out/'features.safetensors'), initial_identity_exact=1003,
                          fresh_feature_sessions=len(fresh_cases), reused_features=len(reused))
            identities = dict(parent['identities'], protocol=digest(PROTOCOL.encode()), sources=digest(canonical(sources)))
            def weights(): return {'repair.'+n: p for n, p in repair.named_parameters()}
            def checkpoint():
                frozen(); path = out/f'checkpoint-{w.scheduled_windows:04d}'
                sha = save(path, w, weights(), identities, {'protocol': PROTOCOL, 'parent_manifest': CHECKPOINT_SHA,
                    'parent_receiver': RECEIVER, 'next_visit': w.scheduled_windows*8, 'order': ids,
                    'feature_sha256': report['feature_sha256']}, reference_frozen=True, reference_id=REFERENCE)
                report['checkpoints'].append({'path': str(path.relative_to(ROOT)), 'sha256': sha}); return path
            def measure():
                frozen(); cases = []
                with torch.no_grad(), (out/f'heads-{w.scheduled_windows:04d}.jsonl').open('x') as stream:
                    for key in ids:
                        row = train.record(key); got = readout(c, repair, features[key], row['operation'])
                        case = {'record': key, 'record_sha256': controls[key]['record_sha256'],
                            'head': {'status': STATUS_NAMES[int(got['status_logits'].argmax())+1], 'raw': float(got['raw']), 'value': float(got['value'])},
                            'losses': {k: float(v) for k, v in losses(got, row).items()}, 'new_causal_positions': 0, 'new_prefills': 0}
                        cases.append(case); stream.write(json.dumps(case, allow_nan=False)+'\n')
                result = summarize_learning(train, cases, ids, kind='heads')
                report['measurements'].append({'window': w.scheduled_windows, 'report': result})
                emit(event='measurement', window=w.scheduled_windows, operations={k:v['counts'] for k,v in result['operations'].items()}); persist()
            checkpoint(); measure(); visits = Counter(); eligible = Counter()
            with (out/'training.jsonl').open('x') as stream:
                for update in range(1, UPDATES+1):
                    keys = visit_ids(ids, update); records = []
                    for key in keys:
                        row = train.record(key); got = readout(c, repair, features[key], row['operation'])
                        values = losses(got, row); eligible.update(values.keys()); visits[key] += 1
                        records.append({'record': key, 'losses': {k: float(v.detach()) for k,v in values.items()}})
                        w.add_episode(values, complete=True, positions=0)
                    result = w.step(); stream.write(json.dumps({'window': result, 'visits': records}, allow_nan=False)+'\n')
                    if update == 1:
                        path = checkpoint(); before = state_fingerprint(weights())
                        def moments(): return state_fingerprint({str(i)+'.'+k:v for i,s in enumerate(w.optimizer.state.values()) for k,v in s.items()})
                        moment_hash = moments(); rng = (torch.cuda.get_rng_state(0), torch.get_rng_state(), random.getstate())
                        # Perturb both a weight and moment: loading must really restore them.
                        with torch.no_grad():
                            next(iter(weights().values())).add_(.125)
                            next(iter(w.optimizer.state.values()))['exp_avg'].add_(.125)
                        torch.rand(1, device='cuda'); random.random()
                        resume(path, w, weights(), identities)
                        check(before == state_fingerprint(weights()) and moments() == moment_hash and w.scheduled_windows == w.optimizer_updates == 1
                              and torch.equal(rng[0], torch.cuda.get_rng_state(0)) and torch.equal(rng[1], torch.get_rng_state())
                              and rng[2] == random.getstate(), 'repair checkpoint/resume mismatch')
                        report['first_update_resume_exact'] = True
                    if update % 100 == 0:
                        stream.flush(); report.update(optimizer_updates=w.optimizer_updates); persist(); emit(event='training', window=update)
                    if update % 500 == 0: checkpoint(); measure()
            check(w.optimizer_updates == UPDATES and set(visits.values()) == {15,16} and set(visits) == set(ids), 'finite exposure completion')
            report['visits'] = dict(visits); report['eligible'] = dict(eligible)
            report['training_seconds'] = time.monotonic()-started
            final_anchors = []
            # One assessed and one non-assessed per operation, selected without development labels.
            anchor_ids = [next(k for k in ids if train.record(k)['operation'] == op and (expected(train.record(k))['status'] == 'Assessed') == assessed)
                          for op in ('Noul','Score') for assessed in (True,False)]
            with installed(c, repair), torch.no_grad(), (out/'fresh-final.jsonl').open('x') as stream:
                for key in anchor_ids:
                    h, case = extract(key); check(torch.equal(h, features[key]), 'fresh final h changed')
                    got = readout(c, repair, features[key], train.record(key)['operation'])
                    check(case['head'] == {'status': STATUS_NAMES[int(got['status_logits'].argmax())+1], 'raw': float(got['raw']), 'value': float(got['value'])}, 'fresh/cached repaired head drift')
                    check(canonical(case['summary']) == canonical(controls[key]['summary']), 'readout changed training ledger')
                    final_anchors.append(case); stream.write(json.dumps(case, allow_nan=False)+'\n')
            encoder = new_encoder()
            encoder.reference_cache = producer.encoder.reference_cache
            all_cases = []
            with installed(c, repair), torch.no_grad():
                for name, keys in (('E',eids), ('X',xids)):
                    cases = []
                    with (out/f'development-{name}.jsonl').open('x') as stream:
                        for key in keys:
                            case = evaluate_case(b,c,ref,dev,dev.record(key),encoder)
                            check(case['view_sha256'] == baseline[key]['view_sha256'], 'development input drift')
                            if dev.record(key)['operation'] == 'Choice': validate_replay(case, baseline[key])
                            cases.append(case); all_cases.append(case)
                            stream.write(json.dumps(case, allow_nan=False)+'\n'); stream.flush()
                            if len(cases) % 25 == 0: emit(event='development', panel=name, completed=len(cases))
                    report['development_'+name] = summarize_learning(dev,cases,keys,kind='behavior')
                    if name == 'X': report['interventions'] = summarize(dev,cases,unchanged=True)
                    persist()
            report['combined_development'] = summarize_learning(dev,all_cases,eids+xids,kind='behavior')
            report['parent_development'] = summarize_learning(dev,[baseline[k] for k in eids+xids],eids+xids,kind='behavior')
            report['choice_retention_exact'] = sum(dev.record(c['id'])['operation'] == 'Choice' for c in all_cases)
            report['accounting'] = {'extraction_sessions':len(fresh_cases), 'extraction_positions':sum(c['summary']['total'] for c in fresh_cases),
                'fresh_final_sessions':len(final_anchors), 'fresh_final_positions':sum(c['summary']['total'] for c in final_anchors),
                'development_sessions':len(all_cases), 'development_positions':sum(c['runtime']['total'] for c in all_cases),
                'cached_fit_visits':sum(visits.values()), 'cached_visit_positions':0, 'cached_visit_prefills':0,
                'new_offline_source_tokens':producer.encoder.offline_source_tokens+encoder.offline_source_tokens,
                'new_reference_annotation_tokens':sum(reference_tokens), 'new_reference_encodings':len(reference_tokens),
                'text_invocation_tokens':0}
            pool_hook.remove()
            report['final_base'] = frozen_check(b,args.snapshot); frozen()
            check(report['final_base']['pass'] and report['final_base']['base_tied_storage'] and state_fingerprint(features) == feature_hash, 'final W/features')
            report.update(status='COMPLETED_REPAIR_NOT_PROMOTED', frozen_unchanged=True, cached_features_unchanged=True,
                repair_fingerprint=state_fingerprint(weights()), seconds=time.monotonic()-started, peak_cuda_bytes=torch.cuda.max_memory_allocated())
        except BaseException as exc:
            report.update(status='FAILED_REPAIR', error=repr(exc), seconds=time.monotonic()-started); raise
        finally: persist()
    emit(status=report['status'], updates=report['optimizer_updates'])


if __name__ == '__main__': main()
