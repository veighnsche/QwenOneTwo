"""Capture existing latent receiver boundaries on the 3090; no fitting or updates."""
import argparse
from collections import defaultdict
from pathlib import Path
import json
import shutil
import time
from scripts.fit_reference import TRAIN_SHA, DEV_SHA
from qwen_one_two.g1_device import PRIMARY, diagnostic_device, require_visibility
from qwen_one_two.training.inputs import canonical, check, digest
from qwen_one_two.training.manifest import file_hash, BASE_SHA, DIAGNOSTIC_SHA
from qwen_one_two.training.release import load_release, decode
from qwen_one_two.training.two_gpu import ROOT, CHECKPOINT, CHECKPOINT_SHA, RECEIVER, REFERENCE, source_hashes
from qwen_one_two.training.behavior_qualification import suite, expected, PROFILE
from qwen_one_two.training.information_loss import (capture_encoder, CaptureRequest, difference,
    spectrum, label_geometry, readout, validate_replay, pair_report)

PROTOCOL = 'information-loss-v1'
OLD = ROOT / 'data/receiver-head-only-20261107/run'


def records(path):
    return [decode(line) for line in path.read_bytes().splitlines()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--execute-gpu', action='store_true')
    args = parser.parse_args()
    out = Path(args.output).resolve()
    check(out.is_relative_to(ROOT/'data') and not out.exists(), 'fresh local data/ output required')
    out.mkdir(parents=True)
    evidence_path = ROOT/'docs/evidence/receiver-head-only-v1.json'
    evidence = decode(evidence_path.read_bytes())
    check(file_hash(OLD/'plan.json') == evidence['plan_sha256'] and
          file_hash(OLD/'report.json') == evidence['report_sha256'] and
          file_hash(OLD/'features.safetensors') == evidence['feature_file_sha256'], 'saved feature evidence drift')
    old_plan = decode((OLD/'plan.json').read_bytes()); old_report = decode((OLD/'report.json').read_bytes())
    # Reuse only code-identical frozen representations and pinned historical receipts.
    for name, sha in evidence['sources'].items():
        check(file_hash(ROOT/name) == sha, 'historical representation source drift: '+name)
    baseline_path = ROOT/'data/receiver-learning-20261106/continued/final-X.jsonl'
    historical = decode((ROOT/'docs/evidence/receiver-training-probe-v1.json').read_bytes())['historical_development']
    check(file_hash(baseline_path) == historical['source_hashes'][str(baseline_path.relative_to(ROOT))], 'parent X receipt drift')
    reuse_paths = [OLD/name for name in ('features.jsonl', 'panel-453.jsonl', 'development-X.jsonl')]
    for path in reuse_paths:
        check(file_hash(path) == evidence['raw_file_sha256'][str(path.relative_to(ROOT))], 'saved head receipt drift')
    train = load_release(ROOT/'data/training-collections-20261105/releases/train', expected_hash=TRAIN_SHA)
    dev = load_release(ROOT/'data/training-collections-20261105/releases/development', expected_hash=DEV_SHA)
    pairs, omitted = suite(dev)
    ids = [k for pair in pairs for k in pair['ids']]
    panel = old_plan['panel']; train_ids = [p['id'] for p in panel]
    check(len(ids) == 180 and len(train_ids) == 32 and old_plan['receiver'] == RECEIVER, 'fixed diagnostic scope')
    sources = source_hashes(); sources['scripts/diagnose_information_loss.py'] = file_hash(Path(__file__))
    plan = {'protocol': PROTOCOL, 'sources': sources, 'parent': str(CHECKPOINT), 'parent_manifest': CHECKPOINT_SHA,
            'receiver': RECEIVER, 'reference': REFERENCE, 'train': TRAIN_SHA, 'development': DEV_SHA,
            'pairs': pairs, 'omitted_reader_ids': omitted, 'reused_training_ids': train_ids,
            'head_evidence_sha256': file_hash(evidence_path), 'feature_file_sha256': evidence['feature_file_sha256'],
            'new_sessions': 180, 'reused_training_features': 32, 'optimizer_updates': 0,
            'snapshot': str(Path(args.snapshot).resolve()), 'gpu_uuid': PRIMARY,
            'scope': 'read-only existing representations and saved readouts; no fitted probes, new inputs or native qualification'}
    (out/'plan.json').write_bytes(canonical(plan))
    for name in sources:
        dst = out/'source'/name; dst.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(ROOT/name, dst)
    if not args.execute_gpu:
        print('PREPARED_GPU_NOT_RUN'); return
    require_visibility()
    check(file_hash(ROOT/CHECKPOINT/'manifest.json') == CHECKPOINT_SHA, 'parent checkpoint manifest')
    check(file_hash(Path(args.snapshot)/'model.safetensors-00001-of-00001.safetensors') == BASE_SHA, 'base hash')
    diagnostic = ROOT/'data/g1-proof/gate-repair-v2-20261105/diagnostic.safetensors'
    check(file_hash(diagnostic) == DIAGNOSTIC_SHA, 'initial codec hash')
    import torch
    from safetensors import safe_open
    from safetensors.torch import load_file, save_file
    from torch.nn.attention import sdpa_kernel, SDPBackend
    from qwen_one_two.g1_proof import configure
    from qwen_one_two.g1_evidence import frozen_check
    from qwen_one_two.codec.qualify import load
    from qwen_one_two.training.reference import Reference, state_fingerprint
    from qwen_one_two.training.receiver_learning import groups
    from qwen_one_two.training.checkpoint import restore_model
    from qwen_one_two.training.behavior_qualification import receiver_hash, evaluate_case, summarize
    from qwen_one_two.training.component_producer import component_view
    configure(); started = time.monotonic()
    report = {'protocol': PROTOCOL, 'status': 'RUNNING', 'gpu_uuid': PRIMARY,
              'plan_sha256': file_hash(out/'plan.json'), 'optimizer_updates': 0, 'training_performed': False,
              'completed_cases': 0, 'receiver': RECEIVER, 'reference': REFERENCE}
    def persist():
        (out/'report.json').write_bytes(canonical(report))
    def emit(**entry):
        print(json.dumps(entry, allow_nan=False), flush=True)
    captures = {}; encoder = None
    with diagnostic_device(PRIMARY, execute_gpu=True), sdpa_kernel(SDPBackend.MATH), torch.no_grad():
        try:
            b, c, _ = load(args.snapshot, diagnostic)
            ref = Reference(b, c, pooler_kind='target-v2'); ref.freeze(); groups(b, c, ref)
            parent = decode((ROOT/CHECKPOINT/'manifest.json').read_bytes())
            restore_model(ROOT/CHECKPOINT, b, c, ref, parent['identities'])
            b.requires_grad_(False); c.requires_grad_(False); ref.requires_grad_(False); b.checkpointing = False
            check(receiver_hash(b,c) == RECEIVER and ref.fingerprint() == REFERENCE, 'loaded identities')
            fixed = state_fingerprint({**{'backbone.'+n:p for n,p in b.named_parameters() if n.endswith(('.R_A','.R_B','.D_A','.D_B'))},
                **{'codec.'+n:p for n,p in c.state_dict(keep_vars=True).items() if n not in ('prototypes','score_row')}})
            check(fixed == evidence['frozen_representation'], 'saved feature representation mismatch')
            report['initial_base'] = frozen_check(b, args.snapshot)
            check(report['initial_base']['pass'] and report['initial_base']['base_tied_storage'], 'initial W')
            cached = load_file(str(OLD/'features.safetensors'), device='cuda')
            check(state_fingerprint(cached) == old_report['feature_tensor_fingerprint'], 'feature tensor identity')
            old_features = {r['record']: r for r in records(OLD/'features.jsonl')}
            old_final = {r['record']: r for r in records(OLD/'panel-453.jsonl')}
            final_path = OLD/'checkpoint-2000'
            check(file_hash(final_path/'manifest.json') == evidence['checkpoints'][-1]['sha256'], 'saved final-head manifest')
            final_manifest = decode((final_path/'manifest.json').read_bytes())
            check(file_hash(final_path/'weights.safetensors') == final_manifest['files']['weights.safetensors'], 'saved final-head weights')
            with safe_open(str(final_path/'weights.safetensors'), framework='pt', device='cuda') as f:
                heads = {'prototypes': f.get_tensor('codec.prototypes'), 'score_row': f.get_tensor('codec.score_row')}
            report['saved_heads'] = {'manifest': file_hash(final_path/'manifest.json'), 'installed': False}
            train_raw = {'parent': {}, 'final': {}}
            for key in train_ids:
                row = train.record(key); view, provenance, _ = component_view(train,row)
                check(digest(view.materialize(provenance)[1]) == old_features[key]['input_sha256'] and
                      digest(canonical(row)) == old_features[key]['record_sha256'], 'cached input mismatch')
                if row['operation'] == 'Choice':
                    continue
                for name, kwargs, old in (('parent', {}, old_features), ('final', heads, old_final)):
                    raw, value = readout(c, cached[key+'.h'], row['operation'], **kwargs)
                    check(float(raw) == old[key]['head']['raw'] and float(value) == old[key]['head']['value'], 'cached head replay mismatch')
                    train_raw[name][key] = raw
            report['reused_features'] = {'cases': 32, 'new_sessions': 0, 'new_positions': 0,
                'parent_and_final_scalar_heads_exact': True, 'fingerprint': fixed}
            # Known analytic CUDA matrices validate report arithmetic, not model numerics.
            toy = torch.diag(torch.tensor([3.,1.,0.], device='cuda'))
            check(spectrum(toy, center=False)['fp32_scale_rank'] == 2 and
                  spectrum(torch.zeros_like(toy), center=False)['fp32_scale_rank'] == 0 and
                  difference(toy,toy)['exact_equal'], 'diagnostic CUDA arithmetic self-check')
            report['cuda_statistics_checks'] = True; persist()
            encoder = capture_encoder(b, ref, checkpoint_hash=RECEIVER, profile_hash=digest(PROFILE.encode()))
            baseline = {r['id']: r for r in records(baseline_path)}
            final_cases = {r['id']: r for r in records(OLD/'development-X.jsonl')}
            actual_cases = []; final_raw = {}
            with (out/'cases.jsonl').open('x') as stream:
                for key in ids:
                    encoder.current_refs = []
                    with CaptureRequest(c) as capture:
                        case = evaluate_case(b,c,ref,dev,dev.record(key),encoder)
                    check(capture.calls == 1 and 'wire' in capture.data, 'one actual latent request required')
                    item = dict(capture.data, refs=list(encoder.current_refs), view=case['view_sha256'])
                    captures[key] = item
                    stream.write(json.dumps(case,allow_nan=False)+'\n'); stream.flush()
                    validate_replay(case, baseline[key])
                    # Reproduce saved final-head measurements without installing or fitting those heads.
                    if dev.record(key)['operation'] != 'Choice':
                        raw, value = readout(c,item['h'],dev.record(key)['operation'],**heads)
                        check(float(value) == final_cases[key]['predicted_value'], 'saved final development readout mismatch')
                        final_raw[key] = raw
                    actual_cases.append(case)
                    report['completed_cases'] += 1; persist()
                    emit(event='captured', completed=report['completed_cases'], id=key)
            rows = {key: dev.record(key) for key in ids}
            comparisons = [pair_report(g,captures,encoder.references,rows) for g in pairs]
            (out/'pairs.json').write_bytes(canonical(comparisons))
            metrics = {}; geometry = {}
            for name, keys, release, feature, raw_sets in (
                ('training_cached',train_ids,train,lambda k:cached[k+'.h'],train_raw),
                ('development_X',ids,dev,lambda k:captures[k]['h'],
                 {'parent':{k:v['raw'] for k,v in captures.items() if 'raw' in v}, 'final':final_raw})):
                for op in ('Noul','Choice','Score'):
                    chosen = [k for k in keys if release.record(k)['operation']==op and expected(release.record(k))['status']=='Assessed']
                    h = torch.stack([feature(k) for k in chosen])
                    metrics[name+'/'+op+'/h'] = spectrum(h)
                    q = torch.stack([c.qev_q(feature(k).float()) for k in chosen])
                    metrics[name+'/'+op+'/q'] = spectrum(q)
                    if op == 'Noul':
                        labels = torch.tensor([expected(release.record(k))['payload']['p_true']>.5 for k in chosen],device='cuda')
                        for head, raw_map in raw_sets.items():
                            geometry[name+'/'+head] = label_geometry(q,labels,torch.stack([raw_map[k] for k in chosen]))
            by_role = defaultdict(list)
            for ref_item in encoder.references.values():
                by_role[ref_item['role']].append(ref_item)
            for role, items in by_role.items():
                metrics['reference/'+role+'/pre_mean'] = spectrum(torch.stack([v['pre'].mean(0) for v in items]))
                metrics['reference/'+role+'/pool'] = spectrum(torch.stack([v['pool'].flatten() for v in items]))
            metrics['qev_q_weights'] = spectrum(c.qev_q.weight,center=False)
            metrics['qev_k_weights'] = spectrum(c.qev_k.weight,center=False)
            (out/'spectra.json').write_bytes(canonical(metrics))
            report['noul_geometry'] = geometry
            report['behavior'] = summarize(dev,actual_cases,unchanged=True)
            report['pair_counts'] = {'total': len(comparisons),
                'identical_permitted_inputs': sum(p['permitted_inputs_equal'] for p in comparisons),
                'changed_inputs_identical_h': sum(not p['permitted_inputs_equal'] and p['h']['exact_equal'] for p in comparisons),
                'changed_reference_identical_pool': sum(not r['input_equal'] and r['pool']['exact_equal']
                    for p in comparisons for r in p['references'] if r['comparable'])}
            report['accounting'] = {'new_sessions': len(actual_cases),
                'session_positions': sum(ca['runtime']['total'] for ca in actual_cases),
                'offline_source_tokens': encoder.offline_source_tokens,
                'offline_reference_tokens': sum(len(v['pre']) for v in encoder.references.values()),
                'unique_reference_encodings': len(encoder.references), 'reference_cache_hits': encoder.reference_cache_hits,
                'cached_training_new_positions': 0, 'text_invocation_tokens': 0}
            report['final_base'] = frozen_check(b,args.snapshot)
            check(report['final_base']['pass'] and report['final_base']['base_tied_storage'] and
                  receiver_hash(b,c)==RECEIVER and ref.fingerprint()==REFERENCE and
                  all(not p.requires_grad and p.grad is None for m in (b,c,ref) for p in m.parameters()), 'frozen/read-only boundary')
            check(state_fingerprint(cached)==old_report['feature_tensor_fingerprint'], 'cached features mutated')
            report.update(status='COMPLETED_READ_ONLY_NOT_QUALIFIED', frozen_unchanged=True,
                gradients_none=True, parent_cases_exact=180, final_scalar_cases_exact=len(final_raw),
                seconds=time.monotonic()-started, peak_cuda_bytes=torch.cuda.max_memory_allocated())
        except BaseException as exc:
            report.update(status='FAILED_DIAGNOSTIC', error=repr(exc), seconds=time.monotonic()-started)
            raise
        finally:
            # Preserve completed captures even if a later analysis fails.
            tensors = {key+'.'+name:value.contiguous() for key,item in captures.items()
                       for name,value in item.items() if isinstance(value,torch.Tensor)}
            if encoder is not None:
                tensors.update({'reference.'+key+'.'+name:value.contiguous()
                    for key,item in encoder.references.items() for name,value in item.items() if isinstance(value,torch.Tensor)})
                (out/'capture-index.json').write_bytes(canonical({key:{n:v for n,v in item.items()
                    if not isinstance(v,torch.Tensor)} for key,item in captures.items()}))
            if tensors:
                save_file(tensors,str(out/'captures.safetensors'))
                report['capture_sha256'] = file_hash(out/'captures.safetensors')
            persist()
    emit(status=report['status'], cases=report['completed_cases'])


if __name__ == '__main__':
    main()
