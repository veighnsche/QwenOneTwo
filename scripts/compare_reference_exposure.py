"""One matched reference-pooler exposure intervention, reusing the old control."""
import argparse
import json
from pathlib import Path
import time

from qwen_one_two.training.inputs import canonical, digest, check, SLOTS
from qwen_one_two.training.reference_exposure import ANCHORS, CONTROLS, reallocate
from scripts.fit_reference import TRAIN_SHA, DEV_SHA, source_snapshot, export_reference

CONTROL = Path('data/reference-encoder-20261106/run')
SEARCH = Path('data/reference-block-search-completion-20261106/run')
CONTRACT = Path('docs/reference-exposure-comparison-v1.md')
LOCKED_SHA = 'd13594f18738056ee179b3498d54d8261f7f7654aad633fabb776810681aa1cf'


def prepare():
    """Bind existing evidence and construct the intervention without model work."""
    from qwen_one_two.training.release import load_release
    from qwen_one_two.training.reference_data import catalogue, ViewSampler
    from qwen_one_two.training.component_producer import reference_record
    receipts = [Path('docs/evidence/reference-encoder-v2.json'),
                Path('docs/evidence/reference-block-search-completion-v1.json')]
    pins = {}; assets = {}
    for path in receipts:
        pins.update(json.loads(path.read_bytes())['assets'])
        assets[str(path)] = digest(path.read_bytes())
    def bound(path):
        raw = path.read_bytes()
        check(digest(raw) == pins[str(path)], 'control evidence changed: '+str(path))
        assets[str(path)] = digest(raw)
        return raw
    original = [json.loads(line) for line in bound(CONTROL/'windows.jsonl').splitlines()]
    check(len(original) == 1000 and all(r['window'] == i+1 and r['optimizer_updates'] == i+1
          and r['eligible_examples'] == {'reconstruction': 8} and not r['failed_exposures']
          and r['target_history'] == [True]*8 for i, r in enumerate(original)), 'control schedule incomplete')
    initial = json.loads(bound(CONTROL/'checkpoint-0000/manifest.json'))
    check(initial['scheduled_windows'] == initial['optimizer_updates'] == 0, 'initial checkpoint counters')
    old_run = json.loads(bound(CONTROL/'run.json'))
    old_report = json.loads(bound(CONTROL/'report.json'))
    old_sources = json.loads(bound(CONTROL/'source-snapshot/manifest.json'))
    for name, sha in old_sources.items():
        check(digest(Path(name).read_bytes()) == sha, 'original computation source changed: '+name)
    old_views = json.loads(bound(CONTROL/'reference-views.json'))
    releases = Path('data/training-collections-20261105/releases')
    for split, sha in [('train', TRAIN_SHA), ('development', DEV_SHA), ('locked', LOCKED_SHA)]:
        path = releases/split/'manifest.json'
        check(digest(path.read_bytes()) == sha, 'release drift: '+split)
        assets[str(path)] = sha
    train = load_release(releases/'train', expected_hash=TRAIN_SHA)
    views = catalogue(train)
    current = [dict(id=v.identity, role=v.role, operation=v.operation,
                   annotation_sha256=digest(v.content), sources=v.sources) for v in views]
    check(canonical(current) == canonical(old_views) and len(views) == 3376, 'view catalogue changed')
    sampler = ViewSampler(views, train.identity)
    check(sampler.state_dict() == initial['sampler'] and sampler.catalogue_hash == old_run['catalogue_hash'],
          'initial sampler changed')
    schedule = [r['samples'] for r in original]
    for ids in schedule:
        check(list(sampler.next_window()) == ids, 'recorded schedule not reproducible')
    splits = json.loads(bound(SEARCH/'cases.json'))['splits']
    baselines = {split: json.loads(bound(SEARCH/(split+'-00.json'))) for split in ('search', 'screen')}
    annotations = {}
    for split, cases in splits.items():
        check(len(cases) == 20 and [c['id'] for c in cases] == [c['id'] for c in baselines[split]['cases']], 'case order changed')
        for case, baseline in zip(cases, baselines[split]['cases']):
            row = train.record(case['id']); annotation, _ = reference_record(train, row)
            check(annotation.sha256 == case['annotation_sha256'] == baseline['annotation_sha256']
                  and digest(canonical(row)) == case['source_sha256'], 'case source changed')
            annotations[case['id']] = annotation
        check(baselines[split]['reference_id'] == '6b90e7b706239374a088462777eb072cd166a055e353135b0058c583b9da1dd8',
              'baseline reference changed')
    desired = {annotations[key].sha256: count for key, count in ANCHORS.items()}
    treatment, audit = reallocate(schedule, {v.identity: v.role for v in views}, desired)
    check(len(audit['changes']) == 336, 'expected 336 substitutions')
    audit['controls'] = {key: dict(annotation_sha256=annotations[key].sha256,
                        before=audit['before'][annotations[key].sha256],
                        after=audit['after'][annotations[key].sha256]) for key in CONTROLS}
    check([audit['controls'][k]['before'] for k in CONTROLS] == [1, 1, 2, 2, 81, 81], 'original anchor counts')
    qualifications = {n: json.loads(bound(CONTROL/f'qualification-{n:04d}.json')) for n in (900, 1000)}
    return dict(train=train, views=views, original=original, initial=initial, old_report=old_report,
                treatment=treatment, audit=audit, splits=splits, baselines=baselines,
                qualifications=qualifications, assets=assets, unchanged_control_sources=old_sources)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--snapshot', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    data = prepare()
    output = Path(args.output); output.mkdir(parents=True, exist_ok=False)
    def write(name, value):
        (output/name).write_bytes(canonical(value))
    write('exposures.json', data['audit'])
    write('schedule.json', data['treatment'])
    write('reused-control.json', dict(assets=data['assets'], baselines=data['baselines'],
                                    qualifications=data['qualifications']))
    sources = source_snapshot(output)
    for path in (Path(__file__).relative_to(Path.cwd()), CONTRACT, Path('tests/test_reference_exposure.py')):
        raw = path.read_bytes(); sources[str(path)] = digest(raw)
        target = output/'source-snapshot'/path; target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(raw)
    write('source-snapshot/manifest.json', sources)
    started = time.monotonic()
    print(json.dumps(dict(event='prepared_metadata', changed_slots=336, distinct_views=3376, windows=1000)), flush=True)
    import torch
    from torch.nn.attention import sdpa_kernel, SDPBackend
    from qwen_one_two.g1_proof import configure
    from qwen_one_two.codec.qualify import load
    from qwen_one_two.g1_evidence import frozen_check
    from qwen_one_two.training.reference import Reference, state_fingerprint
    from qwen_one_two.training.binding import wire_payload
    from qwen_one_two.training.engine import Window, reference_groups
    from qwen_one_two.training.checkpoint import restore_model, save, bundle_tensors
    from qwen_one_two.training.reference_qualification import evaluate, load_reference_evidence
    from qwen_one_two.training.release import load_release
    from qwen_one_two.training.manifest import BASE_SHA, file_hash
    check(file_hash(Path(args.snapshot)/'model.safetensors-00001-of-00001.safetensors') == BASE_SHA, 'base hash')
    configure()
    with sdpa_kernel(SDPBackend.MATH):
        b, c, _ = load(args.snapshot, 'data/g1-proof/gate-repair-v2-20261105/diagnostic.safetensors')
        ref = Reference(b, c, pooler_kind='target-v2'); ref.train_pooler_only()
        groups = reference_groups(ref)
        for group in groups:
            group['named_parameters'] = [('reference.'+name, p) for name, p in group['named_parameters']]
        window = Window(groups, phase='reference')
        prepared = {}; feature_positions = 0
        for view in data['views']:
            text = view.content.decode(); targets = b.tokenizer.encode(text, add_special_tokens=False)+[248046]
            rendered = b.tokenizer.apply_chat_template([{'role': 'user', 'content': text}], tokenize=False,
                                                       add_generation_prompt=True, enable_thinking=True)
            ids = b.tokenizer.encode(rendered, add_special_tokens=False)
            states = ref.annotation_states(ids); feature_positions += len(ids)
            check(len(targets)+SLOTS[view.role][1] <= 512, 'position ceiling')
            prepared[view.identity] = (view, targets, states, states._version)
            if len(prepared) % 500 == 0:
                print(json.dumps(dict(event='features', completed=len(prepared), total=3376)), flush=True)
        # Restore AFTER feature preparation, matching the old fitter's update boundary.
        restored = restore_model(CONTROL/'checkpoint-0000', b, c, ref, data['initial']['identities'], window=window)
        check(ref.fingerprint() == restored['identities']['reference'] and not window.optimizer.state, 'exact initial state')
        decoder_before = ref.decoder_fingerprint()
        live_before = state_fingerprint({**dict(b.named_parameters()), **{'codec.'+n: p for n, p in c.state_dict().items()}})
        check(decoder_before == data['old_report']['decoder_before_sha256']
              and live_before == data['old_report']['live_before_sha256'], 'matched frozen state')
        check(all(name.startswith('reference.pooler.') for name in window.names), 'foreign optimizer owner')
        frozen_versions = {name: p._version for name, p in ref.named_parameters() if not p.requires_grad}
        identities = dict(restored['identities'], sources=digest(canonical(sources)))
        schedule_hash = data['audit']['treatment_sha256']
        def sampler_state():
            return dict(protocol='reference-exposure-comparison-v1', schedule_sha256=schedule_hash,
                        windows=window.scheduled_windows, dataset=TRAIN_SHA)
        run = dict(protocol='reference-exposure-comparison-v1', source_hashes=sources,
                   matched_control_sources=data['unchanged_control_sources'], identities=identities,
                   initial_checkpoint_sha256=digest((CONTROL/'checkpoint-0000/manifest.json').read_bytes()),
                   initial_reference=ref.fingerprint(), gpu_uuid='GPU-225509c3-d702-6c27-de03-76b5e99b33b1',
                   feature_prefills=len(prepared), feature_positions=feature_positions,
                   cache_device='cuda', pooler_parameters=sum(p.numel() for p in ref.pooler.parameters()),
                   optimizer_ownership=window.ownership, frozen_reference_tensors=len(frozen_versions),
                   decoder_before_sha256=decoder_before, live_before_sha256=live_before,
                   recurrence_backend=b.recurrence_backend, agents=0, new_data=0, capability_updates=0)
        write('run.json', run)
        save(output/'checkpoint-0000', window, bundle_tensors(b, c, ref), identities, sampler_state(),
             reference_frozen=False, reference_id=None)
        dev = load_release('data/training-collections-20261105/releases/development', expected_hash=DEV_SHA)
        qualifications = []; diagnostics = []; training_positions = 0; prefix_matched = 0
        first_change = min(row['window'] for row in data['audit']['changes'])
        torch.cuda.reset_peak_memory_stats()
        with (output/'windows.jsonl').open('x') as log:
            for number, ids in enumerate(data['treatment'], 1):
                step_start = time.monotonic(); losses = []; positions = 0
                for key in ids:
                    view, targets, states, version = prepared[key]
                    check(states._version == version, 'mutated feature cache')
                    payload = ref.pooler(states, *SLOTS[view.role]); prefix = wire_payload(ref, payload, view.role)
                    loss = ref.reconstruct_teacher_forced(prefix, targets, target_history=True)
                    losses.append(float(loss.detach())); positions += len(prefix)+len(targets)
                    window.add_episode({'reconstruction': loss}, complete=True, positions=len(prefix)+len(targets))
                    del loss, prefix, payload
                report = window.step(); training_positions += positions
                check(report['stepped'] and not report['failed_exposures']
                      and report['eligible_examples'] == {'reconstruction': 8}, 'failed training window')
                check(report['active_parameters'] == data['original'][number-1]['active_parameters'], 'active ownership changed')
                check(all(p.grad is None for p in b.parameters()) and all(p.grad is None for p in c.parameters()), 'live gradient contamination')
                check(all(p._version == frozen_versions[name] and p.grad is None
                          for name, p in ref.named_parameters() if not p.requires_grad), 'frozen reference mutated')
                mean = sum(losses)/8
                if number < first_change:
                    old = data['original'][number-1]
                    check(mean == old['mean_reconstruction'] and report['unclipped_norm'] == old['unclipped_norm'],
                          'unaltered control prefix fails exact replay')
                    prefix_matched += 1
                record = dict(report, samples=ids, mean_reconstruction=mean, positions=positions,
                              optimizer_updates=window.optimizer_updates, target_history=[True]*8,
                              seconds=time.monotonic()-step_start, elapsed_seconds=time.monotonic()-started)
                log.write(json.dumps(record)+'\n'); log.flush()
                if number == 1 or number % 10 == 0:
                    print(json.dumps({k: record[k] for k in ('window', 'mean_reconstruction', 'seconds', 'elapsed_seconds')}), flush=True)
                if number % 100:
                    continue
                save(output/f'checkpoint-{number:04d}', window, bundle_tensors(b, c, ref), identities, sampler_state(),
                     reference_frozen=False, reference_id=None)
                if number not in (900, 1000):
                    continue
                with torch.random.fork_rng(devices=[0]):
                    evaluated = Reference(b, c, pooler_kind='target-v2')
                    evaluated.load_state_dict(ref.state_dict()); evaluated.freeze()
                    export_hash = export_reference(output/f'reference-{number:04d}', evaluated,
                        dict(window=number, optimizer_updates=number, training_release=TRAIN_SHA,
                             development_release=DEV_SHA, sources=identities['sources']))
                    cases = []
                    with torch.no_grad():
                        for split, items in data['splits'].items():
                            for case in items:
                                view, targets, states, version = prepared[case['annotation_sha256']]
                                check(states._version == version, 'evaluation feature mutation')
                                payload = evaluated.pooler(states, *SLOTS[view.role])
                                recovery = evaluated.exact_recovery(wire_payload(evaluated, payload, view.role), targets)
                                baseline = next(row for row in data['baselines'][split]['cases'] if row['id'] == case['id'])
                                cases.append(dict(id=case['id'], role=view.role, split=split,
                                    annotation_sha256=view.identity, recovery=recovery, baseline_final_recovery=baseline['recovery']))
                    diagnostic = dict(window=number, reference_id=evaluated.artifact_id, cases=cases,
                                      exact=sum(row['recovery']['exact'] for row in cases), total=len(cases))
                    write(f'diagnostics-{number:04d}.json', diagnostic); diagnostics.append(diagnostic)
                    evaluation = evaluate(evaluated, dev)
                    evaluation['reference_bundle_sha256'] = export_hash
                    evaluation['reference_training_updates'] = number
                    path = output/f'qualification-{number:04d}.json'; write(path.name, evaluation)
                    if evaluation['status'] == 'PASSED':
                        load_reference_evidence(path, dev, expected_hash=digest(path.read_bytes())).assert_reference(evaluated)
                    summary = dict(window=number, status=evaluation['status'], recovery=evaluation['recovery'],
                                   contrast=evaluation['contrast'], reference_id=evaluated.artifact_id,
                                   diagnostic_exact=diagnostic['exact'])
                    qualifications.append(summary)
                    print(json.dumps(dict(event='qualification', **summary)), flush=True)
                    del evaluated
        live_after = state_fingerprint({**dict(b.named_parameters()), **{'codec.'+n: p for n, p in c.state_dict().items()}})
        frozen = frozen_check(b, args.snapshot); decoder_after = ref.decoder_fingerprint()
        check(frozen['pass'] and decoder_before == decoder_after and live_before == live_after, 'frozen identity changed')
        check(all(p.grad is None for p in ref.parameters()), 'outstanding reference gradient')
        for name, sha in sources.items():
            check(digest(Path(name).read_bytes()) == sha, 'run source changed')
        for name, sha in data['assets'].items():
            check(digest(Path(name).read_bytes()) == sha, 'control/data evidence changed')
        passed = len(qualifications) == 2 and all(q['status'] == 'PASSED' for q in qualifications)
        result = dict(status='COMPLETED', reference_qualified=passed, qualifications=qualifications,
                      scheduled_windows=window.scheduled_windows, optimizer_updates=window.optimizer_updates,
                      training_exposures=8000, training_positions=training_positions,
                      unchanged_prefix_windows_verified=prefix_matched, frozen_base=frozen,
                      decoder_before_sha256=decoder_before, decoder_after_sha256=decoder_after,
                      live_before_sha256=live_before, live_after_sha256=live_after,
                      all_gradients_none=True, capability_updates=0, new_data=0, agents=0,
                      peak_allocated=torch.cuda.max_memory_allocated(), elapsed_seconds=time.monotonic()-started)
        write('report.json', result); print(json.dumps(result), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
