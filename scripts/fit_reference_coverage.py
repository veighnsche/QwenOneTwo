"""Complete shuffled reference passes until measured recovery or plateau."""
import argparse
from collections import Counter
import json
from pathlib import Path
import time

from qwen_one_two.training.inputs import SLOTS, canonical, check, digest
from scripts.fit_reference import TRAIN_SHA, DEV_SHA, source_snapshot, export_reference

PARENT = Path('data/reference-exposure-20261106/run/checkpoint-1000')
RECEIPT = Path('docs/evidence/reference-exposure-comparison-v1.json')
CONTRACT = Path('docs/reference-complete-coverage-v1.md')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--snapshot', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--resume', help='Exact coverage checkpoint; write a new output directory')
    args = parser.parse_args()
    from qwen_one_two.training.release import load_release
    from qwen_one_two.training.reference_data import catalogue, CompleteViewSampler, coverage_panel, coverage_plateau
    receipt = json.loads(RECEIPT.read_bytes())
    assets = {str(RECEIPT): digest(RECEIPT.read_bytes())}
    def bound(path):
        raw = path.read_bytes(); sha = digest(raw)
        check(sha == receipt['assets'][str(path)], 'prior evidence changed: '+str(path))
        assets[str(path)] = sha
        return json.loads(raw)
    original_parent = bound(PARENT/'manifest.json')
    previous = bound(PARENT.parent/'diagnostics-1000.json')
    previous_sources = bound(PARENT.parent/'source-snapshot/manifest.json')
    model_sources = {n:sha for n,sha in previous_sources.items() if n.startswith('qwen_one_two/')
                     and n not in {'qwen_one_two/training/engine.py', 'qwen_one_two/training/profile.py',
                                  'qwen_one_two/training/checkpoint.py', 'qwen_one_two/training/reference_data.py'}}
    for name, sha in model_sources.items():
        check(digest(Path(name).read_bytes()) == sha, 'unchanged model source required: '+name)
    baseline = {r['annotation_sha256']: r for r in previous['cases']}
    check(len(baseline) == 40, 'prior diagnostic identity')
    train = load_release('data/training-collections-20261105/releases/train', expected_hash=TRAIN_SHA)
    dev = load_release('data/training-collections-20261105/releases/development', expected_hash=DEV_SHA)
    views = catalogue(train); sampler = CompleteViewSampler(views, train.identity)
    check(len(views) == 3376 and sampler.windows_per_pass == 422, 'pinned catalogue size')
    old_views = json.loads(Path('data/reference-encoder-20261106/run/reference-views.json').read_bytes())
    check(canonical([dict(id=v.identity, role=v.role, operation=v.operation,
                         annotation_sha256=v.sha256, sources=v.sources) for v in views]) == canonical(old_views), 'catalogue drift')
    panel = coverage_panel(views, baseline)
    check(len(panel) == 191, 'coverage panel identity/size')
    output = Path(args.output); output.mkdir(parents=True, exist_ok=False)
    def write(name, value):(output/name).write_bytes(canonical(value))
    sources = source_snapshot(output)
    for path in (Path(__file__).relative_to(Path.cwd()), CONTRACT, Path('tests/test_reference_coverage.py'),
                 Path('tests/test_training_checkpoint_gpu.py')):
        raw = path.read_bytes(); sources[str(path)] = digest(raw)
        target = output/'source-snapshot'/path; target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(raw)
    write('source-snapshot/manifest.json', sources)
    write('views.json', [dict(id=v.identity, role=v.role, annotation_sha256=v.sha256, sources=v.sources) for v in views])
    write('panel.json', dict(ids=panel, prior_cases=previous, selection='fixed hash order, up to 32 per role; includes all prior 40'))
    started = time.monotonic()
    def emit(event, **values):print(json.dumps(dict(event=event, elapsed_seconds=time.monotonic()-started, **values)), flush=True)
    emit('metadata_ready', panel=len(panel), views=len(views), windows_per_pass=422)
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
    from qwen_one_two.training.manifest import BASE_SHA, file_hash
    check(file_hash(Path(args.snapshot)/'model.safetensors-00001-of-00001.safetensors') == BASE_SHA, 'base hash')
    configure()
    with sdpa_kernel(SDPBackend.MATH):
        b, c, _ = load(args.snapshot, 'data/g1-proof/gate-repair-v2-20261105/diagnostic.safetensors')
        ref = Reference(b, c, pooler_kind='target-v2'); ref.train_pooler_only()
        groups = reference_groups(ref)
        for group in groups:group['named_parameters'] = [('reference.'+n, p) for n, p in group['named_parameters']]
        window = Window(groups, phase='reference', schedule_profile='reference-coverage-v1' if args.resume else 'a4-v1')
        prepared = {}; feature_positions = 0
        for view in views:
            text = view.content.decode(); targets = b.tokenizer.encode(text, add_special_tokens=False)+[248046]
            rendered = b.tokenizer.apply_chat_template([{'role':'user', 'content':text}], tokenize=False,
                                                       add_generation_prompt=True, enable_thinking=True)
            ids = b.tokenizer.encode(rendered, add_special_tokens=False)
            states = ref.annotation_states(ids); feature_positions += len(ids)
            check(len(targets)+SLOTS[view.role][1] <= 512, 'reference position ceiling')
            prepared[view.identity] = (view, targets, states, states._version)
            if len(prepared)%500 == 0:emit('features', completed=len(prepared), total=3376)
        parent = Path(args.resume) if args.resume else PARENT
        manifest = json.loads((parent/'manifest.json').read_bytes()) if args.resume else original_parent
        if args.resume:
            check(manifest['identities']['sources'] == digest(canonical(sources)), 'resume source profile changed')
        restore_model(parent, b, c, ref, manifest['identities'], window=window)
        if not args.resume:
            check(window.scheduled_windows == window.optimizer_updates == 1000
                  and ref.fingerprint() == 'd741b65f85ef4d381c74bf48b3bf2330de3009f8e561e20d9e3bdc389ffdeb42', 'parent state')
            window.schedule_profile = 'reference-coverage-v1'
        identities = dict(manifest['identities'], sources=digest(canonical(sources)))
        decoder_before = ref.decoder_fingerprint()
        live_before = state_fingerprint({**dict(b.named_parameters()), **{'codec.'+n:p for n,p in c.state_dict().items()}})
        check(decoder_before == receipt['run_report']['decoder_after_sha256']
              and live_before == receipt['run_report']['live_after_sha256'], 'frozen parent state drift')
        check(all(n.startswith('reference.pooler.') for n in window.names), 'foreign optimizer owner')
        fixed_versions = {n:t._version for n,t in ref.state_dict(keep_vars=True).items() if not n.startswith('pooler.')}
        curves = []; loss_sum = 0.; pass_examples = 0; paired_first = None; stop_reason = None; full = None
        if args.resume:
            state = manifest['sampler']; sampler.load_state_dict(state['coverage'])
            curves = state['curves']; loss_sum = state['loss_sum']; pass_examples = state['pass_examples']
            paired_first = state['paired_first']; stop_reason = state['stop_reason']; full = state['full_training']
            check(window.scheduled_windows == 1000+sampler.windows, 'resume window/sampler continuity')
        run = dict(protocol='reference-complete-coverage-v1', parent=str(parent), parent_manifest_sha256=digest((parent/'manifest.json').read_bytes()),
                   source_hashes=sources, identities=identities, assets=assets, gpu_uuid='GPU-225509c3-d702-6c27-de03-76b5e99b33b1',
                   initial_reference=ref.fingerprint(), decoder_before_sha256=decoder_before, live_before_sha256=live_before,
                   schedule_profile=window.schedule_profile, optimizer_ownership=window.ownership,
                   feature_prefills=len(prepared), feature_positions=feature_positions, cache_device='cuda',
                   recurrence_backend=b.recurrence_backend, base_updates=1000, windows_per_pass=422,
                   panel_size=len(panel), resume=args.resume, new_data=0, agents=0, capability_updates=0)
        write('run.json', run)
        training_positions = 0; eval_positions = 0

        def recover(evaluated, ids, label, reused=None):
            nonlocal eval_positions
            cases = []; reused_cases = {}
            if reused is not None:
                check(reused['reference_id'] == evaluated.artifact_id, 'reuse reference identity')
                reused_cases = {r['id']:r for r in reused['cases']}
                check(set(reused_cases) <= set(ids), 'reuse case identity')
            with torch.no_grad():
                for i, key in enumerate(ids, 1):
                    view, targets, states, version = prepared[key]
                    check(states._version == version, 'mutated frozen feature cache')
                    if key in reused_cases:
                        case = reused_cases[key]
                        check(case['role'] == view.role and case['recovery']['target_tokens'] == len(targets), 'reuse targets')
                        cases.append(case)
                        continue
                    payload = evaluated.pooler(states, *SLOTS[view.role])
                    recovery = evaluated.exact_recovery(wire_payload(evaluated, payload, view.role), targets)
                    positions = SLOTS[view.role][1]+recovery['matched_tokens']-int(recovery['exact'])
                    eval_positions += positions
                    cases.append(dict(id=key, role=view.role, recovery=recovery, decoder_positions=positions))
                    if i%100 == 0:emit('recovery_progress', label=label, completed=i, total=len(ids))
            legacy = [r for r in cases if r['id'] in baseline]
            by_role = {role:dict(exact=sum(r['recovery']['exact'] for r in cases if r['role'] == role),
                                total=sum(r['role'] == role for r in cases)) for role in SLOTS}
            return dict(reference_id=evaluated.artifact_id, cases=cases, exact=sum(r['recovery']['exact'] for r in cases),
                        total=len(cases), by_role=by_role, new_recovery_calls=len(cases)-len(reused_cases),
                        reused_recovery_calls=len(reused_cases),
                        legacy40=dict(before=sum(r['recovery']['exact'] for r in baseline.values()),
                                      after=sum(r['recovery']['exact'] for r in legacy),
                                      gains=[baseline[r['id']]['id'] for r in legacy if r['recovery']['exact'] and not baseline[r['id']]['recovery']['exact']],
                                      losses=[baseline[r['id']]['id'] for r in legacy if not r['recovery']['exact'] and baseline[r['id']]['recovery']['exact']]))

        def evaluate_frozen(*, panel_check=False, full_check=False, development=False, reused=None):
            with torch.random.fork_rng(devices=[0]):
                evaluated = Reference(b, c, pooler_kind='target-v2'); evaluated.load_state_dict(ref.state_dict()); evaluated.freeze()
                number = window.scheduled_windows; result = {}
                if panel_check:
                    result['panel'] = recover(evaluated, panel, f'panel-{number}', reused)
                    write(f'panel-{number:06d}.json', result['panel'])
                if full_check:
                    result['full'] = recover(evaluated, tuple(prepared), f'full-{number}', reused)
                    write(f'full-training-{number:06d}.json', result['full'])
                if development:
                    bundle = export_reference(output/f'reference-{number:06d}', evaluated,
                        dict(window=number, optimizer_updates=window.optimizer_updates, training_release=TRAIN_SHA,
                             development_release=DEV_SHA, sources=identities['sources']))
                    q = evaluate(evaluated, dev); q['reference_bundle_sha256'] = bundle
                    q['reference_training_updates'] = window.optimizer_updates
                    path = output/f'qualification-{number:06d}.json'; write(path.name, q)
                    if q['status'] == 'PASSED':
                        load_reference_evidence(path, dev, expected_hash=digest(path.read_bytes())).assert_reference(evaluated)
                    result['development'] = dict(window=number, status=q['status'], recovery=q['recovery'], contrast=q['contrast'],
                                                 reference_id=evaluated.artifact_id, path=str(path), sha256=digest(path.read_bytes()))
                    emit('qualification', **result['development'])
                return result

        def checkpoint():
            state = dict(coverage=sampler.state_dict(), curves=curves, loss_sum=loss_sum, pass_examples=pass_examples,
                         paired_first=paired_first, stop_reason=stop_reason, full_training=full)
            path = output/f'checkpoint-{window.scheduled_windows:06d}'
            save(path, window, bundle_tensors(b, c, ref), identities, state, reference_frozen=False, reference_id=None)

        if not args.resume:
            reused = dict(reference_id=previous['reference_id'], cases=[dict(id=key, role=row['role'], recovery=row['recovery'],
                decoder_positions=SLOTS[row['role']][1]+row['recovery']['matched_tokens']-int(row['recovery']['exact']))
                for key,row in baseline.items()])
            initial = evaluate_frozen(panel_check=True, reused=reused)['panel']
            curves.append(dict(pass_number=0, window=1000, panel_exact=initial['exact'], panel_total=initial['total'],
                               by_role=initial['by_role'], legacy40=initial['legacy40'], mean_reconstruction=None))
            emit('initial_panel', **curves[-1]); checkpoint()
        torch.cuda.reset_peak_memory_stats()
        with (output/'windows.jsonl').open('x') as log, (output/'curves.jsonl').open('x') as curve_log:
            for row in curves:curve_log.write(json.dumps(row)+'\n')
            curve_log.flush()
            while stop_reason is None:
                ids = sampler.next_window(); losses = []; positions = 0; step_started = time.monotonic()
                for key in ids:
                    view, targets, states, version = prepared[key]
                    check(states._version == version, 'mutated feature cache')
                    payload = ref.pooler(states, *SLOTS[view.role]); prefix = wire_payload(ref, payload, view.role)
                    loss = ref.reconstruct_teacher_forced(prefix, targets, target_history=True)
                    losses.append(float(loss.detach())); positions += len(prefix)+len(targets)
                    window.add_episode({'reconstruction':loss}, complete=True, positions=len(prefix)+len(targets))
                    del loss, prefix, payload
                report = window.step()
                branches = {('target' if prepared[k][0].role == 'target' else 'shared') for k in ids}
                expected = {n for n in window.names if n.split('.')[2] in branches}
                check(set(report['active_parameters']) == expected, 'role-local optimizer ownership')
                check(report['eligible_examples'] == {'reconstruction':8} and not report['failed_exposures'] and report['stepped'], 'failed exposure')
                check(all(p.grad is None for p in b.parameters()) and all(p.grad is None for p in c.parameters()), 'live gradients')
                check(all(t._version == fixed_versions[n] for n,t in ref.state_dict(keep_vars=True).items() if n in fixed_versions), 'frozen reference mutation')
                loss_sum += sum(losses); pass_examples += 8; training_positions += positions
                record = dict(report, samples=ids, mean_reconstruction=sum(losses)/8, positions=positions,
                              optimizer_updates=window.optimizer_updates, coverage_window=sampler.windows,
                              seconds=time.monotonic()-step_started, elapsed_seconds=time.monotonic()-started)
                log.write(json.dumps(record)+'\n'); log.flush()
                if sampler.windows%10 == 0:emit('window', window=window.scheduled_windows, coverage_window=sampler.windows,
                                               pass_number=(sampler.windows-1)//422+1, mean_reconstruction=record['mean_reconstruction'])
                in_pass = sampler.windows%422
                if in_pass == 322:
                    paired_first = evaluate_frozen(development=True)['development']
                if in_pass == 0:
                    check(pass_examples == len(views), 'complete pass exposure accounting')
                    observed = evaluate_frozen(panel_check=True, development=True)
                    p = observed['panel']; q = observed['development']
                    curve = dict(pass_number=sampler.windows//422, window=window.scheduled_windows,
                                 panel_exact=p['exact'], panel_total=p['total'], by_role=p['by_role'],
                                 legacy40=p['legacy40'], mean_reconstruction=loss_sum/pass_examples,
                                 first_development=paired_first, development=q)
                    curves.append(curve); plateau = coverage_plateau(curves)
                    if p['exact']/p['total'] >= .95 or plateau:
                        measured = evaluate_frozen(full_check=True, reused=p)['full']
                        full = dict(exact=measured['exact'], total=measured['total'], by_role=measured['by_role'],
                                    path=str(output/f'full-training-{window.scheduled_windows:06d}.json'))
                        if full['exact']/full['total'] >= .95:stop_reason = 'FULL_TRAINING_RECOVERY_TARGET'
                        elif plateau:stop_reason = 'OBSERVED_LEARNING_PLATEAU'
                    curve['plateau'] = plateau; curve['full_training'] = full; curve['stop_reason'] = stop_reason
                    curve_log.write(json.dumps(curve)+'\n'); curve_log.flush(); emit('pass_complete', **curve)
                    loss_sum = 0.; pass_examples = 0
                if sampler.windows%100 == 0 or in_pass in (322, 0):checkpoint()
        final = curves[-1]; pair = [final['first_development'], final['development']]
        check(pair[1]['window']-pair[0]['window'] == 100, 'final paired qualification spacing')
        live_after = state_fingerprint({**dict(b.named_parameters()), **{'codec.'+n:p for n,p in c.state_dict().items()}})
        frozen = frozen_check(b, args.snapshot); decoder_after = ref.decoder_fingerprint()
        check(frozen['pass'] and decoder_before == decoder_after and live_before == live_after, 'frozen state changed')
        check(all(p.grad is None for p in ref.parameters()), 'outstanding reference gradient')
        for name, sha in sources.items():check(digest(Path(name).read_bytes()) == sha, 'source changed during run')
        for name, sha in assets.items():check(digest(Path(name).read_bytes()) == sha, 'prior evidence changed')
        result = dict(status='COMPLETED', stop_reason=stop_reason, passes=sampler.windows//422,
                      coverage_windows=sampler.windows, cumulative_optimizer_updates=window.optimizer_updates,
                      reference_qualified=all(q['status'] == 'PASSED' for q in pair), qualifications=pair,
                      initial_panel=curves[0], final_panel=final, full_training=full,
                      training_positions_this_process=training_positions,
                      teacher_forced_forward_positions_this_process=training_positions-8*sum(1 for _ in (output/'windows.jsonl').open()),
                      recovery_forward_positions_this_process=eval_positions, frozen_base=frozen,
                      decoder_before_sha256=decoder_before, decoder_after_sha256=decoder_after,
                      live_before_sha256=live_before, live_after_sha256=live_after, gradients_none=True,
                      capability_updates=0, new_data=0, agents=0,
                      peak_allocated=torch.cuda.max_memory_allocated(), elapsed_seconds=time.monotonic()-started)
        write('report.json', result); emit('completed', **result)
    return 0


if __name__ == '__main__':raise SystemExit(main())
