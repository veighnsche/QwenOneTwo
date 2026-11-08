"""Read-only controlled-data/receipt audit. No tensor or model library imports."""
import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import statistics

from qwen_one_two.training.inputs import canonical, check, digest
from qwen_one_two.training.manifest import file_hash
from qwen_one_two.training.release import load_release
from qwen_one_two.training.component_producer import component_view, reference_record
from qwen_one_two.training.behavior_qualification import expected, fp32, grade, suite
from scripts.fit_reference import TRAIN_SHA, DEV_SHA
from qwen_one_two.training.architecture_audit import recipe

ROOT = Path(__file__).resolve().parents[1]
CAPTURE = ROOT/'data/information-loss-20261107/run-v2'
DIRECT = ROOT/'data/direct-readout-20261107/run'
TRAINING = ROOT/'data/receiver-repair-20261107/run'
PROTOCOL = 'training-x-semantic-audit-v1'


def read(path):
    return json.loads(path.read_bytes())


def lines(path):
    return [json.loads(line) for line in path.read_bytes().splitlines()]


def unique(rows, key):
    result = {r[key]: r for r in rows}
    check(len(result) == len(rows), 'duplicate receipt identity')
    return result


def scalar_summary(rows, predictions):
    """Arithmetic on saved values only; one-class panels have no AUC/balanced score."""
    check(rows and len({r['operation'] for r in rows}) == 1, 'single operation panel')
    op = rows[0]['operation']; values = []; targets = []
    for row in rows:
        payload = expected(row)['payload']; check(payload is not None, 'assessed scalar panel')
        raw = predictions[row['id']]
        check(isinstance(raw, (float, int)) and math.isfinite(raw), 'missing/nonfinite saved scalar')
        value = min(1., max(0., fp32(raw)))
        target = payload['p_true'] if op == 'Noul' else fp32(payload['score_interval'][0])
        if op == 'Score': check(payload['score_interval'][0] == payload['score_interval'][1], 'point-only panel')
        values.append(value); targets.append(target)
    result = {'records': len(rows), 'families': len({r['family'] for r in rows}),
              'mse': statistics.mean((v-y)**2 for v,y in zip(values, targets))}
    if op == 'Noul':
        pos = [v for v,y in zip(values, targets) if y > .5]
        neg = [v for v,y in zip(values, targets) if y < .5]
        tp = sum(v > .5 for v in pos); tn = sum(v < .5 for v in neg)
        result.update(correct=tp+tn, true_positive=tp, true_negative=tn, positive=len(pos), negative=len(neg),
            balanced_accuracy=(.5*(tp/len(pos)+tn/len(neg))) if pos and neg else None,
            auc=(sum((p>n)+.5*(p==n) for p in pos for n in neg)/(len(pos)*len(neg))) if pos and neg else None)
    else:
        result.update(exact_hits=sum(v == y for v,y in zip(values, targets)),
                      mean_interval_distance=statistics.mean(abs(v-y) for v,y in zip(values, targets)))
    return result


def checked_artifacts():
    hashes = {}
    def pin(path, sha):
        check(file_hash(path) == sha, 'artifact drift: '+str(path))
        hashes[str(path.relative_to(ROOT))] = sha
    capture_evidence = read(ROOT/'docs/evidence/information-loss-v1.json')
    direct_evidence = read(ROOT/'docs/evidence/direct-latent-readout-v1.json')
    repair_evidence = read(ROOT/'docs/evidence/receiver-residual-repair-v1.json')
    coverage_evidence = read(ROOT/'docs/evidence/reference-complete-coverage-v1.json')
    for relative in ('docs/evidence/information-loss-v1.json', 'docs/evidence/direct-latent-readout-v1.json',
                     'docs/evidence/receiver-residual-repair-v1.json', 'docs/evidence/reference-complete-coverage-v1.json'):
        hashes[relative] = file_hash(ROOT/relative)
    for evidence, paths in ((capture_evidence, ('pairs.json', 'cases.jsonl', 'capture-index.json')),
                            (direct_evidence, ('predictions.jsonl', 'plan.json', 'report.json'))):
        root = CAPTURE if evidence is capture_evidence else DIRECT
        for name in paths:
            path = root/name; pin(path, evidence['raw_hashes'][str(path.relative_to(ROOT))])
    path = TRAINING/'features.jsonl'; pin(path, repair_evidence['raw_hashes'][str(path.relative_to(ROOT))])
    pin(TRAINING/'features.safetensors', direct_evidence['training_feature_sha256'])
    pin(CAPTURE/'captures.safetensors', direct_evidence['development_capture_sha256'])
    pin(DIRECT/'readouts.safetensors', direct_evidence['readout_sha256'])
    coverage_path = ROOT/'data/reference-coverage-20261106/final-pass/views.json'
    pin(coverage_path, coverage_evidence['assets'][str(coverage_path.relative_to(ROOT))])
    plan = read(DIRECT/'plan.json')
    # The current representation implementation must still be the captured one.
    for name, sha in capture_evidence['plan']['sources'].items():
        pin(ROOT/name, sha)
    check(coverage_evidence['run_report']['final_panel']['development']['reference_id'] == plan['reference'], 'current reference lineage')
    return hashes, plan, read(coverage_path), coverage_evidence['run_report']['final_panel']


def audit(train, dev):
    check(train.purpose == 'training' and dev.purpose == 'development', 'training/development only; no locked data')
    hashes, plan, coverage_views, coverage_panel = checked_artifacts()
    train_rows = [train.record(k) for k in sorted(train.selected_ids) if train.record(k)['training_category'] == 'receiver']
    pairs, omitted = suite(dev)
    dev_rows = [dev.record(k) for pair in pairs for k in pair['ids']]
    tcase = unique(lines(TRAINING/'features.jsonl'), 'record')
    dcase = unique(lines(CAPTURE/'cases.jsonl'), 'id')
    predictions = unique(lines(DIRECT/'predictions.jsonl'), 'id')
    captures = unique(read(CAPTURE/'pairs.json'), 'id')
    index = read(CAPTURE/'capture-index.json')
    check(set(tcase) == {r['id'] for r in train_rows if r['operation'] in ('Noul','Score')}, 'complete training cache')
    check(set(dcase) == set(index) == {r['id'] for r in dev_rows}, 'complete development captures')
    check(set(predictions) == {k for panel in plan['panels'].values() for ids in panel.values() for k in ids}, 'complete probe predictions')
    reference = defaultdict(set)
    broad_reference = defaultdict(set)
    check(len(coverage_views) == 3376, 'complete current reference training catalogue')
    for view in coverage_views:
        broad_reference[view['role']].add(view['annotation_sha256'])
    for key in sorted(train.selected_ids):
        row = train.record(key)
        if row['training_category'] == 'reference':
            ref, _ = reference_record(train, row); reference[ref.role].add(ref.sha256)
    panels = {}; detail = {}; view_hashes = {}; source_checks = 0; reference_checks = 0
    directions_by_view = defaultdict(set)
    for name, release, rows in (('training', train, train_rows), ('development_X', dev, dev_rows)):
        recipes = Counter(); domains = Counter(); counts = Counter(); truth = Counter(); numeric = Counter()
        role_requests = defaultdict(list); view_hashes[name] = {}; detail[name] = {}
        for row in rows:
            key = row['id']; view, provenance, indices = component_view(release, row, intervention=name != 'training')
            excerpts, body = view.materialize(provenance); sha = digest(body); view_hashes[name][key] = sha
            directions_by_view[sha].add(indices['direction'])
            info = recipe(row); detail[name][key] = info
            recipes[row['operation']+'/'+info['recipe']] += 1; domains[row['domain']] += 1
            counts[row['operation']+'/'+expected(row)['status']] += 1
            if 'truth' in info: truth[str(info['truth'])] += 1
            if 'denominator' in info: numeric[str(info['denominator'])] += 1
            for ref in view.references: role_requests[ref.role].append(ref.sha256)
            check(excerpts == tuple(text.encode() for text in row['model_inputs']['evidence']['facts'] + row['model_inputs']['evidence']['rules']), 'permitted source order/bytes')
            check(all(ref.role != 'reason' for ref in view.references), 'receiver must exclude Reason targets')
            receipt = (tcase if name == 'training' else dcase).get(key)
            if receipt is not None:
                check(receipt['record_sha256'] == digest(canonical(row)), 'receipt row binding')
                check(receipt['input_sha256' if name == 'training' else 'view_sha256'] == sha == plan['input_hashes'][name][key], 'saved view binding')
                lineage = receipt['source_audit' if name == 'training' else 'construction_audit']
                # Training receipts retain source tensors only; their view hash
                # binds reference input bytes, not individual reference tensors.
                expected_lineage = len(excerpts) + (len(view.references) if name != 'training' else 0)
                check(len(lineage) == expected_lineage, 'complete recorded lineage')
                for i, (excerpt, binding) in enumerate(zip(excerpts, view.evidence)):
                    saved = lineage[i]; parent = saved['parent']
                    check(saved['input_hash'] == digest(excerpt) and saved['local_source'] == i, 'source bytes binding')
                    check(parent == {'source': binding.source, 'version': binding.version,
                        'sha256': provenance.sources[binding.source].sha256, 'byte_start': binding.start, 'byte_end': binding.end}, 'exact source coordinates')
                    source_checks += 1
                for ref, saved in zip(view.references, lineage[len(excerpts):]):
                    check(saved['role'] == ref.role and saved['input_hash'] == ref.sha256, 'reference role/hash')
                    reference_checks += 1
                if name != 'training':
                    check(index[key]['view'] == sha and index[key]['refs'] == [{'role': r.role, 'sha256': r.sha256} for r in view.references], 'capture-index binding')
            if key in predictions:
                pred = predictions[key]
                check(pred['split'] == name and pred['operation'] == row['operation'] and pred['input_sha256'] == sha
                      and pred['record_sha256'] == digest(canonical(row)), 'prediction row/split/view binding')
                check(fp32(pred['raw_fp64']) == pred['raw_fp32'], 'saved scalar ABI')
            info.update(view_sha256=sha, direction=indices['direction'], references=[{'role': r.role, 'sha256': r.sha256} for r in view.references])
        panels[name] = {'records': len(rows), 'families': len({r['family'] for r in rows}),
            'recipes': dict(recipes), 'domains': dict(domains), 'operation_status': dict(counts), 'noul_truth': dict(truth),
            'assessed_score_denominators': dict(numeric), 'unique_views': len(set(view_hashes[name].values())),
            'reference_exposure': {role: {'requested': len(keys), 'unique_requested': len(set(keys)),
                'distinct_original_reference_row_annotations': len(reference[role]),
                'distinct_actual_reference_training_annotations': len(broad_reference[role]),
                'requests_matching_actual_reference_training': sum(k in broad_reference[role] for k in keys),
                'unique_matching_actual_reference_training': len(set(keys)&broad_reference[role])} for role, keys in role_requests.items()}}
    check(not {r['family'] for r in train_rows} & {r['family'] for r in dev_rows}, 'family overlap')
    check(not set(view_hashes['training'].values()) & set(view_hashes['development_X'].values()), 'permitted-view overlap')
    values = {'parent': {k: r.get('predicted_value') for k,r in dcase.items()},
              'direct': {k: r['raw_fp32'] for k,r in predictions.items() if r['split'] == 'development_X'}}
    scalar_panels = {}; pair_rows = []; categories = {}
    for op in ('Noul','Score'):
        groups = defaultdict(list)
        for row in dev_rows:
            if row['operation'] == op and expected(row)['status'] == 'Assessed':
                name = detail['development_X'][row['id']]['recipe']; groups[name].append(row)
        for name, rows in groups.items():
            scalar_panels[op+'/'+name] = {who: scalar_summary(rows, pred) for who,pred in values.items()}
    for pair in pairs:
        old = captures[pair['id']]
        check(all(old[k] == pair[k] for k in ('id','ids','category','operation')), 'saved pair identity')
        a,b = [dev.record(k) for k in pair['ids']]
        ia,ib = [detail['development_X'][k] for k in pair['ids']]
        check(old['reviewed'] == [expected(a),expected(b)], 'saved paired targets')
        check(old['permitted_inputs_equal'] == (ia['view_sha256'] == ib['view_sha256']), 'paired input identity')
        check(len(ia['references']) == len(ib['references']) == len(old['references']), 'complete reference pairs')
        for ra, rb, saved in zip(ia['references'],ib['references'],old['references']):
            check(saved['input_equal'] == (ra['sha256'] == rb['sha256']), 'saved reference contrast')
        item = dict(pair, input_equal=old['permitted_inputs_equal'], h_equal=old['h']['exact_equal'],
            wire_equal=old['wire']['exact_equal'], q_equal=old['q']['exact_equal'],
            h_relative_l2=old['h']['relative_l2'], direction_changed=ia['direction'] != ib['direction'],
            changed_reference_roles=[r['role'] for r in old['references'] if not r['input_equal']],
            parent_behavior=[grade(row,dcase[row['id']])['passed'] for row in (a,b)])
        if all(k in values['direct'] for k in pair['ids']):
            target = [(expected(r)['payload']['p_true'] if r['operation'] == 'Noul' else fp32(expected(r)['payload']['score_interval'][0])) for r in (a,b)]
            item['scalar'] = {}
            for who,pred in values.items():
                v = [min(1.,max(0.,fp32(pred[k]))) for k in pair['ids']]
                delta = target[1]-target[0]
                item['scalar'][who] = {'target': target, 'value': v, 'target_delta': delta, 'value_delta': v[1]-v[0],
                    'direction_correct': ((v[1]-v[0])*delta > 0) if delta else None,
                    'both_correct': all((x != .5 and (x>.5) == (y>.5)) if pair['operation'] == 'Noul' else x == y for x,y in zip(v,target))}
        pair_rows.append(item)
    for category in sorted({p['category'] for p in pairs}):
        selected = [p for p in pair_rows if p['category'] == category]
        sides = {}
        for side in (0,1):
            rows = [dev.record(p['ids'][side]) for p in selected]
            sides[str(side)] = {'behavior_passed': sum(grade(r,dcase[r['id']])['passed'] for r in rows),
                'status_correct': sum(grade(r,dcase[r['id']])['status_correct'] for r in rows)}
            if all(r['id'] in values['direct'] for r in rows):
                sides[str(side)]['scalar'] = {who: scalar_summary(rows, pred) for who,pred in values.items()}
        categories[str(category)] = {'pairs': len(selected), 'input_equal': sum(p['input_equal'] for p in selected),
            'h_equal': sum(p['h_equal'] for p in selected), 'h_relative_l2_median': statistics.median(p['h_relative_l2'] for p in selected),
            'sides': sides, 'scalar_pair': {who: {'both_correct': sum(p['scalar'][who]['both_correct'] for p in selected),
                'direction_correct': sum(p['scalar'][who]['direction_correct'] is True for p in selected)} for who in values} if all('scalar' in p for p in selected) else None}
    return {'protocol': PROTOCOL, 'status': 'COMPLETED_READ_ONLY_NOT_QUALIFIED', 'dataset': train.identity,
        'development': dev.identity, 'artifact_hashes': hashes, 'panels': panels, 'scalar_panels': scalar_panels,
        'categories': categories, 'pairs': pair_rows, 'record_details': detail,
        'actual_reference_catalogue': {'distinct_views': len(coverage_views),
            'by_role': {role:len(keys) for role,keys in broad_reference.items()},
            'final_pass_12_panel_historical': coverage_panel},
        'checks': {'bound_training_feature_records': len(tcase), 'bound_X_captures': len(dcase),
            'bound_predictions': len(predictions), 'source_coordinate_checks': source_checks, 'reference_checks': reference_checks,
            'paired_boundary_receipts': len(pairs), 'family_overlap': 0, 'view_overlap': 0, 'omitted_native_reader_cases': len(omitted),
            'legacy_view_hash_groups_with_multiple_directions': sum(len(values)>1 for values in directions_by_view.values())},
        'accounting': {'new_sessions': 0, 'new_prefills': 0, 'new_causal_positions': 0, 'new_reference_encodings': 0,
            'model_execution': False, 'tensor_execution': False, 'training': False, 'probe_fitting': False,
            'new_data': False, 'new_reviews': False, 'agents': False, 'locked_opened': False},
        'scope': 'CPU metadata/receipt arithmetic; reuse prior GPU captures; no meaning certificate or model promotion'}


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--output', required=True)
    args = parser.parse_args(); output = Path(args.output).resolve()
    check(output.is_relative_to(ROOT/'data') and not output.exists(), 'fresh data output required')
    releases = ROOT/'data/training-collections-20261105/releases'
    result = audit(load_release(releases/'train',expected_hash=TRAIN_SHA), load_release(releases/'development',expected_hash=DEV_SHA))
    sources = ('scripts/audit_latent_semantics.py','docs/training-x-semantic-audit-v1.md','scripts/produce_training_data.py')
    result['audit_source_hashes'] = {name: file_hash(ROOT/name) for name in sources}
    output.parent.mkdir(parents=True,exist_ok=True); output.write_bytes(canonical(result))
    print(json.dumps({'status': result['status'], 'checks': result['checks'], 'scalar_panels': result['scalar_panels']}))


if __name__ == '__main__':
    main()
