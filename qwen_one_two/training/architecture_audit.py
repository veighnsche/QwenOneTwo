"""CPU metadata audit of existing annotations; never loads a model or locked data."""
from collections import Counter, defaultdict
from .inputs import canonical, check, digest
from .behavior_qualification import expected, suite
from .component_producer import component_view, reference_record


def summarize_rows(rows):
    cells = Counter(); domains = Counter(); truth = Counter(); intervals = Counter()
    directions = Counter(); families = set(); operations = Counter(); point_targets = 0
    for row in rows:
        families.add(row['family']); domains[row['domain']] += 1
        op = row['operation']; operations[op] += 1
        gold = expected(row); cells[op + '/' + gold['status']] += 1
        if gold['status'] != 'Assessed':
            continue
        if op == 'Noul':
            p = gold['payload']['p_true']
            truth['true' if p > .5 else 'false' if p < .5 else 'tie'] += 1
        elif op == 'Score':
            lo, hi = gold['payload']['score_interval']
            check(0 <= lo <= hi <= 1, 'invalid reviewed Score interval')
            intervals[canonical([lo, hi]).decode()] += 1
            point_targets += int(lo == hi)
            directions[row['semantics']['rubric']['direction']] += 1
    return {'records': len(rows), 'families': len(families), 'domains': dict(sorted(domains.items())),
            'operations': dict(sorted(operations.items())), 'operation_status': dict(sorted(cells.items())),
            'noul_truth': dict(sorted(truth.items())), 'score_intervals': dict(sorted(intervals.items())),
            'score_directions': dict(sorted(directions.items())),
            'score_point_targets': point_targets}


def audit(train, dev):
    check(train.purpose == 'training' and dev.purpose == 'development', 'training/development only; no locked release')
    collections = {}; selected = {}
    for name, release in (('train', train), ('development', dev)):
        rows = [release.record(k) for k in sorted(release.selected_ids)]
        selected[name] = rows
        collections[name] = {'release': release.identity, 'records': len(rows),
            'families': len({r['family'] for r in rows}),
            'allocations': dict(sorted(Counter(r.get('training_category') or r.get('evaluation_slice') for r in rows).items()))}
    groups, omitted = suite(dev)
    panels = {'train_receiver': [r for r in selected['train'] if r['training_category'] == 'receiver'],
              'development_E': [r for r in selected['development'] if r.get('evaluation_slice') == 'E'],
              'development_X': [dev.record(k) for g in groups for k in g['ids']]}
    summaries = {name: summarize_rows(rows) for name, rows in panels.items()}
    views = {}; conflicts = {}
    for name, rows in panels.items():
        release = train if name == 'train_receiver' else dev
        inputs = defaultdict(list)
        for row in rows:
            view, provenance, _ = component_view(release, row, intervention=name != 'train_receiver')
            sha = digest(view.materialize(provenance)[1])
            inputs[sha].append({'id': row['id'], 'family': row['family'], 'target': digest(canonical(expected(row)))})
        views[name] = inputs
        summaries[name]['unique_permitted_inputs'] = len(inputs)
        conflicts[name] = {sha: records for sha, records in inputs.items() if len({r['target'] for r in records}) > 1}
    reference = {}
    for name, release in (('train', train), ('development', dev)):
        roles = defaultdict(list)
        for row in selected[name]:
            if row['training_category'] == 'reference' or row.get('evaluation_slice') == 'RF':
                ref, _ = reference_record(release, row)
                roles[ref.role].append(ref.sha256)
        reference[name] = {role: {'records': len(keys), 'unique_annotations': len(set(keys))} for role, keys in sorted(roles.items())}
    train_families = {r['family'] for r in panels['train_receiver']}
    development_families = {r['family'] for name in ('development_E', 'development_X') for r in panels[name]}
    return {'protocol': 'architecture-baseline-v1', 'collections': collections, 'panels': summaries,
            'reference_roles': reference, 'contradictory_identical_inputs': conflicts,
            'receiver_family_overlap': sorted(train_families & development_families),
            'receiver_permitted_input_overlap': sorted(set(views['train_receiver']) &
                (set(views['development_E']) | set(views['development_X']))),
            'development_pairs': len(groups), 'omitted_reader_cases': len(omitted),
            'model_execution': False, 'locked_opened': False,
            'scope': 'metadata and permitted-input audit; no new annotations, review, fitting or semantic certification'}
