"""RD-01 metadata manifests. No model, tokenizer, tensor or prediction input."""
from collections import Counter
import re

from .inputs import canonical, check, digest
from .architecture_audit import recipe, summarize_rows
from .behavior_qualification import expected, suite
from .binding import annotation
from .component_producer import component_view, reference_record
from .producer import native_plan

PROTOCOL = 'redesign-coverage-v1'


def reviewed_profile(row):
    """Recognize the recorded controlled templates only, not arbitrary prose.

    Unknown E meanings remain unknown. A template/target inconsistency never
    provides evidence of familiarity. Gold is external audit metadata only.
    """
    if expected(row)['status'] != 'Assessed':
        return {'recipe': 'non_assessed', 'classification': 'not_applicable'}
    try:
        if row['operation'] == 'Choice':
            criterion = row['semantics']['question']['criterion']
            check(re.fullmatch(r'Capacity at least \d+ .+, then greatest capacity', criterion),
                  'unknown reviewed Choice criterion')
        result = recipe(row)
    except (ValueError, KeyError, TypeError) as error:
        return {'recipe': 'unclassified', 'classification': 'ambiguous_or_unsupported',
                'gap': str(error)}
    return dict(result, classification='reviewed_controlled_template')


def _count(items):
    return dict(sorted(Counter(items).items()))


def _roles(meanings, operation):
    fields = [('question', meanings['question']), ('target', meanings['target'])]
    fields += [('alternative', x) for x in meanings.get('alternatives', [])]
    fields += [(k, meanings['rubric'][k]) for k in
               ('dimension', 'conditions', 'zero', 'one', 'midpoint')
               if k in meanings.get('rubric', {})]
    return [{'role': role, 'sha256': annotation(role, value, operation).sha256}
            for role, value in fields]


def _bindings(release, row, evidence):
    """All exact byte occurrences; ambiguous occurrences never become pointers."""
    result = []
    for kind in ('facts', 'rules'):
        for ordinal, text in enumerate(evidence.get(kind, [])):
            body = text.encode(); matches = []
            for source in row['sources']:
                content = release.asset(source['path'])
                check(digest(content) == source['sha256'], 'stale source identity')
                start = content.find(body)
                while start >= 0:
                    matches.append({'source': source['id'], 'version': source['version'],
                                    'sha256': source['sha256'], 'start': start, 'end': start+len(body)})
                    start = content.find(body, start+1)
            result.append({'kind': kind, 'ordinal': ordinal, 'sha256': digest(body),
                           'matches': matches, 'unique_pointer': len(matches) == 1})
    return result


def _native(release, row):
    category = row['training_category'] or row['construction'].get('transformation')
    if row.get('intervention', {}).get('category') == 9:
        category = 'reader'
    inputs = row['model_inputs']; supervision = row['supervision']
    continuations = supervision.get('continuations', [])
    item = {'category': category, 'calls': [], 'continuation_sha256':
            [digest(x.encode()) for x in continuations], 'gaps': []}
    if row['training_category'] in {'writer', 'interleaved', 'direct', 'language_replay'}:
        plan = native_plan(release, row, 'components', 1)
        item['native_prompt_sha256'] = digest(plan.prompt.encode())
    elif category in {'writer', 'interleaved', 'direct'}:
        item['native_admission'] = 'development metadata only; current producer requires a native training category'
    if category in {'writer', 'interleaved'}:
        calls = row.get('trajectory', {}).get('calls', []) if category == 'interleaved' else [
            {'operation': row['operation'], 'request_target': supervision['request_semantics'],
             'result_target': supervision['result']}]
        for i, call in enumerate(calls):
            op = call['operation']; meanings = call['request_target']; gold = call['result_target']
            check(op == gold['operation'], 'native request/result mismatch')
            reason = supervision.get('reason_sequence', [None]*len(calls))[i]
            if i == 0 and reason is None:
                reason = supervision.get('reason')
            scope = meanings['question'].get('scope')
            evidence = inputs.get('evidence', {})
            if category == 'interleaved' and scope is not None:
                evidence = evidence.get(scope.lower(), {})
            bindings = _bindings(release, row, evidence)
            gaps = []
            if not bindings: gaps.append('missing_scoped_evidence')
            if any(not b['matches'] for b in bindings): gaps.append('missing_source_bytes')
            if any(len(b['matches']) > 1 for b in bindings): gaps.append('ambiguous_source_occurrences')
            if gold['status'] != 'Assessed' and reason is None: gaps.append('missing_call_reason')
            item['calls'].append({'index': i, 'operation': op, 'status': gold['status'],
                'coverage': reviewed_profile(dict(row, operation=op, semantics=meanings,
                    model_inputs={'evidence': evidence}, supervision={'result': gold})),
                'scope': scope, 'request_sha256': digest(canonical(meanings)),
                'references': _roles(meanings, op), 'result_sha256': digest(canonical(gold)),
                'reason_sha256': None if reason is None else annotation('reason', reason, op).sha256,
                'source_bindings': bindings, 'gaps': gaps,
                'runtime_source_pointer': 'not_materialized_by_metadata_audit',
                'metadata_targets_present': True,
                'binding_condition': 'exact actual snapshot roles, candidate identity/order, direction and complete grounded evidence',
                'return_condition': 'compatible_return: exact Noul probability; Score within original interval; unique Choice target; exact non-assessed Reason without attachments',
                'choice_target_unique': len((gold.get('payload') or {}).get('acceptable_selection_ids', [])) == 1
                    if op == 'Choice' and gold['status'] == 'Assessed' else None})
        item['continuation_condition'] = 'all ordered actual requests and returns bound and compatible; otherwise masked'
        item['gaps'] = sorted({g for call in item['calls'] for g in call['gaps']})
        if not continuations: item['gaps'].append('missing_continuation')
        item['execution_qualified'] = False
    elif category == 'reader':
        supplied = inputs.get('bound_return')
        item['supplied_return_sha256'] = None if supplied is None else digest(canonical(supplied))
        item['continuation_condition'] = 'isolated supplied-return only; never a label for an arbitrary actual return'
        if supplied is None: item['gaps'].append('missing_supplied_return')
        elif supplied['status'] != 'Assessed' and supplied.get('reason') is None:
            item['gaps'].append('missing_supplied_reason')
        if not continuations: item['gaps'].append('missing_continuation')
        item['metadata_targets_present'] = supplied is not None and bool(continuations)
    elif category in {'direct', 'language_replay'}:
        item['metadata_targets_present'] = bool(continuations)
        item['continuation_condition'] = 'original permitted prompt; no intervening decision changed task state'
    elif category == 'decision_replay':
        item['metadata_targets_present'] = 'request_semantics' in supervision and 'result' in supervision
        item['continuation_condition'] = 'component replay only; not native no-call or actual-return supervision'
    else:
        item['metadata_targets_present'] = False
        item['continuation_condition'] = 'not a native supervision category'
    return item


def validate_identity(release, entry):
    """Resolve a raw inventory entry back to its immutable release snapshot."""
    check(entry['release_sha256'] == release.identity, 'stale release identity')
    row = release.record(entry['id'])
    check(row['id'] in release.selected_ids and entry['record_sha256'] == digest(canonical(row)),
          'stale record identity')
    check(entry['family'] == row['family'] and entry['split'] == row['split'], 'split/family drift')
    check(entry['reviews'] == row['reviews'], 'stale review identity')
    for review in entry['reviews']:
        check(digest(release.asset(review['artifact_path'])) == review['artifact_hash'], 'stale review artifact')
    for source in entry['sources']:
        check(digest(release.asset(source['path'])) == source['sha256'], 'stale source identity')
    check(entry['sources'] == row['sources'], 'source descriptor drift')


def assert_disjoint(train_rows, dev_rows):
    check(not {r['family'] for r in train_rows} & {r['family'] for r in dev_rows},
          'train/development family overlap')


def build_manifest(train, dev):
    check(train.purpose == 'training' and dev.purpose == 'development', 'training/development only; no locked data')
    rows = {name: [release.record(k) for k in sorted(release.selected_ids)]
            for name, release in (('training', train), ('development', dev))}
    assert_disjoint(rows['training'], rows['development'])
    pairs, omitted = suite(dev)
    panels = {'train_receiver': [r['id'] for r in rows['training'] if r['training_category'] == 'receiver'],
              'development_E': [r['id'] for r in rows['development'] if r.get('evaluation_slice') == 'E'],
              'development_X': [k for pair in pairs for k in pair['ids']]}
    receiver_ids = set(k for ids in panels.values() for k in ids)
    inventory = []
    for name, release in (('training', train), ('development', dev)):
        entries = {e['id']: e for e in release.manifest['records'] if e['selected']}
        for row in rows[name]:
            key = row['id']; cat = row['training_category'] or row.get('evaluation_slice')
            entry = {'id': key, 'family': row['family'], 'split': row['split'], 'category': cat,
                'release_sha256': release.identity, 'record_path': entries[key]['path'],
                'record_sha256': digest(canonical(row)), 'reviews': row['reviews'], 'sources': row['sources'],
                'operation': row['operation'], 'domain': row['domain'],
                'assessment_applicable': cat not in {'reference', 'RF', 'language_replay'},
                'native': _native(release, row)}
            check(entry['record_sha256'] == entries[key]['sha256'], 'original record byte identity')
            validate_identity(release, entry)
            if key in receiver_ids or cat in {'reader', 'decision_replay'}:
                view, provenance, _ = component_view(release, row, intervention=key in receiver_ids and name == 'development')
                excerpts, body = view.materialize(provenance)
                entry['permitted_input'] = {'sha256': digest(body),
                    'references': [{'role': r.role, 'sha256': r.sha256} for r in view.references],
                    'source_bindings': [{'source': b.source, 'version': b.version, 'start': b.start, 'end': b.end,
                                         'sha256': digest(text)} for b, text in zip(view.evidence, excerpts)]}
            elif cat in {'reference', 'RF'}:
                ref, _ = reference_record(release, row)
                entry['permitted_input'] = {'sha256': ref.sha256, 'role': ref.role}
            else:
                entry['permitted_input'] = {'sha256': digest(canonical(row['model_inputs'])),
                    'scope': 'declared branch inputs; supplied returns permitted only for isolated reading'}
            gold = expected(row)
            entry['status'] = gold['status']
            entry['targets_sha256'] = digest(canonical(row['supervision']))
            if cat not in {'reference', 'RF', 'language_replay'}:
                entry['coverage'] = reviewed_profile(row)
                if row['operation'] == 'Noul' and gold['status'] == 'Assessed':
                    entry['polarity'] = 'true' if gold['payload']['p_true'] > .5 else 'false' if gold['payload']['p_true'] < .5 else 'tie'
                if row['operation'] == 'Score' and gold['status'] == 'Assessed':
                    lo, hi = gold['payload']['score_interval']
                    entry['score_region'] = 'endpoint' if lo == hi and lo in (0, 1) else 'interior' if 0 < lo <= hi < 1 else 'boundary_interval'
                    entry['rubric_direction'] = row['semantics']['rubric']['direction']
            inventory.append(entry)
    index = {e['id']: e for e in inventory}
    training = [index[k] for k in panels['train_receiver']]
    recipes = {(e['operation'], e['coverage']['recipe']) for e in training if e['status'] == 'Assessed'}
    denominators = {e['coverage']['denominator'] for e in training if 'denominator' in e['coverage']}
    numerators = [e['coverage']['numerator'] for e in training if 'numerator' in e['coverage']]
    role_hashes = {r['sha256'] for e in training for r in e['permitted_input']['references']}
    for name in ('development_E', 'development_X'):
        for key in panels[name]:
            e = index[key]; profile = e['coverage']; known = profile['classification'] == 'reviewed_controlled_template'
            familiar_op = (e['operation'], profile['recipe']) in recipes if known else None
            familiar_range = (profile['denominator'] in denominators and min(numerators) <= profile['numerator'] <= max(numerators)) if 'denominator' in profile else None
            e['axes'] = {'unseen_family': True, 'familiar_operator': familiar_op,
                'familiar_numeric_range': familiar_range,
                'unseen_operator_or_range': familiar_op is False or familiar_range is False,
                'changed_wording_or_binding': any(r['sha256'] not in role_hashes for r in e['permitted_input']['references']),
                'scope': 'overlapping descriptive axes; unknown meanings stay unclassified; no isolated causal claim'}
            for axis, selected in (('familiar_operations_ranges', familiar_op is True and familiar_range is not False),
                                   ('unseen_operator_range', e['axes']['unseen_operator_or_range']),
                                   ('wording_binding_changes', e['axes']['changed_wording_or_binding']),
                                   ('unclassified', profile['classification'] == 'ambiguous_or_unsupported')):
                if selected: panels.setdefault(name+'/'+axis, []).append(key)
    panels['choice_retention'] = [k for name in ('development_E', 'development_X') for k in panels[name]
                                  if index[k]['operation'] == 'Choice' and index[k]['status'] == 'Assessed']
    for name in ('train_receiver', 'development_E', 'development_X'):
        for key in panels[name]:
            e = index[key]
            panels.setdefault(name+'/cell/'+e['operation']+'/'+e['status'], []).append(key)
            if 'score_region' in e:
                panels.setdefault(name+'/Score/'+e['score_region'], []).append(key)
    summaries = {}
    for name, ids in panels.items():
        release = train if name == 'train_receiver' or name.startswith('train_receiver/') else dev
        numeric = [index[k]['coverage'] for k in ids if 'denominator' in index[k]['coverage']]
        summaries[name] = dict(summarize_rows([release.record(k) for k in ids]),
            ids_sha256=digest(canonical(ids)), assessed=sum(index[k]['status'] == 'Assessed' for k in ids),
            nonassessed=sum(index[k]['status'] != 'Assessed' for k in ids),
            recipes=_count(index[k]['operation']+'/'+index[k]['coverage']['recipe'] for k in ids),
            score_regions=_count(index[k]['score_region'] for k in ids if 'score_region' in index[k]),
            numeric_ranges={'classified_records': len(numeric),
                'denominators': _count(str(p['denominator']) for p in numeric),
                'numerator_min': min((p['numerator'] for p in numeric), default=None),
                'numerator_max': max((p['numerator'] for p in numeric), default=None),
                'units': _count(p['unit'] for p in numeric)})
    support = {}
    for op, names in {'Noul': ('positive_and', 'negated_and', 'positive_or'),
                      'Score': ('completion', 'remaining_fraction', 'half_completion'),
                      'Choice': ('capacity_maximum',)}.items():
        for recipe_name in names:
            selected = [e for e in training if e['operation'] == op and e['coverage']['recipe'] == recipe_name and e['status'] == 'Assessed']
            support[op+'/'+recipe_name] = {'assessed_training_records': len(selected),
                'polarity': _count(e['polarity'] for e in selected if 'polarity' in e),
                'denominators': _count(str(e['coverage']['denominator']) for e in selected if 'denominator' in e['coverage'])}
    native = {}
    for name in rows:
        selected = [e for e in inventory if e['split'] == ('train' if name == 'training' else 'development')]
        native[name] = {cat: {'records': len(group),
            'metadata_target_records': sum(bool(e['native'].get('metadata_targets_present') or e['native']['calls']) for e in group),
            'request_targets': sum(len(e['native']['calls']) for e in group),
            'gap_records': _count(g for e in group for g in e['native']['gaps']),
            'operation_status': _count(e['operation']+'/'+e['status'] for e in group if e['assessment_applicable']),
            'recipes': _count(e['operation']+'/'+e['coverage']['recipe'] for e in group if 'coverage' in e),
            'call_recipes': _count(c['operation']+'/'+c['coverage']['recipe'] for e in group for c in e['native']['calls']),
            'call_denominators': _count(str(c['coverage']['denominator']) for e in group for c in e['native']['calls'] if 'denominator' in c['coverage']),
            'call_statuses': _count(c['operation']+'/'+c['status'] for e in group for c in e['native']['calls']),
            'nonunique_choice_targets': sum(c['choice_target_unique'] is False for e in group for c in e['native']['calls']),
            'calls_missing_reason': sum('missing_call_reason' in c['gaps'] for e in group for c in e['native']['calls'])}
            for cat in sorted({e['native']['category'] for e in selected})
            for group in [[e for e in selected if e['native']['category'] == cat]]}
    unsupported = [{'cell': cell, 'assessed_training_records': 0} for cell, counts in support.items()
                   if counts['assessed_training_records'] == 0]
    development_receiver_ids = set(panels['development_E']+panels['development_X'])
    for denominator in sorted({e['coverage']['denominator'] for e in inventory
                               if e['id'] in development_receiver_ids
                               and 'denominator' in e['coverage']} - denominators):
        unsupported.append({'cell': 'Score/denominator/'+str(denominator), 'assessed_training_records': 0})
    return {'protocol': PROTOCOL, 'releases': {'training': train.identity, 'development': dev.identity},
        'collections': {name: {'records': len(group), 'categories': _count(r['training_category'] or r.get('evaluation_slice') for r in group)} for name, group in rows.items()},
        'panels': panels, 'coverage': summaries, 'training_support': support, 'native_supervision': native,
        'unsupported_cells': unsupported,
        'receiver_rows_accounted': len(training), 'receiver_rows_sha256': digest(canonical(panels['train_receiver'])),
        'family_overlap': [], 'pairs': pairs, 'separate_reader_X_ids': omitted, 'inventory': inventory,
        'accounting': {'model_execution': False, 'training': False, 'new_data': False, 'new_reviews': False, 'locked_opened': False},
        'limits': ['development remains repeatedly inspected development',
                   'no profile/token lengths or runtime semantic binding certified by metadata',
                   'unclassified prose is not evidence of familiar or unseen operations',
                   'receiver abstentions do not establish assessed operation/range coverage',
                   'native target availability is conditional, not execution or learned-flow qualification']}
