"""Model-free receiver plans and binding guards; no launcher or training loop."""
from collections import Counter
from pathlib import Path

from ..g1_device import PRIMARY
from .checkpoint_metadata import inspect
from .component_producer import component_view
from .host_metrics import Recorder
from .inputs import canonical, check, digest
from .learning_report import probe_panel
from .loss_config import COEFFICIENTS
from .manifest import BASE_SHA, DIAGNOSTIC_SHA, file_hash
from .receiver_profiles import LEGACY, profile, receiver_parameter
from .reference_qualification import PROFILE
from .release import decode, load_release
from .two_gpu import ROOT, CHECKPOINT, CHECKPOINT_SHA, REFERENCE

PROTOCOL = 'receiver-preparation-v1'
COVERAGE = 'docs/evidence/redesign-coverage-v1.json'
TRAIN_SHA = 'c70d4f44135a34c00e7fa0348833b1ec56613def15f5753231589492ed35478f'
DEV_SHA = '5b8f4cffdc4f33e255d3a34306cac03a68c50dcd9dab905ae8d8adcf89c81e77'


def _sha(value):
    check(isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value),
          'identity hash required')
    return value


def plan_sources(root=ROOT):
    paths = sorted((root/'qwen_one_two').rglob('*.py')) + [root/'scripts/train_receiver_probe.py', root/'scripts/fit_reference.py']
    return {str(path.relative_to(root)): file_hash(path) for path in paths}


def initial_state(root=ROOT):
    path = root/CHECKPOINT
    check(file_hash(path/'manifest.json') == CHECKPOINT_SHA, 'initial checkpoint manifest drift')
    saved = decode((path/'manifest.json').read_bytes())
    inspect(path, saved['identities'])  # Opaque bytes only, including Adam/RNG files.
    check(saved['reference_frozen'] and saved['reference_id'] == REFERENCE
          and saved['identities']['reference'] == REFERENCE and saved['identities']['base'] == BASE_SHA
          and saved['identities']['dataset'] == TRAIN_SHA, 'initial checkpoint frozen/data identity')
    check(saved['identities']['qev'] == file_hash(root/'data/qev/release_manifest.json')
          and saved['identities']['precision'] == digest(PROFILE.encode()), 'initial Qev/precision drift')
    check(saved['phase'] == 'components' and saved['schedule_profile'] == 'a4-v1'
          and saved['scheduled_windows'] == saved['optimizer_updates'] == 188, 'initial complete-window state')
    ownership = []
    names = set()
    for group in saved['ownership']:
        entries = [p for p in group['parameters'] if receiver_parameter(group['name'], p['name'])]
        for entry in entries:
            name = entry['name']
            check(name not in names and saved['weights'][name] == {k: entry[k] for k in ('shape', 'dtype')},
                  'initial owner duplicate/shape drift')
            check(entry['dtype'] == 'torch.float32' and not name.startswith('reference.')
                  and name != 'codec.budget_row' and not name.endswith(('.R_A', '.R_B')), 'frozen owner in receiver fit')
            names.add(name)
        if entries: ownership.append(dict(group, parameters=entries))
    check(ownership == saved['ownership'], 'initial receiver ownership mismatch')
    return saved, ownership


def prepare(snapshot, input_profile, *, regime='component', root=ROOT):
    spec = profile(input_profile, regime)
    snapshot = Path(snapshot).resolve()
    check(file_hash(snapshot/'model.safetensors-00001-of-00001.safetensors') == BASE_SHA, 'base snapshot identity mismatch')
    metadata = {p.name: file_hash(p) for p in sorted(snapshot.iterdir()) if p.suffix in {'.json', '.jinja'}}
    check({'config.json', 'tokenizer.json', 'tokenizer_config.json', 'chat_template.jinja'} <= set(metadata),
          'complete tokenizer/template/configuration fingerprints required')
    coverage = decode((root/COVERAGE).read_bytes())
    check(coverage['protocol'] == 'redesign-coverage-v1' and coverage['releases'] ==
          {'training': TRAIN_SHA, 'development': DEV_SHA}, 'coverage/release identity')
    raw = {}
    for relative, sha in coverage['artifacts'].items():
        path = (root/relative).resolve()
        check(path.is_relative_to(root/'data') and file_hash(path) == sha, 'stale coverage artifact')
        raw[path.name] = path
    panels = decode(raw['panels.json'].read_bytes())['panels']
    inventory = {row['id']: row for row in (decode(line) for line in raw['inventory.jsonl'].read_bytes().splitlines())}
    train = load_release(root/'data/training-collections-20261105/releases/train', expected_hash=TRAIN_SHA)
    dev = load_release(root/'data/training-collections-20261105/releases/development', expected_hash=DEV_SHA)
    check(set(panels['train_receiver']) == {key for key in train.selected_ids
          if train.record(key)['training_category'] == 'receiver'}, 'complete receiver coverage required')
    panel_bindings = {}
    checked_views = set()
    for name, keys in panels.items():
        release = train if name == 'train_receiver' or name.startswith('train_receiver/') else dev
        for key in keys:
            row = release.record(key)
            check(inventory[key]['record_sha256'] == digest(canonical(row))
                  and inventory[key]['family'] == row['family'], 'stale panel row')
            if key not in checked_views:
                view, provenance, _ = component_view(release, row, intervention=release.purpose == 'development')
                check(digest(view.materialize(provenance)[1]) == inventory[key]['permitted_input']['sha256'],
                      'stale permitted input identity')
                checked_views.add(key)
        check(digest(canonical(keys)) == coverage['coverage'][name]['ids_sha256'], 'panel order drift')
        panel_bindings[name] = {'ids': keys, 'ids_sha256': digest(canonical(keys))}
    selected = [p['id'] for p in probe_panel(train)] if input_profile == LEGACY else panels['train_receiver']
    passes = 8 if input_profile == LEGACY else 4  # Recorded probe / proposed finite RD-08 exposure.
    order = selected*passes
    check(len(order) % 8 == 0, 'complete eight-example windows; never pad or drop rows')
    saved, owners = initial_state(root)
    diagnostic = 'data/g1-proof/gate-repair-v2-20261105/diagnostic.safetensors'
    check(file_hash(root/diagnostic) == DIAGNOSTIC_SHA, 'diagnostic initialization drift')
    qev = decode((root/'data/qev/release_manifest.json').read_bytes())
    qev_files = {name: file_hash(root/'data/qev'/name) for name in
                 ('adapter/adapter_model.safetensors', 'adapter/adapter_config.json', 'pointer.safetensors', 'qev_config.json')}
    check(all(qev['files'][name]['sha256'] == sha for name, sha in qev_files.items()), 'Qev initialization artifact drift')
    sources = plan_sources(root)
    dependencies = {'base_sha256': BASE_SHA, 'snapshot_configuration': metadata,
        'reference': REFERENCE, 'precision': saved['identities']['precision'],
        'qev': saved['identities']['qev'], 'qev_files': qev_files,
        'diagnostic': {'path': diagnostic, 'sha256': DIAGNOSTIC_SHA}, 'sources': sources,
        'historical_numeric_evidence': {name: file_hash(root/name) for name in
            ('docs/evidence/codec-feedback-v3-summary.json', 'docs/evidence/codec-1024-v1-summary.json')}}
    contract = {'protocol': PROTOCOL, 'input_profile': input_profile, 'profile': spec, 'regime': regime,
        'gpu_uuid': PRIMARY, 'snapshot': str(snapshot), 'releases': coverage['releases'],
        'coverage_sha256': file_hash(root/COVERAGE), 'panels': panel_bindings,
        'initial_checkpoint': {'path': str(CHECKPOINT), 'manifest_sha256': CHECKPOINT_SHA,
            'files': saved['files'], 'identities': saved['identities'],
            'scheduled_windows': 188, 'optimizer_updates': 188, 'schedule_profile': 'a4-v1',
            'restore': 'exact named masters, Adam moments, CUDA/torch/Python RNG; preserve schedule position'},
        'owners': owners, 'frozen_owners': ['W', 'R', 'budget_row', 'reference'],
        'profile_only_owners': [] if input_profile != 'direct-role-states-v1' else
            [{'status': 'pending RD-04', 'owner': 'live identity-initialized FP32 role residual projection',
              'parameter_names_shapes': None, 'initialization': 'not yet constructed; no fabricated weights/moments'}],
        'objectives': {k: COEFFICIENTS[k] for k in ('status', 'noul', 'choice', 'score', 'feasibility', 'alignment', 'structure')},
        'objective_masks': 'existing decision_losses Eligibility; Reason only after actual non-assessed status; no new auxiliary loss',
        'dependencies': dependencies, 'training_order': order, 'exposures': {'passes': passes,
            'distinct_records': len(selected), 'presentations': len(order), 'windows': len(order)//8,
            'examples_per_window': 8, 'failure_policy': 'failed attempts remain scheduled; no replacement/retry'},
        'required_evidence': {'RD-01': file_hash(root/COVERAGE),
            'pending_slices': spec['required_slices'], 'native_regime': 'unsupported until RD-06',
            'new_profile_gpu_qualification': 'pending RD-07' if input_profile != LEGACY else 'historical legacy path only'},
        'authorization': 'preparation only; this manifest grants no model execution, fitting or promotion'}
    run_sha = digest(canonical(contract))
    identities = dict(saved['identities'], protocol=run_sha, sources=digest(canonical(sources)))
    return {'contract': contract, 'run_sha256': run_sha, 'checkpoint_identities': identities,
            'status': 'PREPARED_NOT_EXECUTED'}


def validate_plan(plan):
    check(plan['status'] == 'PREPARED_NOT_EXECUTED' and digest(canonical(plan['contract'])) == plan['run_sha256'],
          'prepared plan identity drift')
    contract = plan['contract']
    check(contract['profile'] == profile(contract['input_profile'], contract['regime']), 'prepared profile definition drift')
    check(contract['gpu_uuid'] == PRIMARY and plan['checkpoint_identities'] == dict(
        contract['initial_checkpoint']['identities'], protocol=plan['run_sha256'],
        sources=digest(canonical(contract['dependencies']['sources']))), 'prepared profile/checkpoint identity')


def load_plan(path, snapshot, input_profile, regime='component'):
    plan = decode(Path(path).read_bytes()); validate_plan(plan)
    check(plan == prepare(snapshot, input_profile, regime=regime), 'stale profile/release/dependency/checkpoint plan')
    return plan


def write_preparation(output, plan, timing):
    validate_plan(plan)
    output = Path(output).resolve()
    check(output.is_relative_to(ROOT/'data') and not output.exists(), 'fresh local data output required')
    output.mkdir(parents=True)
    (output/'plan.json').write_bytes(canonical(plan))
    receipt = {'protocol': PROTOCOL, 'status': 'PREPARED_NOT_EXECUTED', 'run_sha256': plan['run_sha256'],
        'plan_sha256': file_hash(output/'plan.json'), 'accounting': empty_accounting(), 'host': timing,
        'scope': 'host preparation includes byte hashing only; not GPU timing or execution'}
    (output/'preparation.json').write_bytes(canonical(receipt))
    return receipt


def empty_accounting():
    return {'scheduled_training_attempts': 0, 'failed_training_attempts': 0,
            'offline_source_encodings': 0, 'offline_reference_encodings': 0, 'sessions': 0, 'prefills': 0,
            'causal_positions': 0, 'optimizer_updates': 0}


def cache_identity(plan, kind, input_sha256, mapping_sha256):
    validate_plan(plan); contract = plan['contract']; dependencies = contract['dependencies']
    check(kind in {'source', 'reference'}, 'only frozen source/reference caches; never live consumer features')
    check(kind != 'source' or contract['profile']['source_policy'] == 'frozen-w-source-v1',
          'mutable D-dependent source cannot be cached')
    check(kind != 'reference' or contract['profile']['roles'] == 'frozen-reference-pooled-v1',
          'direct role states cannot use a pooled reference cache')
    frozen = {'base': dependencies['base_sha256'], 'configuration': dependencies['snapshot_configuration'],
              'precision': dependencies['precision'], 'sources': dependencies['sources']}
    if kind == 'reference': frozen['reference'] = dependencies['reference']
    return {'profile': contract['input_profile'], 'regime': contract['regime'], 'kind': kind,
            'policy': contract['profile']['source_policy'] if kind == 'source' else contract['profile']['roles'],
            'frozen': frozen, 'input_sha256': _sha(input_sha256), 'mapping_sha256': _sha(mapping_sha256)}


def validate_cache(plan, incoming, *, kind, input_sha256, mapping_sha256):
    check(incoming == cache_identity(plan, kind, input_sha256, mapping_sha256), 'incompatible cache identity')


def sampler_state(plan, cursor, receipts):
    validate_plan(plan); contract = plan['contract']; order = contract['training_order']
    check(type(cursor) is int and 0 <= cursor <= len(order) and cursor % 8 == 0, 'complete-window sampler cursor')
    check(receipts['exposures'] == dict(Counter(order[:cursor])), 'scheduled exposure receipts mismatch')
    check(all(type(v) is int and 0 <= v <= cursor for v in receipts['eligible'].values())
          and all(type(v) is int and v >= 0 for v in receipts['failures'].values())
          and sum(receipts['failures'].values()) <= cursor, 'invalid eligibility/failure receipts')
    return {'protocol': PROTOCOL, 'run_sha256': plan['run_sha256'], 'cursor': cursor,
            'order_sha256': digest(canonical(order)), 'receipts': receipts}


def inspect_resume(plan, path, expected_hash):
    validate_plan(plan)
    check(file_hash(Path(path)/'manifest.json') == _sha(expected_hash), 'resume checkpoint manifest drift')
    saved = inspect(path, plan['checkpoint_identities']); contract = plan['contract']
    check(saved['ownership'] == contract['owners'] and saved['phase'] == 'components'
          and saved['schedule_profile'] == 'a4-v1' and saved['reference_frozen']
          and saved['reference_id'] == contract['dependencies']['reference'], 'resume owner/phase/reference mismatch')
    sampler = saved['sampler']
    check(sampler == sampler_state(plan, sampler['cursor'], sampler['receipts']), 'resume sampler identity mismatch')
    windows = sampler['cursor']//8
    check(type(saved['scheduled_windows']) is int and type(saved['optimizer_updates']) is int
          and saved['scheduled_windows'] == contract['initial_checkpoint']['scheduled_windows']+windows
          and contract['initial_checkpoint']['optimizer_updates'] <= saved['optimizer_updates'] <=
              contract['initial_checkpoint']['optimizer_updates']+windows, 'resume scheduled/update counters mismatch')
    return saved


def restore_prepared(plan, path, expected_hash, backbone, codec, reference, window):
    """Use the existing transactional CUDA restore, after model-free admission."""
    inspect_resume(plan, path, expected_hash)
    from .checkpoint import restore_model
    return restore_model(path, backbone, codec, reference, plan['checkpoint_identities'], window=window)


def admit_execution(plan, *, gpu, input_profile, regime):
    validate_plan(plan)
    check(gpu == PRIMARY, '3090 UUID required; never CPU/3060')
    check(input_profile == plan['contract']['input_profile'] and regime == plan['contract']['regime'],
          'execution profile/regime mismatch')
    check(plan['contract']['profile']['implementation'] == 'legacy',
          'profile implementation/qualification pending RD-03..07; preparation is not execution authorization')


def prepare_to(output, snapshot, input_profile, regime):
    meter = Recorder()
    with meter.span('prepare'):
        plan = prepare(snapshot, input_profile, regime=regime)
    return write_preparation(output, plan, meter.snapshot())
