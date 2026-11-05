"""Versioned small-weight export/resume, with no base-weight duplication.

Artifacts stay in caller-selected local data directories. Loading does not
promote, deploy or claim semantic approval of any checkpoint.
"""
import json
import os
from pathlib import Path
import random
import shutil
import tempfile
import torch
from safetensors.torch import load_file, save_file
from ..codec.neural import cuda_only
from .engine import PHASES, clone_state, tensors
from .inputs import canonical, digest, check

FORMAT = 'qot-training-bundle-v1'
IDENTITIES = {'base', 'qev', 'protocol', 'precision', 'reference', 'dataset', 'sources'}


def bundle_tensors(backbone, codec, reference):
    result = {name:p for name,p in backbone.named_parameters() if p.requires_grad}
    result.update({'codec.'+name:t for name,t in codec.state_dict(keep_vars=True).items()})
    result.update({'reference.'+name:t for name,t in reference.state_dict(keep_vars=True).items()})
    return result


def validate_identities(identities):
    check(set(identities) == IDENTITIES, 'bundle identity fields')
    check(all(isinstance(v,str) and len(v) == 64 and all(c in '0123456789abcdef' for c in v)
              for v in identities.values()), 'bundle identity hashes')


def save(path, window, weights, identities, sampler, *, reference_frozen, reference_id):
    cuda_only(); validate_identities(identities)
    check(window.examples == 0 and not window.buffers, 'save only at complete window boundary')
    check(all(p.grad is None for p in window.parameters), 'outstanding parameter gradients')
    check(set(window.names) <= set(weights), 'optimizer owner absent from export')
    check(all(weights[n] is p for n,p in zip(window.names,window.parameters)), 'export/optimizer owner mismatch')
    check(all(t.is_cuda and bool(t.isfinite().all()) for t in weights.values()), 'export requires finite CUDA tensors')
    check(reference_frozen or window.phase == 'reference', 'live export requires frozen reference')
    check(not reference_frozen or reference_id == identities['reference'], 'reference identity mismatch')
    # Validate sampler as JSON, never pickle arbitrary application objects.
    sampler = json.loads(canonical(sampler))
    path = Path(path)
    check(not path.exists(), 'never overwrite a checkpoint')
    path.parent.mkdir(parents=True,exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix='.qot-checkpoint-',dir=path.parent))
    try:
        save_file({n:t.detach().contiguous().clone() for n,t in weights.items()},str(temp/'weights.safetensors'))
        rows = {n:weights[n].detach().to(torch.bfloat16).contiguous() for n in ('codec.budget_row','codec.score_row') if n in weights}
        if rows:
            save_file(rows,str(temp/'effective-rows.safetensors'))
        torch.save(window.optimizer.state_dict(),temp/'optimizer.pt')
        torch.save({'cuda':torch.cuda.get_rng_state(0),'torch':torch.get_rng_state(),
                    'python':random.getstate()},temp/'rng.pt')
        manifest = {'format':FORMAT,'identities':identities,'ownership':window.ownership,
                    'phase':window.phase,'scheduled_windows':window.scheduled_windows,
                    'optimizer_updates':window.optimizer_updates,'sampler':sampler,
                    'reference_frozen':reference_frozen,'reference_id':reference_id,
                    'weights':{n:{'shape':list(t.shape),'dtype':str(t.dtype)} for n,t in weights.items()},
                    'files':{p.name:digest(p.read_bytes()) for p in temp.iterdir()},
                    'capability_promoted':False}
        (temp/'manifest.json').write_bytes(canonical(manifest))
        os.rename(temp,path)
    except BaseException:
        shutil.rmtree(temp,ignore_errors=True)
        raise
    return digest((path/'manifest.json').read_bytes())


def inspect(path, expected_identities):
    validate_identities(expected_identities)
    path = Path(path)
    manifest = json.loads((path/'manifest.json').read_bytes())
    check(manifest['format'] == FORMAT and manifest['identities'] == expected_identities, 'incompatible bundle')
    check({'weights.safetensors','optimizer.pt','rng.pt'} <= set(manifest['files']), 'incomplete bundle')
    check(set(manifest['files']) <= {'weights.safetensors','effective-rows.safetensors','optimizer.pt','rng.pt'}, 'unexpected bundle files')
    for name,sha in manifest['files'].items():
        check(digest((path/name).read_bytes()) == sha, 'artifact hash mismatch: '+name)
    return manifest


def _load_weights(path, targets, manifest):
    incoming = load_file(str(Path(path)/'weights.safetensors'),device='cuda')
    check(set(incoming) == set(targets) == set(manifest['weights']), 'bundle tensor names')
    for name,target in targets.items():
        cuda_only(target)
        value = incoming[name]
        check(value.shape == target.shape and value.dtype == target.dtype and bool(value.isfinite().all()), 'tensor shape/dtype/value: '+name)
        check(manifest['weights'][name] == {'shape':list(target.shape),'dtype':str(target.dtype)}, 'tensor manifest drift')
    if 'effective-rows.safetensors' in manifest['files']:
        effective = load_file(str(Path(path)/'effective-rows.safetensors'),device='cuda')
        check(set(effective) == {'codec.budget_row','codec.score_row'}, 'private rows')
        check(all(torch.equal(value,incoming[name].to(torch.bfloat16)) for name,value in effective.items()), 'scalar effective row mismatch')
    return incoming


def load_export(path, targets, expected_identities):
    """Install compatible masters into existing shared-W modules, not base W."""
    cuda_only()
    manifest = inspect(path,expected_identities)
    incoming = _load_weights(path,targets,manifest)
    backup = {n:t.detach().clone() for n,t in targets.items()}
    try:
        with torch.no_grad():
            for n,t in targets.items():t.copy_(incoming[n])
    except BaseException:
        with torch.no_grad():
            for n,t in targets.items():t.copy_(backup[n])
        raise
    return manifest


def resume(path, window, targets, expected_identities):
    cuda_only()
    check(window.examples == 0 and not window.buffers, 'resume over active window')
    manifest = inspect(path,expected_identities)
    check(manifest['ownership'] == window.ownership, 'optimizer ownership mismatch')
    check(manifest['phase'] == window.phase and 0 <= manifest['scheduled_windows'] <= PHASES[window.phase], 'resume phase/window')
    check(manifest['optimizer_updates'] >= 0, 'optimizer counter')
    check(all(targets[n] is p for n,p in zip(window.names,window.parameters)), 'resume optimizer/weight aliases')
    incoming = _load_weights(path,targets,manifest)
    state = torch.load(Path(path)/'optimizer.pt',map_location='cuda',weights_only=True)
    check(len(state['param_groups']) == len(window.optimizer.param_groups), 'optimizer group count')
    parameter_ids = []
    by_id = {}
    for saved,current in zip(state['param_groups'],window.optimizer.param_groups):
        check(len(saved['params']) == len(current['params']), 'optimizer group shape')
        for key in ('name','peak_lr','weight_decay','betas','eps','foreach','fused','capturable'):
            check(saved[key] == current[key], 'optimizer setting mismatch: '+key)
        for key,p in zip(saved['params'],current['params']):
            parameter_ids.append(key); by_id[key] = p
    check(len(set(parameter_ids)) == len(parameter_ids) and set(state['state']) <= set(parameter_ids), 'optimizer duplicate/foreign state')
    for key,values in state['state'].items():
        check(set(values) == {'step','exp_avg','exp_avg_sq'}, 'AdamW state schema')
        check(values['step'].numel() == 1 and values['step'].item() >= 0, 'AdamW step')
        for name in ('exp_avg','exp_avg_sq'):
            check(values[name].shape == by_id[key].shape and values[name].dtype == torch.float32, 'AdamW moment shape/dtype')
        check(bool((values['exp_avg_sq'] >= 0).all()), 'negative second moment')
    check(all(t.is_cuda and bool(t.isfinite().all()) for t in tensors(state['state'])), 'invalid optimizer tensor')
    rng = torch.load(Path(path)/'rng.pt',map_location='cpu',weights_only=True)
    # CPU byte RNG bookkeeping is not a CPU model workload.
    check(set(rng) == {'cuda','torch','python'}, 'RNG schema')
    backup = {n:t.detach().clone() for n,t in targets.items()}
    old_state = clone_state(window.optimizer.state_dict())
    old_rng = (torch.cuda.get_rng_state(0),torch.get_rng_state(),random.getstate())
    try:
        with torch.no_grad():
            for n,t in targets.items():t.copy_(incoming[n])
        window.optimizer.load_state_dict(state)
        torch.cuda.set_rng_state(rng['cuda'],0); torch.set_rng_state(rng['torch']); random.setstate(rng['python'])
    except BaseException:
        with torch.no_grad():
            for n,t in targets.items():t.copy_(backup[n])
        window.optimizer.load_state_dict(old_state)
        torch.cuda.set_rng_state(old_rng[0],0); torch.set_rng_state(old_rng[1]); random.setstate(old_rng[2])
        raise
    window.scheduled_windows = manifest['scheduled_windows']
    window.optimizer_updates = manifest['optimizer_updates']
    window.optimizer.zero_grad(set_to_none=True)
    return manifest
