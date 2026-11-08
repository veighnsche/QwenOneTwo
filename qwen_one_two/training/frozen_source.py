"""Offline source policy and cache identities. Runtime tensor imports are lazy."""
from pathlib import Path

from ..g1_device import PRIMARY
from .inputs import canonical, check, digest
from .manifest import BASE_SHA, file_hash

D_SOURCE = 'current-d-source-v1'
FROZEN_SOURCE = 'frozen-w-source-v1'
GPU_CHECKS = {
    'RD03-GPU-D-INDEPENDENCE': 'test_frozen_features_ignore_D_updates',
    'RD03-GPU-CACHE-PARITY': 'test_cache_matches_uncached_and_preserves_bindings',
    'RD03-GPU-CONSUMER-GRADIENTS': 'test_consumer_gradients_and_frozen_W',
}


def source_mode(policy, *, cache=False, regime='offline_component'):
    check(regime == 'offline_component', 'source encoding is offline-only; never re-encode native states')
    check(policy in {D_SOURCE, FROZEN_SOURCE}, 'unknown source policy')
    check(not cache or policy == FROZEN_SOURCE, 'mutable D sources cannot be cached')
    return 'D' if policy == D_SOURCE else 'W'


def validate_arithmetic(backbone):
    check(backbone.compensated and backbone.fp32_residual and backbone.feedback_precision == 'fp32'
          and backbone.precision_parts == 2 and backbone.temporal_chunk == 32
          and backbone.recurrence_backend in {'eager', 'graph-v1'}
          and backbone.attention_block_backend in {'eager', 'graph-v1'}
          and not backbone.round_reference_inputs, 'frozen source requires qualified v3 arithmetic')


def _hash(value):
    check(isinstance(value,str) and len(value)==64 and all(c in '0123456789abcdef' for c in value),
          'complete frozen dependency hash required')
    return value


def fingerprint(*, base, live_w, configuration, tokenizer, arithmetic, implementation):
    check({'config.json','tokenizer.json','tokenizer_config.json','chat_template.jinja'} <= set(configuration),
          'complete source tokenizer/template/configuration required')
    check(bool(implementation) and bool(arithmetic), 'source implementation/arithmetic fingerprint required')
    for sha in [base, live_w, tokenizer, *configuration.values(), *implementation.values()]: _hash(sha)
    return {'policy':FROZEN_SOURCE,'base':base,'live_w':live_w,'configuration':dict(configuration),
            'tokenizer':tokenizer,'arithmetic':dict(arithmetic),'implementation':dict(implementation)}


def source_key(identity, excerpt, ids, selected, offsets):
    check(set(identity)=={'policy','base','live_w','configuration','tokenizer','arithmetic','implementation'},
          'unexpected source cache dependency')
    check(identity['policy']==FROZEN_SOURCE, 'source cache requires frozen W')
    check(identity==fingerprint(**{k:v for k,v in identity.items() if k!='policy'}), 'frozen source fingerprint drift')
    check(isinstance(excerpt,bytes) and bool(excerpt) and len(ids)<=512, 'complete permitted source required; never crop')
    check(all(type(i) is int and i>=0 for i in ids) and all(type(i) is int for i in selected)
          and all(type(a) is int and type(b) is int for a,b in offsets), 'source token/byte mapping types')
    check(len(selected)==len(offsets)>0 and selected==list(range(selected[0],selected[-1]+1))
          and 0<=selected[0]<=selected[-1]<len(ids), 'complete source token selection required')
    check(offsets[0][0]==0 and offsets[-1][1]==len(excerpt)
          and all(0<=a<b<=len(excerpt) for a,b in offsets)
          and all(offsets[i-1][1]==offsets[i][0] for i in range(1,len(offsets))), 'complete source byte mapping required')
    for a,b in offsets:
        excerpt[:a].decode('utf-8');excerpt[:b].decode('utf-8')
    # Parent identity/hidden bytes stay in the fresh external audit. The cache
    # stores only derived content states; it never stores registries or pointers.
    return digest(canonical({'frozen':identity,'input':digest(excerpt),'ids':ids,
                             'selected':selected,'offsets':offsets,'limit':512}))


def _w_parameters(backbone):
    return {n:p for n,p in backbone.named_parameters() if not n.endswith(('.R_A','.R_B','.D_A','.D_B'))}


def _stamp(parameters):
    return tuple((n,id(p),p._version,str(p.dtype),str(p.device),p.data_ptr(),p.requires_grad)
                 for n,p in parameters.items())


def _tokenizer_hash(tokenizer):
    return digest(canonical({'backend':tokenizer.backend_tokenizer.to_str(),
        'template':tokenizer.chat_template,'special_tokens':tokenizer.special_tokens_map,
        'padding_side':tokenizer.padding_side,'truncation_side':tokenizer.truncation_side,
        'model_max_length':tokenizer.model_max_length}))


def runtime_arithmetic(backbone):
    import torch
    import importlib.metadata
    validate_arithmetic(backbone)
    check(backbone.precision==torch.bfloat16, 'frozen source requires original BF16 W storage')
    settings = {'feedback_precision':backbone.feedback_precision,'parts':backbone.precision_parts,
        'temporal_chunk':backbone.temporal_chunk,'recurrence':backbone.recurrence_backend,
        'attention_block':backbone.attention_block_backend,'round_reference_inputs':backbone.round_reference_inputs,
        'model_configuration':digest(backbone.model.config.to_json_string().encode()),
        'torch':str(torch.__version__),'torch_git':torch.version.git_version,'cuda_runtime':torch.version.cuda,
        'packages':{name:importlib.metadata.version(name) for name in ('transformers','tokenizers','safetensors')},
        'fp32':torch.backends.fp32_precision,'matmul':torch.backends.cuda.matmul.fp32_precision,
        'convolution':torch.backends.cudnn.conv.fp32_precision}
    check(all(settings[k]=='ieee' for k in ('fp32','matmul','convolution')), 'frozen source requires IEEE FP32')
    return settings


def bind_frozen_source(backbone):
    """Verify W once per unchanged resident instance; no adapter substitution."""
    import os
    check(os.environ.get('CUDA_VISIBLE_DEVICES')==PRIMARY, 'frozen source is 3090-only')
    import torch
    from ..g1_evidence import frozen_check
    from .reference import state_fingerprint
    parameters=_w_parameters(backbone)
    check(len(parameters)==320 and all(p.is_cuda and not p.requires_grad for p in parameters.values()),
          'all 320 W tensors must remain frozen on CUDA')
    frozen=dict(parameters,**{'buffer.'+n:t for n,t in backbone.named_buffers()})
    check(all(t.is_cuda and not t.requires_grad for t in frozen.values()), 'source buffers must be frozen CUDA tensors')
    stamp=_stamp(frozen);arithmetic=runtime_arithmetic(backbone)
    tokenizer=_tokenizer_hash(backbone.tokenizer)
    previous=getattr(backbone,'_offline_frozen_source',None)
    if previous is not None:
        check(digest(canonical(previous['identity']))==previous['identity_sha256'], 'frozen source fingerprint mutation')
        check(previous['stamp']==stamp, 'frozen W storage/version/ownership changed')
        check(previous['identity']['tokenizer']==tokenizer and previous['identity']['arithmetic']==arithmetic,
              'frozen tokenizer/arithmetic changed')
        return previous
    snapshot=Path(backbone.snapshot_path)
    check(file_hash(snapshot/'model.safetensors-00001-of-00001.safetensors')==BASE_SHA, 'source W file identity')
    verified=frozen_check(backbone,snapshot)
    check(verified['pass'] and verified['base_tied_storage'], 'source W/tie differs from pinned snapshot')
    configuration={p.name:file_hash(p) for p in sorted(snapshot.iterdir()) if p.suffix in {'.json','.jinja'}}
    root=Path(__file__).resolve().parents[2]
    names=['qwen_one_two/g1_backbone.py','qwen_one_two/g1_precision_math.py','qwen_one_two/g1_device.py',
           'qwen_one_two/codec/backbone.py','qwen_one_two/codec/precision.py','qwen_one_two/codec/recurrence.py',
           'qwen_one_two/codec/attention_block.py','qwen_one_two/codec/finite.py',
           'qwen_one_two/training/clean.py','qwen_one_two/training/frozen_source.py']
    import inspect
    implementation={n:file_hash(root/n) for n in names}
    implementation.update({str(path):file_hash(path) for path in
        sorted({Path(inspect.getfile(type(module))).resolve() for module in backbone.model.modules()})})
    identity=fingerprint(base=BASE_SHA,live_w=state_fingerprint(frozen),configuration=configuration,
                         tokenizer=tokenizer,arithmetic=arithmetic,implementation=implementation)
    binding={'identity':identity,'identity_sha256':digest(canonical(identity)),'stamp':stamp,'verified_W':verified}
    # Ordinary metadata attribute, never a module, parameter owner or tensor cache.
    object.__setattr__(backbone,'_offline_frozen_source',binding)
    return binding
