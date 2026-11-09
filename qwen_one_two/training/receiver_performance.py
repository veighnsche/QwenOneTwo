"""Explicit receiver execution policy. Importing this module executes no model."""
from contextlib import contextmanager
from pathlib import Path
from .inputs import check
from .manifest import file_hash
from .release import decode
from .host_metrics import fingerprint_transfers

LEGACY='checkpointed-v1'
RETAINED='retained-activations-v1'
PROFILES=(LEGACY,RETAINED)
GPU_CHECKS=('PERF-01-packed-hash-parity','PERF-02-retained-joint-update-resume',
            'PERF-03-matched-throughput-memory')
EVIDENCE='docs/evidence/receiver-performance-v1.json'


def policy(name):
    check(name in PROFILES,'unknown receiver execution policy')
    return {'name':name,'activation_checkpointing':name==LEGACY,
            'recurrence_cache_entries':8 if name==LEGACY else 32,
            'fingerprint_batch_bytes':0 if name==LEGACY else 32*1024*1024,
            'episode_graphs_retained':1,'component_VJPs':'unchanged separate gradients',
            'optimizer_and_objectives':'unchanged','automatic_OOM_fallback':False}


def require_matching_policy(peer_plan,name):
    check(peer_plan.get('execution_policy',policy(LEGACY))==policy(name),
          'matched peer receiver execution policy mismatch')


def require_qualification(contract,root=None):
    """Fail closed before model loading; historical evidence cannot qualify this."""
    p=contract.get('execution_policy')
    if p is not None:
        check(p==policy(p['name']),'receiver execution policy drift')
        if p['name']==LEGACY:return
        check(root is not None and (Path(root)/EVIDENCE).is_file(),
              'receiver performance GPU qualification pending PERF-01..03')
        path=Path(root)/EVIDENCE;evidence=decode(path.read_bytes())
        check(evidence['status']=='PASS' and evidence['execution_policy']==p and
              evidence['checks']=={name:'PASS' for name in GPU_CHECKS},
              'receiver performance GPU qualification pending PERF-01..03')
        check(evidence['source_hashes']==contract['dependencies']['sources'] and
              file_hash(path)==contract['required_evidence']['performance_qualification_sha256'],
              'receiver performance qualification dependency drift')


@contextmanager
def execution_scope(backbone,name):
    """Only a dedicated, admitted runner may select this; restore on failure.

    The production default stays checkpointed. Retention changes storage and
    recomputation only: one completed episode still releases its graph before
    the next. No padding, combined losses, altered checks or optimizer settings.
    """
    from ..codec import recurrence
    p=policy(name);old=(backbone.checkpointing,recurrence.CACHE_LIMIT)
    recurrence.clear_graph_cache();recurrence.CACHE_LIMIT=p['recurrence_cache_entries']
    backbone.checkpointing=p['activation_checkpointing']
    try:
        with fingerprint_transfers(p['fingerprint_batch_bytes']):yield p
    finally:
        recurrence.clear_graph_cache();recurrence.CACHE_LIMIT=old[1]
        backbone.checkpointing=old[0]
