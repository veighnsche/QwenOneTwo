"""Explicitly executed read-only child. Importing this file does no GPU work."""
import os
from pathlib import Path
import sys
import time
import traceback
from ..g1_device import diagnostic_device,require_cuda
from .inputs import check,digest
from .manifest import BASE_SHA,DIAGNOSTIC_SHA,file_hash
from .two_gpu import ROOT,CHECKPOINT,PROFILE,RECEIVER,REFERENCE


def worker(connection, uuid, prepared, execute_gpu=False):
    try:
        check(execute_gpu is True,'GPU execution must be explicitly requested')
        check('torch' not in sys.modules,'set worker CUDA visibility before model runtime import')
        os.environ['CUDA_VISIBLE_DEVICES']=uuid
        os.chdir(ROOT)
        with diagnostic_device(uuid,execute_gpu=True):
            import torch
            from torch.nn.attention import sdpa_kernel,SDPBackend
            from ..g1_proof import configure
            from ..g1_evidence import frozen_check
            from ..codec.qualify import load
            from .reference import Reference
            from .receiver_learning import groups
            from .checkpoint import restore_model
            from .behavior_qualification import receiver_hash,evaluate_case
            from .release import load_release,decode
            from .clean import CleanEncoder
            from .two_gpu import DEV_SHA,source_hashes
            check(prepared['sources']==source_hashes(),'worker source drift')
            require_cuda(torch.cuda);check(torch.cuda.device_count()==1,'one visible GPU per process')
            configure();torch.set_grad_enabled(False)
            snapshot=Path(prepared['snapshot']);initial=ROOT/'data/g1-proof/gate-repair-v2-20261105/diagnostic.safetensors'
            check(file_hash(snapshot/'model.safetensors-00001-of-00001.safetensors')==BASE_SHA,'base file hash')
            check(file_hash(initial)==DIAGNOSTIC_SHA,'initial diagnostic hash')
            checkpoint=ROOT/CHECKPOINT
            check(file_hash(checkpoint/'manifest.json')==prepared['checkpoint_manifest_sha256'],'checkpoint metadata drift')
            manifest=decode((checkpoint/'manifest.json').read_bytes())
            release=load_release(ROOT/'data/training-collections-20261105/releases/development',expected_hash=DEV_SHA)
            with sdpa_kernel(SDPBackend.MATH),torch.no_grad():
                b,c,_=load(snapshot,initial)
                ref=Reference(b,c,pooler_kind='target-v2');ref.freeze();groups(b,c,ref)
                restore_model(checkpoint,b,c,ref,manifest['identities'])
                b.requires_grad_(False);c.requires_grad_(False)
                check(b.compensated and b.precision_parts==2 and b.feedback_precision=='fp32'
                      and b.recurrence_backend=='graph-v1','v3 arithmetic required')
                def integrity():
                    base=frozen_check(b,snapshot)
                    check(base['pass'] and base['base_tied_storage'],'frozen tied W mismatch')
                    check(receiver_hash(b,c)==RECEIVER and ref.fingerprint()==REFERENCE==ref.artifact_id,'receiver/reference changed')
                    check(all(not p.requires_grad and p.grad is None for m in (b,c,ref) for p in m.parameters()),'read-only model ownership')
                    return {'uuid':uuid,'device_name':torch.cuda.get_device_name(0),'base_frozen':True,'base_check':base,
                            'receiver':RECEIVER,'reference':REFERENCE,'gradients_none':True,'optimizer_updates':0,
                            'torch_version':torch.__version__,'cuda_runtime':torch.version.cuda,
                            'profile':PROFILE,'accelerator_qualification_granted':False}
                initial_integrity=integrity()
                encoder=CleanEncoder(b,ref,checkpoint_hash=RECEIVER,profile_hash=digest(PROFILE.encode()),cache_references=True)
                def evaluated(key):return evaluate_case(b,c,ref,release,release.record(key),encoder)
                connection.send({'kind':'ready',**initial_integrity,'anchors':[evaluated(k) for k in prepared['anchors']]})
                while True:
                    message=connection.recv()
                    if message['kind']=='stop':break
                    check(message['kind']=='job','unknown worker command')
                    start=time.monotonic();cases=[evaluated(k) for k in message['job']['ids']]
                    connection.send({'kind':'result','index':message['index'],'cases':cases,
                                     'seconds':time.monotonic()-start})
                connection.send({'kind':'final',**integrity()})
    except BaseException:
        try:connection.send({'kind':'error','error':traceback.format_exc()})
        except (BrokenPipeError,EOFError):pass
        raise
    finally:connection.close()
