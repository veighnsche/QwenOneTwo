"""Model-free artifact/profile pinning; reads local files, never downloads."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
from .inputs import check, canonical

BASE_SHA = '04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696'
DIAGNOSTIC_SHA = '1dd2e8527e14caa9002d672b4187253cdc174fb3c204ab56e80f257f23943640'
OWNERS = ('architecture-and-training-plan.md','latent-packet-codec.md','typed-value-abi.md',
          'neural-input-encoding-v1.md','control-flow-v1.md','position-limits-and-budget-accounting.md',
          'numeric-profile-v1.md','numeric-verification.md','codec-feedback-precision-v3.md',
          'codec-1024-protocol-v1.md','training-input-rules-v1.md','clean-test-construction-v1.md',
          'module-and-optimizer-contract-v1.md','training-run-profile-v1.md',
          'loss-configuration-and-training-schedule.md','scalar-row-handling.md','data-operations.md',
          'reviewed-release-format-v1.md')


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def build(root, snapshot, checkpoint):
    root,snapshot,checkpoint = Path(root),Path(snapshot),Path(checkpoint)
    base = snapshot/'model.safetensors-00001-of-00001.safetensors'
    check(file_hash(base) == BASE_SHA, 'pinned base weights mismatch')
    check(file_hash(checkpoint) == DIAGNOSTIC_SHA, 'pinned diagnostic weights mismatch')
    qev = root/'data/qev'
    upstream = json.loads((qev/'upstream.json').read_text())
    check(upstream == {'repo':'twainsk/qev-0.8b','revision':'1d2a6d7f891daab7483a49c3abe507c0c9d7ea62'}, 'Qev upstream mismatch')
    release = json.loads((qev/'release_manifest.json').read_text())
    qev_files = ('adapter/adapter_model.safetensors','adapter/adapter_config.json','pointer.safetensors','qev_config.json')
    hashes = {name:file_hash(qev/name) for name in qev_files}
    check(all(sha == release['files'][name]['sha256'] for name,sha in hashes.items()), 'Qev release file mismatch')
    sources = sorted(set((root/'qwen_one_two/codec').glob('*.py')) |
                     set((root/'qwen_one_two/training').glob('*.py')) |
                     set((root/'qwen_one_two').glob('g1_*.py')) | {root/'qwen_one_two/decision.py'})
    packages = {name:importlib.metadata.version(name) for name in ('torch','transformers','safetensors','peft','accelerate')}
    return {'format':'qot-backend-profile-v1','scope':'engineering implementation/diagnostic identity; no capability promotion',
            'base':{'revision':'2fc06364715b967f1860aea9cf38778875588b17','weights_sha256':BASE_SHA,
                    'configuration':{p.name:file_hash(p) for p in snapshot.iterdir() if p.suffix in {'.json','.jinja'}}},
            'qev':dict(upstream,files=hashes),'diagnostic_sha256':DIAGNOSTIC_SHA,
            'gpu_uuid':'GPU-225509c3-d702-6c27-de03-76b5e99b33b1',
            'precision':{'frozen_storage':'BF16','transformer_computation':'FP32',
                         'vocabulary':'two-part compensated BF16','reference':'genuine FP32'},
            'protocol':'v3-1024-v1','packages':packages,
            'source_hashes':{str(p.relative_to(root)):file_hash(p) for p in sources},
            'contract_hashes':{name:file_hash(root/'docs'/name) for name in OWNERS}}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--snapshot',required=True);parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[2]
    manifest=build(root,args.snapshot,args.checkpoint)
    with Path(args.output).open('xb') as stream:
        stream.write(json.dumps(manifest,indent=2,sort_keys=True).encode()+b'\n')
    print(json.dumps({'output':args.output,'sha256':file_hash(args.output)}))


if __name__ == '__main__':main()
