"""Model-free identity and opaque-byte checks shared with CUDA checkpoint loading."""
from pathlib import Path

from .inputs import check
from .manifest import file_hash
from .release import decode

FORMAT = 'qot-training-bundle-v1'
IDENTITIES = {'base', 'qev', 'protocol', 'precision', 'reference', 'dataset', 'sources'}


def validate_identities(identities):
    check(set(identities) == IDENTITIES, 'bundle identity fields')
    check(all(isinstance(v, str) and len(v) == 64 and all(c in '0123456789abcdef' for c in v)
              for v in identities.values()), 'bundle identity hashes')


def inspect(path, expected_identities):
    validate_identities(expected_identities)
    path = Path(path)
    manifest = decode((path/'manifest.json').read_bytes())
    check(manifest['format'] == FORMAT and manifest['identities'] == expected_identities, 'incompatible bundle')
    check({'weights.safetensors', 'optimizer.pt', 'rng.pt'} <= set(manifest['files']), 'incomplete bundle')
    check(set(manifest['files']) <= {'weights.safetensors', 'effective-rows.safetensors', 'optimizer.pt', 'rng.pt'}, 'unexpected bundle files')
    for name, sha in manifest['files'].items():
        check(file_hash(path/name) == sha, 'artifact hash mismatch: '+name)
    return manifest
