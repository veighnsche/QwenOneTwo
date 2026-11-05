"""Artifact integrity checks using byte files only; no model computation."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from qwen_one_two.decision import MODEL,REVISION,verify_qev


class ArtifactTests(unittest.TestCase):
    def test_qev_exact_backbone_and_checksums(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);(p/'adapter').mkdir()
            config={'base_model':MODEL,'base_revision':REVISION,'hidden_size':1024,'native_generation':'adapter_disabled'}
            (p/'qev_config.json').write_text(json.dumps(config))
            for name in ('adapter/adapter_config.json','adapter/adapter_model.safetensors','pointer.safetensors'):(p/name).write_bytes(b'not model weights')
            names=('qev_config.json','adapter/adapter_config.json','adapter/adapter_model.safetensors','pointer.safetensors')
            manifest={'files':{n:{'sha256':hashlib.sha256((p/n).read_bytes()).hexdigest()} for n in names}}
            (p/'release_manifest.json').write_text(json.dumps(manifest))
            self.assertEqual(verify_qev(p)['hidden_size'],1024)
            (p/'pointer.safetensors').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'checksum mismatch'):verify_qev(p)
            config['base_model']='Qwen/Qwen3.5-9B';(p/'qev_config.json').write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError,'does not match'):verify_qev(p)
